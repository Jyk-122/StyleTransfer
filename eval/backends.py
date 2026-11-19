"""Lazy model imports: manifests/reports/API judging do not require GPU libraries."""
from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
from pathlib import Path
import urllib.request

from .common import local_model_identity, read_image, sha256_file
from .qwen_prompt import PROMPT_VERSION, RUBRIC, SYSTEM_PROMPT


class DinoEncoder:
    def __init__(self, args):
        from transformers import AutoImageProcessor, AutoModel

        self.device = args.device
        options = {"revision": args.dino_revision, "local_files_only": args.local_files_only}
        self.processor = AutoImageProcessor.from_pretrained(args.dino_model, use_fast=False, **options)
        self.model = AutoModel.from_pretrained(args.dino_model, **options).float().eval().to(self.device)
        if self.model.config.model_type != "dinov2":
            raise ValueError("Expected a DINOv2 model (e.g. facebook/dinov2-base)")
        self.protocol = {
            "model": args.dino_model, "revision": args.dino_revision,
            "resolved_revision": getattr(self.model.config, "_commit_hash", None),
            "local_identity": local_model_identity(args.dino_model),
            "feature": "last_hidden_state[:, 0] (CLS), L2 normalized; cosine",
            "processor": self.processor.to_dict(), "dtype": "float32", "device": self.device,
        }

    def encode(self, images):
        import torch
        with torch.inference_mode():
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            features = self.model(**inputs).last_hidden_state[:, 0].float()
            return torch.nn.functional.normalize(features, dim=-1).cpu().tolist()


def load_csd_state(path):
    """Official training checkpoint or an exported tensor-only state dictionary."""
    import numpy as np
    import torch

    if Path(path).suffix == ".safetensors":
        from safetensors.torch import load_file
        state = load_file(str(path), device="cpu")
    else:
        # The official checkpoint includes argparse/NumPy training metadata.
        # Keep weights_only=True; never silently fall back to arbitrary unpickling.
        numpy_core = getattr(np, "_core", None) or np.core
        safe_types = [argparse.Namespace, np.dtype, numpy_core.multiarray.scalar,
                      type(np.dtype("float64")), type(np.dtype("float32")), type(np.dtype("int64"))]
        if numpy_core.multiarray.scalar.__module__ != "numpy.core.multiarray":
            safe_types.append((numpy_core.multiarray.scalar, "numpy.core.multiarray.scalar"))
        with torch.serialization.safe_globals(safe_types):
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict):
            raise ValueError("CSD checkpoint must be a dictionary")
        state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    if not isinstance(state, dict):
        raise ValueError("CSD state dictionary is missing")
    normalized = {}
    for name, tensor in state.items():
        key = name.removeprefix("module.")
        if key in normalized:
            raise ValueError(f"Duplicate CSD parameter after prefix removal: {key}")
        normalized[key] = tensor
    return normalized


def build_csd_visual(state):
    """Use the official CLIP visual architecture with CSD's learned STYLE projection.

    This is equivalent to CSD_CLIP.forward's style branch, without downloading a
    second, unused CLIP checkpoint or instantiating its text encoder/content head.
    Architecture and projection contract: https://github.com/learn2phoenix/CSD
    """
    import torch
    from clip.model import VisionTransformer

    if "last_layer_style" not in state:
        raise ValueError("CSD checkpoint lacks last_layer_style; plain CLIP weights are not CSD")
    visual_state = {key[len("backbone."):]: value for key, value in state.items() if key.startswith("backbone.")}
    width, _, patch_size, _ = visual_state["conv1.weight"].shape
    grid = math.isqrt(visual_state["positional_embedding"].shape[0] - 1)
    if grid * grid + 1 != visual_state["positional_embedding"].shape[0]:
        raise ValueError("Unsupported CSD positional embedding shape")
    blocks = {int(key.split(".")[2]) for key in visual_state if key.startswith("transformer.resblocks.")}
    if not blocks or blocks != set(range(len(blocks))) or width % 64:
        raise ValueError("Unsupported CSD visual transformer architecture")
    projection = state["last_layer_style"]
    if projection.ndim != 2 or projection.shape[0] != width:
        raise ValueError("CSD style projection shape mismatch")
    visual = VisionTransformer(grid * patch_size, patch_size, width, len(blocks), width // 64,
                               projection.shape[1])
    # CSD removes the original CLIP projection. Restore only its learned style head.
    if "proj" in visual_state:
        raise ValueError("Unexpected backbone.proj: expected an official CSD checkpoint")
    visual_state["proj"] = projection
    visual.load_state_dict(visual_state, strict=True)
    if not all(torch.isfinite(tensor).all() for tensor in visual.state_dict().values()):
        raise ValueError("Non-finite CSD parameters")
    return visual, grid * patch_size


class CsdEncoder:
    def __init__(self, args):
        from torchvision import transforms

        path = args.csd_weights
        if not path:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download("tomg-group-umd/CSD-ViT-L", "pytorch_model.bin",
                                   revision=args.csd_revision, local_files_only=args.local_files_only)
        self.device = args.device
        self.model, size = build_csd_visual(load_csd_state(path))
        self.model = self.model.float().eval().to(self.device)
        self.processor = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(size), transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        self.protocol = {
            "model": "CSD_CLIP style branch", "weights_sha256": sha256_file(path),
            "feature": "backbone @ last_layer_style, L2 normalized; cosine",
            "processor": f"RGB, bicubic Resize({size}), CenterCrop({size}), official CLIP normalization",
            "dtype": "float32", "device": self.device,
        }

    def encode(self, images):
        import torch
        with torch.inference_mode():
            batch = torch.stack([self.processor(image) for image in images]).to(self.device)
            features = self.model(batch).float()
            return torch.nn.functional.normalize(features, dim=-1).cpu().tolist()


def judge_images(row, max_side):
    from PIL import Image
    images = [read_image(row[role]) for role in ("content", "style", "output")]
    for image in images:
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return images


def image_data_url(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def qwen_messages(images, api=False, repair=False):
    content = []
    for label, image in zip(("CONTENT", "STYLE", "OUTPUT"), images):
        content.append({"type": "text", "text": label + ":"})
        content.append({"type": "image_url", "image_url": {"url": image_data_url(image)}} if api
                       else {"type": "image", "image": image})
    instruction = RUBRIC
    if repair:
        instruction += "\nA prior response was invalid. Follow the complete JSON schema exactly; include every required field."
    content.append({"type": "text", "text": instruction})
    system_content = SYSTEM_PROMPT if api else [{"type": "text", "text": SYSTEM_PROMPT}]
    return [{"role": "system", "content": system_content}, {"role": "user", "content": content}]


class QwenJudge:
    def __init__(self, args):
        self.args = args
        self.chat_template_kwargs = {"enable_thinking": False} if getattr(args, "qwen_disable_thinking", False) else {}
        self.protocol = {
            "model": args.qwen_model, "revision": args.qwen_revision, "backend": args.qwen_backend,
            "prompt_version": PROMPT_VERSION, "system_prompt": SYSTEM_PROMPT, "rubric": RUBRIC,
            "max_side": args.qwen_max_side, "max_new_tokens": args.qwen_max_tokens,
            "do_sample": False, "temperature": 0, "seed": 0, "retries": args.qwen_retries,
            "resize": "EXIF transpose, RGB, aspect-preserving downsize only, no crop",
            "chat_template_kwargs": self.chat_template_kwargs,
        }
        if args.qwen_backend == "api":
            from urllib.parse import urlsplit
            url = urlsplit(args.qwen_base_url or "")
            if url.scheme not in {"http", "https"} or not url.netloc or url.username or url.password or url.query:
                raise ValueError("--qwen-base-url must be an explicit http(s) URL without credentials/query")
            self.protocol["base_url"] = args.qwen_base_url.rstrip("/")
            self.protocol["api_revision_label"] = args.qwen_api_revision
            self.protocol["json_mode"] = args.qwen_json_mode
        else:
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
            options = {"revision": args.qwen_revision, "local_files_only": args.local_files_only}
            self.processor = AutoProcessor.from_pretrained(args.qwen_model, use_fast=True, **options)
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                args.qwen_model, dtype=args.qwen_dtype,
                device_map=args.qwen_device_map or args.device, **options).eval()
            self.protocol.update({
                "resolved_revision": getattr(self.model.config, "_commit_hash", None),
                "local_identity": local_model_identity(args.qwen_model),
                "dtype": str(self.model.dtype), "device_map": args.qwen_device_map or args.device,
                "processor": self.processor.to_dict(),
                "image_processor": self.processor.image_processor.to_dict(),
                "chat_template": self.processor.chat_template,
            })

    def judge(self, row, repair=False):
        images = judge_images(row, self.args.qwen_max_side)
        messages = qwen_messages(images, api=self.args.qwen_backend == "api", repair=repair)
        if self.args.qwen_backend == "api":
            payload = {"model": self.args.qwen_model, "messages": messages, "temperature": 0,
                       "seed": 0, "max_tokens": self.args.qwen_max_tokens}
            if self.args.qwen_json_mode:
                payload["response_format"] = {"type": "json_object"}
            if self.chat_template_kwargs:
                payload["chat_template_kwargs"] = self.chat_template_kwargs
            headers = {"Content-Type": "application/json"}
            token = os.environ.get(self.args.qwen_api_key_env, "")
            if token:
                headers["Authorization"] = "Bearer " + token
            request = urllib.request.Request(self.args.qwen_base_url.rstrip("/") + "/chat/completions",
                                             data=json.dumps(payload).encode(), headers=headers)
            with urllib.request.urlopen(request, timeout=self.args.qwen_timeout) as response:
                result = json.load(response)
            return result["choices"][0]["message"]["content"]
        import torch
        inputs = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                    return_dict=True, return_tensors="pt",
                                                    **self.chat_template_kwargs).to(self.model.device)
        with torch.inference_mode():
            generated = self.model.generate(**inputs, do_sample=False, max_new_tokens=self.args.qwen_max_tokens)
        return self.processor.batch_decode(generated[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True,
                                          clean_up_tokenization_spaces=False)[0]
