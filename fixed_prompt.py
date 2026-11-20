"""Project-local fixed conditioning for FLUX.2-klein-4B; no DiffSynth source changes."""
from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


DEFAULT_PROMPT = (
    "Transfer the visual style of image 2 to image 1 while preserving the content, "
    "structure, objects and layout of image 1."
)
CACHE_FORMAT = "styletransfer.klein-4b.prompt.v1"


def _validate(tensors):
    if set(tensors) != {"prompt_embeds", "text_ids"}:
        raise ValueError("Cache must contain prompt_embeds and text_ids")
    embeds, ids = tensors["prompt_embeds"], tensors["text_ids"]
    if tuple(embeds.shape) != (1, 512, 7680) or not embeds.is_floating_point():
        raise ValueError("Expected Klein-4B embeddings [1, 512, 7680]; retain all padding tokens")
    expected_ids = torch.zeros((1, 512, 4), dtype=torch.int64)
    expected_ids[0, :, 3] = torch.arange(512)
    if ids.dtype != torch.int64 or not torch.equal(ids.cpu(), expected_ids):
        raise ValueError("Invalid Klein text_ids")
    if not torch.isfinite(embeds).all().item():
        raise ValueError("Prompt embedding contains NaN or Inf")


def save_prompt_cache(path, prompt_embeds, text_ids, prompt, source=None):
    tensors = {"prompt_embeds": prompt_embeds.detach().cpu().contiguous(),
               "text_ids": text_ids.detach().cpu().contiguous()}
    _validate(tensors)
    metadata = {"format": CACHE_FORMAT, "prompt": prompt,
                "dtype": str(prompt_embeds.dtype).removeprefix("torch."), "source": source or {}}
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path), metadata={"prompt_cache": json.dumps(metadata, ensure_ascii=False)})


def load_prompt_cache(path, dtype):
    with safe_open(str(Path(path).expanduser()), framework="pt", device="cpu") as handle:
        metadata = json.loads((handle.metadata() or {}).get("prompt_cache", "{}"))
        if (not isinstance(metadata, dict) or metadata.get("format") != CACHE_FORMAT
                or not isinstance(metadata.get("prompt"), str)):
            raise ValueError("Not a Klein-4B prompt cache; use export_prompt_embedding.py")
        tensors = {key: handle.get_tensor(key) for key in handle.keys()}
    _validate(tensors)
    actual_dtype = str(tensors["prompt_embeds"].dtype).removeprefix("torch.")
    if metadata.get("dtype") != actual_dtype or str(dtype).removeprefix("torch.") != actual_dtype:
        raise ValueError(f"Cache dtype is {actual_dtype}; re-export using the requested --dtype")
    return {**tensors, "metadata": metadata, "prompt": metadata["prompt"]}


def without_text_encoder_configs(configs):
    """Drop standard text_encoder paths before loading; keep DiT/VAE configs."""
    kept = []
    for config in configs:
        paths = config.path if config.path is not None else config.origin_file_pattern
        paths = paths if isinstance(paths, list) else [paths]
        flags = [any(p == "text_encoder" or p.startswith("text_encoder_")
                     for p in Path(str(path)).parts) for path in paths]
        if any(flags) and not all(flags):
            raise ValueError("Use separate model configs for text_encoder and DiT/VAE")
        if not any(flags):
            kept.append(config)
    return kept


def attach_fixed_prompt(pipe, cache):
    """Replace only this instance's prompt units using DiffSynth's existing unit API."""
    from diffsynth.diffusion.base_pipeline import PipelineUnit
    from diffsynth.pipelines.flux2_image import Flux2Unit_PromptEmbedder, Flux2Unit_Qwen3PromptEmbedder

    if pipe.text_encoder is not None or pipe.text_encoder_qwen3 is not None or pipe.tokenizer is not None:
        raise ValueError("Cache mode must load only DiT/VAE with tokenizer_config=None")
    if pipe.dit is not None and pipe.dit.context_embedder.in_features != 7680:
        raise ValueError("This prompt cache requires a Klein-4B DiT")
    if cache["prompt_embeds"].dtype != pipe.torch_dtype:
        raise ValueError("Cache and pipeline dtype must match")

    class FixedPromptUnit(PipelineUnit):
        def __init__(self):
            super().__init__(take_over=True, input_params=("cfg_scale", "prompt"),
                             output_params=("prompt_embeds", "text_ids"))
            self.embeds, self.ids = cache["prompt_embeds"], cache["text_ids"]

        def process(self, pipe, inputs_shared, inputs_posi, inputs_nega):
            if inputs_shared["cfg_scale"] != 1:
                raise ValueError("Fixed positive prompt cache requires CFG 1")
            prompt = inputs_posi.get("prompt")
            if prompt not in (None, "", cache["prompt"]):
                raise ValueError("Prompt does not match the fixed cache; use text mode for per-sample captions")
            # .to() reuses the tensor once on the correct device; never re-encode text.
            self.embeds = self.embeds.to(device=pipe.device, dtype=pipe.torch_dtype)
            self.ids = self.ids.to(device=pipe.device)
            inputs_posi.update(prompt_embeds=self.embeds, text_ids=self.ids)
            inputs_nega.update(prompt_embeds=self.embeds, text_ids=self.ids)
            return inputs_shared, inputs_posi, inputs_nega

    prompt_types = (Flux2Unit_PromptEmbedder, Flux2Unit_Qwen3PromptEmbedder)
    indices = [i for i, unit in enumerate(pipe.units) if isinstance(unit, prompt_types)]
    if len(indices) != 2:
        raise ValueError("Expected the original two FLUX.2 prompt units; pipeline layout has changed")
    replacement = FixedPromptUnit()
    pipe.units = [replacement if i == indices[0] else unit
                  for i, unit in enumerate(pipe.units) if i != indices[1]]
    return pipe


def make_cached_pipeline(args, cache):
    from diffsynth.pipelines.flux2_image import Flux2ImagePipeline, ModelConfig

    configs = []
    for pattern in ("transformer/*.safetensors", "vae/diffusion_pytorch_model.safetensors"):
        files = sorted(str(p) for p in args.model_path.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No weights matched {args.model_path / pattern}")
        configs.append(ModelConfig(path=files))
    pipe = Flux2ImagePipeline.from_pretrained(
        torch_dtype=getattr(torch, args.dtype), device=args.device,
        model_configs=configs, tokenizer_config=None)
    attach_fixed_prompt(pipe, cache)
    if args.lora_path and args.alpha != 0:
        pipe.load_lora(pipe.dit, str(args.lora_path), alpha=args.alpha)
    return pipe
