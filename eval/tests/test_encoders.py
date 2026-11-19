"""Small, random local models test adapter correctness; no downloaded weights/GPU."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np
from PIL import Image
import torch
from clip.model import VisionTransformer
from transformers import BitImageProcessor, Dinov2Config, Dinov2Model

from eval.backends import CsdEncoder, DinoEncoder, QwenJudge, build_csd_visual


class EncoderTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        torch.manual_seed(7)
        torch.set_num_threads(1)

    def tearDown(self):
        self.temporary.cleanup()

    def test_dino_uses_normalized_cls_embedding(self):
        config = Dinov2Config(hidden_size=32, num_hidden_layers=1, num_attention_heads=4, image_size=28, patch_size=14)
        original = Dinov2Model(config).eval()
        processor = BitImageProcessor(size={"shortest_edge": 32}, crop_size={"height": 28, "width": 28})
        original.save_pretrained(self.root)
        processor.save_pretrained(self.root)
        args = argparse.Namespace(device="cpu", dino_model=str(self.root), dino_revision="main", local_files_only=True)
        encoder = DinoEncoder(args)
        images = [Image.new("RGB", (40, 30), "red"), Image.new("RGB", (30, 40), "blue")]
        with torch.inference_mode():
            expected = torch.nn.functional.normalize(original(**processor(images=images, return_tensors="pt")).last_hidden_state[:, 0], dim=-1)
        actual = torch.tensor(encoder.encode(images))
        torch.testing.assert_close(actual, expected)
        self.assertAlmostEqual(float(actual[0] @ actual[0]), 1.0, places=5)

    def test_csd_matches_official_style_branch_and_loads_training_metadata(self):
        visual = VisionTransformer(input_resolution=28, patch_size=14, width=64, layers=1, heads=1, output_dim=32).eval()
        style_head = torch.randn(64, 32)
        visual.proj = None
        state = {"module.backbone." + k: v for k, v in visual.state_dict().items()}
        state.update({"module.last_layer_style": style_head, "module.last_layer_content": -style_head})
        checkpoint = self.root / "csd.pth"
        torch.save({"model_state_dict": state, "args": argparse.Namespace(epoch=10), "best": np.float64(0.8)}, checkpoint)
        # The published CSD checkpoint was produced with NumPy 1.x. Exercise that
        # legacy scalar module name even when this test runs under NumPy 2.x.
        legacy = self.root / "legacy_csd.pth"
        with zipfile.ZipFile(checkpoint) as source, zipfile.ZipFile(legacy, "w") as destination:
            for item in source.infolist():
                data = source.read(item.filename)
                if item.filename.endswith("data.pkl"):
                    data = data.replace(b"cnumpy._core.multiarray\nscalar\n", b"cnumpy.core.multiarray\nscalar\n")
                destination.writestr(item, data)
        checkpoint = legacy
        encoder = CsdEncoder(argparse.Namespace(csd_weights=checkpoint, device="cpu"))
        images = [Image.new("RGB", (40, 30), "red"), Image.new("RGB", (30, 40), "blue")]
        batch = torch.stack([encoder.processor(image) for image in images])
        with torch.inference_mode():
            expected = torch.nn.functional.normalize(visual(batch) @ style_head, dim=-1)
        torch.testing.assert_close(torch.tensor(encoder.encode(images)), expected)
        with self.assertRaisesRegex(ValueError, "plain CLIP"):
            build_csd_visual({})
        normalized = {k.removeprefix("module."): v for k, v in state.items()}
        del normalized["backbone.ln_pre.weight"]
        with self.assertRaises(RuntimeError):
            build_csd_visual(normalized)

    def test_local_qwen_three_image_processor_and_generation_interface(self):
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import (Qwen2TokenizerFast, Qwen2VLImageProcessor, Qwen3VLConfig,
                                  Qwen3VLForConditionalGeneration, Qwen3VLProcessor, Qwen3VLVideoProcessor)

        tokens = ["[UNK]", "[PAD]", "[EOS]", "<|image_pad|>", "<|video_pad|>",
                  "<|vision_start|>", "<|vision_end|>", "word"]
        tokenizer_backend = Tokenizer(WordLevel({t: i for i, t in enumerate(tokens)}, unk_token="[UNK]"))
        tokenizer_backend.pre_tokenizer = Whitespace()
        tokenizer = Qwen2TokenizerFast(tokenizer_object=tokenizer_backend, unk_token="[UNK]", pad_token="[PAD]",
                                       eos_token="[EOS]", additional_special_tokens=tokens[3:7])
        template = ("{% for m in messages %}{% if m.content is string %}{{ m.content }}"
                    "{% else %}{% for c in m.content %}{% if c.type == 'image' %}"
                    "<|vision_start|><|image_pad|><|vision_end|>{% elif c.type == 'text' %}"
                    "{{ c.text }} {% endif %}{% endfor %}{% endif %}{% endfor %}")
        processor = Qwen3VLProcessor(
            image_processor=Qwen2VLImageProcessor(patch_size=16, min_pixels=1024, max_pixels=1024),
            tokenizer=tokenizer, video_processor=Qwen3VLVideoProcessor(), chat_template=template)
        config = Qwen3VLConfig(
            text_config={"vocab_size": len(tokens), "hidden_size": 64, "intermediate_size": 128,
                         "num_hidden_layers": 1, "num_attention_heads": 4, "num_key_value_heads": 4,
                         "head_dim": 16, "rope_scaling": {"rope_type": "default", "mrope_section": [2, 3, 3]},
                         "pad_token_id": 1, "eos_token_id": 2},
            vision_config={"depth": 1, "hidden_size": 64, "intermediate_size": 128, "num_heads": 4,
                           "out_hidden_size": 64, "deepstack_visual_indexes": [], "num_position_embeddings": 16},
            image_token_id=3, video_token_id=4, vision_start_token_id=5, vision_end_token_id=6)
        model = Qwen3VLForConditionalGeneration(config)
        model.save_pretrained(self.root)
        processor.save_pretrained(self.root)
        row = {}
        for role, color in (("content", "red"), ("style", "blue"), ("output", "purple")):
            path = self.root / (role + ".png")
            Image.new("RGB", (32, 32), color).save(path)
            row[role] = str(path)
        args = argparse.Namespace(qwen_model=str(self.root), qwen_revision="main", qwen_backend="local",
                                  qwen_max_side=64, qwen_max_tokens=1, qwen_retries=0, local_files_only=True,
                                  qwen_dtype="float32", qwen_device_map=None, device="cpu")
        judge = QwenJudge(args)
        json.dumps(judge.protocol)  # All provenance must be serializable before a long run.
        self.assertIsInstance(judge.judge(row), str)


if __name__ == "__main__":
    unittest.main()
