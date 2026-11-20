"""CPU integration checks against the unmodified DiffSynth pipeline. No downloaded weights."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image
import torch

import export_prompt_embedding
import infer
from fixed_prompt import DEFAULT_PROMPT, attach_fixed_prompt, load_prompt_cache, make_cached_pipeline, save_prompt_cache
from diffsynth.pipelines.flux2_image import Flux2ImagePipeline, Flux2Unit_Qwen3PromptEmbedder
from diffsynth.models.flux2_dit import Flux2DiT
from examples.flux2.model_training.train import Flux2ImageTrainingModule


class Tokenizer:
    chat_template = "test-template"

    def apply_chat_template(self, messages, **kwargs):
        return messages[0]["content"]

    def __call__(self, text, **kwargs):
        ids = torch.zeros((1, kwargs["max_length"]), dtype=torch.long)
        values = [ord(c) % 31 for c in text][:ids.shape[1]]
        ids[0, :len(values)] = torch.tensor(values)
        return {"input_ids": ids, "attention_mask": ids.ne(0).long()}


class Encoder(torch.nn.Module):
    """Deterministic Qwen stand-in; embedding extraction and DiT are real code."""
    def __init__(self):
        super().__init__()
        self.model = SimpleNamespace(config=SimpleNamespace(hidden_size=2560))

    def forward(self, input_ids, **kwargs):
        features = input_ids.float().unsqueeze(-1) / 31 + torch.arange(2560).float() / 2560
        states = [None] * 28
        for i in (9, 18, 27):
            states[i] = features * (i / 27)
        return SimpleNamespace(hidden_states=states)


class VAE(torch.nn.Module):
    def encode(self, image):
        pooled = torch.nn.functional.avg_pool2d(image.mean(dim=1, keepdim=True), 16)
        return pooled.expand(-1, 128, -1, -1).contiguous()

    def decode(self, latent):
        return torch.nn.functional.interpolate(latent[:, :3].float(), scale_factor=16).tanh()


def tiny_pipe(dtype=torch.float32):
    pipe = Flux2ImagePipeline(device="cpu", torch_dtype=dtype)
    pipe.dit = Flux2DiT(num_layers=1, num_single_layers=1, num_attention_heads=2, attention_head_dim=16,
                       joint_attention_dim=7680, axes_dims_rope=(4, 4, 4, 4), guidance_embeds=False).to(dtype)
    pipe.vae, pipe.text_encoder_qwen3, pipe.tokenizer = VAE(), Encoder(), Tokenizer()
    return pipe.eval()


class FixedPromptTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.path = self.root / "prompt.safetensors"
        self.pipe = tiny_pipe()
        self.save_cache(self.pipe)

    def tearDown(self):
        self.tmp.cleanup()

    def save_cache(self, pipe):
        outputs = Flux2Unit_Qwen3PromptEmbedder().process(pipe, DEFAULT_PROMPT)
        save_prompt_cache(self.path, **outputs, prompt=DEFAULT_PROMPT)
        return load_prompt_cache(self.path, pipe.torch_dtype)

    def test_four_step_dit_predictions_and_image_match_without_text_encoder(self):
        cache = load_prompt_cache(self.path, "float32")
        image = Image.new("RGB", (16, 16), "red")
        kwargs = dict(prompt=DEFAULT_PROMPT, edit_image=[image, image], edit_image_auto_resize=False,
                      seed=3, height=16, width=16, num_inference_steps=4, cfg_scale=1,
                      progress_bar_cmd=lambda steps: steps)
        predictions = []
        hook = self.pipe.dit.register_forward_hook(lambda m, i, o: predictions.append(o.detach().clone()))
        original_units = list(self.pipe.units)
        expected = self.pipe(**kwargs)
        self.pipe.text_encoder_qwen3, self.pipe.tokenizer = None, None
        attach_fixed_prompt(self.pipe, cache)
        actual = self.pipe(**kwargs)
        hook.remove()
        self.assertEqual(expected.tobytes(), actual.tobytes())
        self.assertEqual(len(predictions), 8)
        for a, b in zip(predictions[:4], predictions[4:]):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        # The other pipeline units and the source class remain unchanged.
        self.assertIs(self.pipe.units[2], original_units[3])
        self.assertEqual(len(tiny_pipe().units), len(original_units))
        with self.assertRaisesRegex(ValueError, "CFG 1"):
            self.pipe(**{**kwargs, "cfg_scale": 2})
        with self.assertRaisesRegex(ValueError, "match"):
            self.pipe(**{**kwargs, "prompt": "different"})

    def test_training_loss_and_lora_gradients_match_with_promptless_metadata(self):
        original = tiny_pipe(torch.bfloat16)
        self.save_cache(original)
        weights = ",".join(f"model:{folder}/*.safetensors" for folder in ("text_encoder", "transformer", "vae"))
        kwargs = dict(model_id_with_origin_paths=weights, device="cpu", lora_base_model="dit",
                      lora_target_modules="to_q,to_k,to_v", lora_rank=2,
                      extra_inputs="edit_image,edit_image_auto_resize", use_gradient_checkpointing=False)
        torch.manual_seed(11)
        with patch.object(Flux2ImagePipeline, "from_pretrained", return_value=copy.deepcopy(original)):
            text_model = Flux2ImageTrainingModule(**kwargs)
        cached_pipe = copy.deepcopy(original)
        cached_pipe.text_encoder_qwen3, cached_pipe.tokenizer = None, None
        torch.manual_seed(11)
        with patch.object(Flux2ImagePipeline, "from_pretrained", return_value=cached_pipe) as loader:
            cache_model = Flux2ImageTrainingModule(**kwargs, prompt_embedding_path=str(self.path))
        self.assertIsNone(loader.call_args.kwargs["tokenizer_config"])
        self.assertEqual(len(loader.call_args.kwargs["model_configs"]), 2)
        image = Image.new("RGB", (16, 16), "blue")
        data = {"image": image, "edit_image": [image, image], "edit_image_auto_resize": False}
        torch.manual_seed(5)
        text_loss = text_model({**data, "prompt": DEFAULT_PROMPT})
        text_loss.backward()
        torch.manual_seed(5)
        cache_loss = cache_model(data)
        cache_loss.backward()
        torch.testing.assert_close(text_loss, cache_loss, rtol=0, atol=0)
        gradients = {k: p.grad for k, p in text_model.named_parameters() if p.requires_grad}
        self.assertTrue(any(g is not None and torch.count_nonzero(g) for g in gradients.values()))
        for k, p in cache_model.named_parameters():
            if p.requires_grad:
                torch.testing.assert_close(gradients[k], p.grad, rtol=0, atol=0)

    def test_loading_only_needed_models_and_export_roundtrip(self):
        for filename in ("text_encoder/model.safetensors", "transformer/model.safetensors", "vae/diffusion_pytorch_model.safetensors"):
            file = self.root / filename
            file.parent.mkdir(exist_ok=True)
            file.touch()
        output = self.root / "export.safetensors"
        with patch.object(Flux2ImagePipeline, "from_pretrained", return_value=self.pipe) as loader:
            export_prompt_embedding.main(["--model-path", str(self.root), "--output", str(output),
                                          "--device", "cpu", "--dtype", "float32"])
        self.assertEqual(len(loader.call_args.kwargs["model_configs"]), 1)
        self.assertIn("text_encoder", str(loader.call_args.kwargs["model_configs"][0].path))
        cache = load_prompt_cache(output, "float32")
        self.assertEqual(cache["prompt"], DEFAULT_PROMPT)
        self.assertIn("chat_template_sha256", cache["metadata"]["source"])
        (self.root / "text_encoder/model.safetensors").unlink()
        self.pipe.text_encoder_qwen3, self.pipe.tokenizer = None, None
        args = SimpleNamespace(model_path=self.root, dtype="float32", device="cpu", lora_path=None)
        with patch.object(Flux2ImagePipeline, "from_pretrained", return_value=self.pipe) as loader:
            make_cached_pipeline(args, cache)
        self.assertIsNone(loader.call_args.kwargs["tokenizer_config"])
        self.assertEqual(len(loader.call_args.kwargs["model_configs"]), 2)
        with self.assertRaisesRegex(ValueError, "dtype"):
            load_prompt_cache(output, "bfloat16")
        with self.assertRaisesRegex(ValueError, "padding"):
            save_prompt_cache(output, cache["prompt_embeds"][:, :10], cache["text_ids"][:, :10], DEFAULT_PROMPT)

    def test_infer_cache_mode_warmup_and_cache_change_invalidates_resume(self):
        for name in ("content_00", "style_00"):
            Image.new("RGB", (16, 16), "red").save(self.root / f"{name}.png")
        output = self.root / "outputs"
        args = ["--references-dir", str(self.root), "--content-count", "1", "--style-count", "1",
                "--model-path", str(self.root), "--output-dir", str(output), "--no-lora", "--device", "cpu",
                "--dtype", "float32", "--width", "16", "--height", "16", "--warmup-runs", "1",
                "--prompt-embedding-path", str(self.path)]
        with patch("fixed_prompt.make_cached_pipeline", return_value=lambda **kw: Image.new("RGB", (16, 16), "blue")):
            self.assertEqual(infer.main(args), 0)
        timing = json.loads((output / "timing.json").read_text())
        self.assertEqual(timing["conditioning_mode"], "cached")
        self.assertEqual(timing["successful_pairs"], 1)
        self.assertEqual(timing["warmup_runs"], 1)
        with self.assertRaisesRegex(ValueError, "CFG 1"):
            infer.main(args + ["--cfg", "2"])
        cache = load_prompt_cache(self.path, "float32")
        save_prompt_cache(self.path, cache["prompt_embeds"] + 1, cache["text_ids"], DEFAULT_PROMPT)
        with self.assertRaisesRegex(ValueError, "settings changed"):
            infer.main(args)


if __name__ == "__main__":
    unittest.main()
