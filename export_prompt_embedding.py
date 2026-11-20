"""Export the fixed style-transfer prompt using the exact DiffSynth Klein encoder path."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", type=Path, required=True, help="Local FLUX.2-klein-4B directory")
    p.add_argument("--output", type=Path, required=True, help="Output .safetensors file")
    p.add_argument("--prompt", default=None, help="Defaults to the project's fixed style-transfer instruction")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    args.output = args.output.expanduser().resolve()
    if args.output.exists():
        raise FileExistsError(f"Use a new cache filename: {args.output}")
    import torch
    from diffsynth.pipelines.flux2_image import Flux2ImagePipeline, Flux2Unit_Qwen3PromptEmbedder, ModelConfig
    from fixed_prompt import DEFAULT_PROMPT, save_prompt_cache
    from eval.common import local_model_identity, runtime_versions

    args.model_path = args.model_path.expanduser().resolve()
    weights = sorted(str(p) for p in (args.model_path / "text_encoder").glob("*.safetensors"))
    if not weights:
        raise FileNotFoundError(f"No text encoder weights in {args.model_path / 'text_encoder'}")
    prompt = DEFAULT_PROMPT if args.prompt is None else args.prompt
    # No DiT or VAE is needed to export the text conditioning.
    pipe = Flux2ImagePipeline.from_pretrained(
        device=args.device, torch_dtype=getattr(torch, args.dtype),
        model_configs=[ModelConfig(path=weights)],
        tokenizer_config=ModelConfig(path=str(args.model_path / "tokenizer")),
    )
    if pipe.text_encoder_qwen3 is None or pipe.text_encoder_qwen3.model.config.hidden_size != 2560:
        raise ValueError("This exporter requires the Qwen3-4B encoder from FLUX.2-klein-4B")
    pipe.eval()
    unit = Flux2Unit_Qwen3PromptEmbedder()
    with torch.no_grad():
        outputs = unit.process(pipe, prompt)
    template = pipe.tokenizer.chat_template
    source = {
        "model_path": str(args.model_path),
        "text_encoder_identity": local_model_identity(args.model_path / "text_encoder"),
        "tokenizer_identity": local_model_identity(args.model_path / "tokenizer"),
        "chat_template_sha256": hashlib.sha256(str(template).encode()).hexdigest(),
        "pipeline_sha256": hashlib.sha256((Path(__file__).parent / "diffsynth/pipelines/flux2_image.py").read_bytes()).hexdigest(),
        "runtime": runtime_versions(), "device": args.device,
        "add_generation_prompt": True, "enable_thinking": False, "padding": "max_length",
        "hidden_states_layers": [9, 18, 27], "max_sequence_length": 512,
    }
    save_prompt_cache(args.output, **outputs, prompt=prompt, source=source)
    print(f"Saved {args.output.resolve()}: {list(outputs['prompt_embeds'].shape)}, {args.dtype}")
    print(f"Prompt: {prompt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
