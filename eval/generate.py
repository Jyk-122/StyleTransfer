"""Generate one checkpoint/alpha run. One fresh pipeline per invocation prevents LoRA accumulation."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import sys
import time

from .common import (fingerprint, local_model_identity, metadata_cases, read_image, runtime_versions,
                     sha256_file, write_json, write_jsonl)


def make_pipeline(args):
    import torch
    from diffsynth.pipelines.flux2_image import Flux2ImagePipeline, ModelConfig

    def model_files(pattern):
        files = sorted(str(p) for p in args.model_path.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No weights matched {args.model_path / pattern}")
        return ModelConfig(path=files)

    pipe = Flux2ImagePipeline.from_pretrained(
        torch_dtype=getattr(torch, args.dtype), device=args.device,
        model_configs=[model_files("text_encoder/*.safetensors"), model_files("transformer/*.safetensors"),
                       model_files("vae/diffusion_pytorch_model.safetensors")],
        tokenizer_config=ModelConfig(path=str(args.model_path / "tokenizer")),
    )
    if args.lora_path and args.alpha != 0:
        pipe.load_lora(pipe.dit, str(args.lora_path), alpha=args.alpha)
    return pipe


def main(argv=None):
    p = argparse.ArgumentParser(description="Generate a fixed style-transfer evaluation subset with FLUX.2-klein-4B.")
    p.add_argument("--metadata-path", type=Path, required=True)
    p.add_argument("--dataset-base-path", type=Path)
    p.add_argument("--model-path", type=Path, required=True, help="Local FLUX.2-klein-4B directory")
    p.add_argument("--lora-path", type=Path, help="Omit for the original model")
    p.add_argument("--alpha", type=float, default=0.5, help="Inference LoRA loading strength")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--limit", type=int, default=100, help="Random fixed subset size; 0 means all rows")
    p.add_argument("--selection-seed", type=int, default=42)
    p.add_argument("--seed", type=int, default=0, help="Same generation seed for each selected case")
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--cfg", type=float, default=1.0)
    p.add_argument("--height", type=int, default=768)
    p.add_argument("--width", type=int, default=768)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    p.add_argument("--plan-only", action="store_true", help="Write/validate manifest without loading the generator")
    args = p.parse_args(argv)
    if (args.limit < 0 or args.steps < 1 or not math.isfinite(args.alpha) or not math.isfinite(args.cfg)
            or args.height <= 0 or args.width <= 0 or args.height % 16 or args.width % 16):
        raise ValueError("Invalid generation parameters; width/height must be positive multiples of 16")
    args.model_path = args.model_path.expanduser().resolve()
    args.lora_path = args.lora_path.expanduser().resolve() if args.lora_path else None
    args.output_dir = args.output_dir.expanduser().resolve()
    cases = metadata_cases(args.metadata_path, args.dataset_base_path)
    if args.limit and args.limit < len(cases):
        indices = sorted(random.Random(args.selection_seed).sample(range(len(cases)), args.limit))
        cases = [cases[i] for i in indices]
    settings = {
        "model_path": str(args.model_path), "model_identity": local_model_identity(args.model_path),
        "lora_path": str(args.lora_path) if args.lora_path else None,
        "lora_sha256": sha256_file(args.lora_path) if args.lora_path else None,
        "alpha": args.alpha if args.lora_path else 0.0, "steps": args.steps, "cfg": args.cfg,
        "height": args.height, "width": args.width, "seed": args.seed, "device": args.device,
        "dtype": args.dtype, "edit_image_auto_resize": True, "embedded_guidance": 4.0,
        "rand_device": "cpu", "selection_seed": args.selection_seed,
    }
    reference_hashes = {}
    for row in cases:
        if not row["prompt"].strip():
            raise ValueError(f"{row['id']}: missing prompt in metadata")
        for role in ("content", "style"):
            if row[role] not in reference_hashes:
                read_image(row[role])
                reference_hashes[row[role]] = sha256_file(row[role])
        row.update(output=str(args.output_dir / "images" / (row["id"] + ".png")), seed=args.seed,
                   generation=settings, reference_hashes={r: reference_hashes[row[r]] for r in ("content", "style")})
    source_root = Path(__file__).resolve().parents[1]
    specification = {
        "settings": settings, "metadata_sha256": sha256_file(args.metadata_path),
        "cases": [{k: row[k] for k in ("id", "reference_hashes", "prompt")} for row in cases],
        "implementation": {name: sha256_file(source_root / name) for name in (
            "eval/generate.py", "eval/common.py", "diffsynth/pipelines/flux2_image.py", "diffsynth/diffusion/base_pipeline.py")},
        "runtime": runtime_versions(),
    }
    identity = fingerprint(specification)
    config_path = args.output_dir / "generation.json"
    if config_path.exists():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous["identity"] != identity:
            raise ValueError("Generation settings/inputs changed. Use a new --output-dir to avoid mixing runs.")
    elif (args.output_dir / "images").exists() and any((args.output_dir / "images").iterdir()):
        raise ValueError("Found images without generation.json; use an empty output directory")
    write_json(config_path, {"identity": identity, **specification})
    write_jsonl(args.output_dir / "manifest.jsonl", cases)
    if args.plan_only:
        print(f"Prepared {len(cases)} cases: {args.output_dir / 'manifest.jsonl'}")
        return 0
    state_path = args.output_dir / "generation_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    pending = []
    for row in cases:
        previous = state.get(row["id"], {})
        path = Path(row["output"])
        if previous.get("status") == "ok" and path.exists() and previous.get("sha256") == sha256_file(path):
            continue
        pending.append(row)
    if not pending:
        print("All outputs already exist and match their recorded hashes.")
        return 0
    pipe = make_pipeline(args)
    import torch
    for index, row in enumerate(pending):
        try:
            references = [read_image(row[role]) for role in ("content", "style")]
            if args.device.startswith("cuda"):
                torch.cuda.synchronize(args.device)
            start = time.perf_counter()
            image = pipe(prompt=row["prompt"], edit_image=references, seed=args.seed,
                         num_inference_steps=args.steps, cfg_scale=args.cfg, height=args.height, width=args.width)
            if args.device.startswith("cuda"):
                torch.cuda.synchronize(args.device)
            elapsed = time.perf_counter() - start
            path = Path(row["output"])
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp.png")
            image.save(temporary)
            temporary.replace(path)
            state[row["id"]] = {"status": "ok", "sha256": sha256_file(path), "pipeline_seconds": elapsed}
        except Exception as exc:
            state[row["id"]] = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
            print(f"[generation] {row['id']}: {exc}", file=sys.stderr)
        write_json(state_path, state)
        print(f"[generation] {index + 1}/{len(pending)} {row['id']}: {state[row['id']]['status']}", flush=True)
    return int(any(state.get(row["id"], {}).get("status") != "ok" for row in cases))


if __name__ == "__main__":
    sys.exit(main())
