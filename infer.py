"""Generate a content/style grid, lossless predictions, previews and an eval manifest.

python infer.py --content-count 10 --style-count 10 --output-dir /data/results/step2000_a05
python -m eval.run --manifest /data/results/step2000_a05/manifest.jsonl --output-dir /data/results/step2000_a05/scores
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

from eval.common import fingerprint, local_model_identity, read_image, runtime_versions, sha256_file, write_json, write_jsonl
from eval.generate import make_pipeline

PROMPT = ("Transfer the visual style of image 2 to image 1 while preserving the content, "
          "structure, objects and layout of image 1.")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", type=Path, default=Path("/data3/jiangyikun/AIGC/models/FLUX.2-klein-4B"))
    p.add_argument("--lora-path", type=Path,
                   default=Path("./models/style_transfer/FLUX.2-klein-4B_lora_StyleExpert-40k/step-2000.safetensors"))
    p.add_argument("--no-lora", action="store_true")
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--references-dir", type=Path, default=Path("/data3/jiangyikun/AIGC/StyleExpert/assets/examples"))
    p.add_argument("--content-pattern", default="content_{index:02d}.png")
    p.add_argument("--style-pattern", default="style_{index:02d}.png")
    p.add_argument("--content-count", type=int, default=4)
    p.add_argument("--style-count", type=int, default=4)
    p.add_argument("--output-dir", type=Path,
                   default=Path("/data3/jiangyikun/AIGC/StyleTransfer/datasets/outputs/lora_styleexpert_2000"))
    p.add_argument("--prompt", default=PROMPT)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--cfg", type=float, default=1.0)
    p.add_argument("--height", type=int, default=768)
    p.add_argument("--width", type=int, default=768)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    p.add_argument("--plan-only", action="store_true", help="Check references and write the manifest without loading FLUX")
    return p


def build_cases(args):
    settings = {key: getattr(args, key) for key in ("alpha", "seed", "steps", "cfg", "height", "width", "dtype", "device")}
    settings.update(model_path=str(args.model_path), lora_path=str(args.lora_path) if args.lora_path else None,
                    alpha=args.alpha if args.lora_path else 0.0, output_format="raw_png",
                    edit_image_auto_resize=True, embedded_guidance=4.0, rand_device="cpu")
    cases = []
    for i in range(args.content_count):
        for j in range(args.style_count):
            cases.append({
                "id": f"out_{i}_{j}", "content_id": f"content_{i:02d}", "style_id": f"style_{j:02d}",
                "content": str((args.references_dir / args.content_pattern.format(index=i)).resolve()),
                "style": str((args.references_dir / args.style_pattern.format(index=j)).resolve()),
                "output": f"out_{i}_{j}.png", "preview": f"out_{i}_{j}_triplet.jpg",
                "prompt": args.prompt, "seed": args.seed, "generation": settings,
            })
    return cases


def save_outputs(output_dir, row, content, style, result, size):
    from PIL import Image, ImageDraw, ImageOps

    # Save the actual prediction first. The resized collage is only for visual inspection.
    prediction_path = output_dir / row["output"]
    temporary = prediction_path.with_suffix(".tmp.png")
    result.save(temporary)
    temporary.replace(prediction_path)

    gap, title_h = 10, 40
    canvas = Image.new("RGB", (size[0] * 3 + gap * 2, size[1] + title_h), "white")
    draw = ImageDraw.Draw(canvas)
    for column, (title, image) in enumerate(zip(("Content", "Style", "Result"), (content, style, result))):
        x = column * (size[0] + gap)
        preview = ImageOps.pad(image, size, method=Image.Resampling.LANCZOS, color="white")
        canvas.paste(preview, (x, title_h))
        draw.text((x + 12, 10), title, fill="black")
    canvas.save(output_dir / row["preview"], quality=95)


def main(argv=None):
    args = parser().parse_args(argv)
    if (args.content_count < 1 or args.style_count < 1 or args.steps < 1
            or not math.isfinite(args.alpha) or not math.isfinite(args.cfg) or not args.prompt.strip()
            or min(args.height, args.width) < 16 or args.height % 16 or args.width % 16):
        raise ValueError("Invalid counts/parameters; height and width must be positive multiples of 16")
    args.model_path = args.model_path.expanduser().resolve()
    args.lora_path = None if args.no_lora else args.lora_path.expanduser().resolve()
    args.references_dir = args.references_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    cases = build_cases(args)
    # Check every reference before loading the generator; wrong naming/counts fail early.
    references = {row[role] for row in cases for role in ("content", "style")}
    images = {path: read_image(path) for path in sorted(references)}
    hashes = {path: sha256_file(path) for path in images}
    root = Path(__file__).resolve().parent
    specification = {"cases": cases, "reference_sha256": hashes,
                     "lora_sha256": sha256_file(args.lora_path) if args.lora_path else None,
                     "model_identity": local_model_identity(args.model_path), "runtime": runtime_versions(),
                     "implementation": {name: sha256_file(root / name) for name in (
                         "infer.py", "eval/generate.py", "eval/common.py", "diffsynth/pipelines/flux2_image.py")}}
    identity = fingerprint(specification)
    config_path = args.output_dir / "infer_config.json"
    if config_path.exists():
        if json.loads(config_path.read_text(encoding="utf-8"))["identity"] != identity:
            raise ValueError("Inputs/checkpoint/alpha/settings changed. Use a new --output-dir for this run.")
    elif any((args.output_dir / row["output"]).exists() for row in cases):
        raise ValueError("Existing raw predictions have no infer_config.json; choose a new --output-dir")
    write_json(config_path, {"identity": identity, **specification})
    # Keep expected pairs even if generation fails later, so evaluation reports full coverage.
    write_jsonl(args.output_dir / "manifest.jsonl", cases)
    print(f"Evaluation manifest: {args.output_dir / 'manifest.jsonl'} ({len(cases)} pairs)")
    if args.plan_only:
        return 0
    state_path = args.output_dir / "infer_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    pending = [row for row in cases if not (
        state.get(row["id"], {}).get("status") == "ok"
        and (args.output_dir / row["output"]).exists()
        and (args.output_dir / row["preview"]).exists()
        and state[row["id"]].get("sha256") == sha256_file(args.output_dir / row["output"]))]
    if not pending:
        print("All predictions and previews already exist; prediction hashes verified.")
        return 0
    pipe = make_pipeline(args)
    import torch
    times = []
    for index, row in enumerate(pending):
        try:
            content, style = images[row["content"]], images[row["style"]]
            if args.device.startswith("cuda"):
                torch.cuda.synchronize(args.device)
            start = time.perf_counter()
            result = pipe(prompt=args.prompt, edit_image=[content, style], seed=args.seed,
                          num_inference_steps=args.steps, cfg_scale=args.cfg, height=args.height, width=args.width)
            if args.device.startswith("cuda"):
                torch.cuda.synchronize(args.device)
            elapsed = time.perf_counter() - start
            save_outputs(args.output_dir, row, content, style, result, (args.width, args.height))
            times.append(elapsed)
            state[row["id"]] = {"status": "ok", "pipeline_seconds": elapsed,
                                "sha256": sha256_file(args.output_dir / row["output"])}
        except Exception as exc:
            state[row["id"]] = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
            print(f"{row['id']}: {exc}", file=sys.stderr)
        write_json(state_path, state)
        print(f"{index + 1}/{len(pending)} {row['id']}: {state[row['id']]['status']}", flush=True)
    if times:
        print(f"Mean pipeline time over {len(times)} new successful pairs: {sum(times) / len(times):.3f}s")
    return int(any(state.get(row["id"], {}).get("status") != "ok" for row in cases))


if __name__ == "__main__":
    sys.exit(main())
