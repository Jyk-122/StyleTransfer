"""python -m eval.run --manifest ... --output-dir ... [--metrics dino csd qwen]"""
from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import sys
import time

from .common import (METRIC_SCORES, SCORE_NAMES, fingerprint, inspect_inputs, load_manifest,
                     metadata_cases, numeric, read_image, runtime_versions, sha256_file,
                     summarize, write_csv, write_json, write_jsonl)
from .qwen_prompt import parse_judgment


def parser():
    p = argparse.ArgumentParser(description="Evaluate generated style transfers, separately on content and style.")
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=Path, help="JSON/JSONL with id, content, style, output")
    source.add_argument("--metadata-path", type=Path, help="DiffSynth metadata; requires --outputs-dir")
    p.add_argument("--dataset-base-path", type=Path)
    p.add_argument("--outputs-dir", type=Path, help="Existing predictions, NOT training targets")
    p.add_argument("--output-pattern", default="{index:06d}.png", help="Prediction filename; supports {index}, {id}")
    p.add_argument("--limit", type=int, help="First N rows (use a fixed held-out manifest for real experiments)")
    p.add_argument("--output-dir", type=Path, required=True, help="Scores/report directory")
    p.add_argument("--run-name", help="Display label, e.g. step18000_alpha0.5")
    p.add_argument("--cache-dir", type=Path, help="Shared cache across checkpoints; default OUTPUT_DIR/cache")
    p.add_argument("--metrics", nargs="+", choices=tuple(METRIC_SCORES), default=list(METRIC_SCORES))
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=8, help="DINO/CSD image batch size")
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--check-only", action="store_true", help="Decode/hash inputs, without loading scoring models")
    p.add_argument("--dino-model", default="facebook/dinov2-base")
    p.add_argument("--dino-revision", default="main")
    p.add_argument("--csd-weights", type=Path, help="Official checkpoint or tensor-only .safetensors")
    p.add_argument("--csd-revision", default="main")
    p.add_argument("--qwen-backend", choices=("local", "api"), default="local")
    p.add_argument("--qwen-model", default="Qwen/Qwen3-VL-8B-Instruct")
    p.add_argument("--qwen-revision", default="main")
    p.add_argument("--qwen-dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    p.add_argument("--qwen-device-map", help="e.g. auto, cuda:1; defaults to --device")
    p.add_argument("--qwen-max-side", type=int, default=1024)
    p.add_argument("--qwen-max-tokens", type=int, default=1024)
    p.add_argument("--qwen-retries", type=int, default=1, help="Additional attempts for API/JSON failures")
    p.add_argument("--qwen-base-url", help="Explicit OpenAI-compatible endpoint, e.g. http://localhost:8000/v1")
    p.add_argument("--qwen-api-key-env", default="QWEN_API_KEY", help="Environment variable name; key is never saved")
    p.add_argument("--qwen-api-revision", default="unspecified", help="Deployed model revision label for API provenance")
    p.add_argument("--qwen-json-mode", action="store_true", help="Request JSON mode if the API supports it")
    p.add_argument("--qwen-disable-thinking", action="store_true",
                   help="Set enable_thinking=False in the chat template (for compatible local/API models)")
    p.add_argument("--qwen-timeout", type=float, default=180)
    return p


def input_rows(args):
    if args.manifest:
        if args.outputs_dir or args.dataset_base_path:
            raise ValueError("Manifest paths are relative to its parent; do not also pass dataset/outputs roots")
        rows = load_manifest(args.manifest)
    else:
        if not args.outputs_dir:
            raise ValueError("--metadata-path requires --outputs-dir containing MODEL predictions")
        rows = metadata_cases(args.metadata_path, args.dataset_base_path)
        for row in rows:
            name = args.output_pattern.format(index=row["metadata_index"], id=row["id"])
            row["output"] = str((args.outputs_dir / name).resolve())
            if row["output"] == row.get("target"):
                raise ValueError("Prediction path equals training target")
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        rows = rows[:args.limit]
    return rows


def valid_embedding(value):
    return (isinstance(value, list) and bool(value) and all(numeric(x) for x in value)
            and math.isclose(sum(x * x for x in value), 1.0, abs_tol=1e-3))


def feature_scores(backend, rows, files, metric, cache_dir, batch_size):
    reference = "content" if metric == "dino" else "style"
    paths = list(dict.fromkeys(r[role] for r in rows for role in (reference, "output")
                               if role not in r["input_errors"]))
    features, errors, pending, pending_keys = {}, {}, [], set()
    for path in paths:
        key = files[path]["sha256"]
        if key in features:
            continue
        cached = cache_dir / "features" / (key + ".json")
        if cached.exists():
            try:
                vector = json.loads(cached.read_text(encoding="utf-8"))
                if valid_embedding(vector):
                    features[key] = vector
                    continue
            except (ValueError, OSError):
                pass
        if key not in pending_keys:
            pending.append((path, key))
            pending_keys.add(key)
    for start in range(0, len(pending), batch_size):
        batch = pending[start:start + batch_size]
        try:
            vectors = backend.encode([read_image(path) for path, _ in batch])
            if len(vectors) != len(batch) or any(not valid_embedding(v) for v in vectors):
                raise ValueError("Encoder returned invalid/zero/non-finite embeddings")
            for (_, key), vector in zip(batch, vectors):
                features[key] = vector
                write_json(cache_dir / "features" / (key + ".json"), vector)
        except Exception as exc:
            for _, key in batch:
                errors[key] = f"{type(exc).__name__}: {exc}"
            print(f"[{metric}] batch failed: {exc}", file=sys.stderr, flush=True)
        print(f"[{metric}] encoded {min(start + batch_size, len(pending))}/{len(pending)} uncached images", flush=True)
    for row in rows:
        bad_roles = {role: row["input_errors"][role] for role in (reference, "output") if role in row["input_errors"]}
        if bad_roles:
            row["details"][metric] = {"status": "error", "error": bad_roles}
            continue
        keys = [row["input_hashes"][role] for role in (reference, "output")]
        if any(key not in features for key in keys):
            row["details"][metric] = {"status": "error", "error": [errors.get(k) for k in keys if k not in features]}
            continue
        left, right = (features[k] for k in keys)
        if len(left) != len(right):
            raise ValueError("Mismatched cached embedding dimensions; remove the metric's cache")
        score = max(-1.0, min(1.0, sum(x * y for x, y in zip(left, right))))
        row["scores"][METRIC_SCORES[metric][0]] = score
        row["details"][metric] = {"status": "ok"}


def qwen_scores(backend, rows, cache_dir, retries):
    for index, row in enumerate(rows):
        if row["input_errors"]:
            row["details"]["qwen"] = {"status": "error", "error": row["input_errors"]}
            continue
        path = cache_dir / "judgments" / (fingerprint(row["input_hashes"]) + ".json")
        result = None
        if path.exists():
            try:
                cached = json.loads(path.read_text(encoding="utf-8"))
                if cached.get("status") == "ok":
                    parse_judgment(json.dumps(cached["judgment"]))
                    result = cached
            except (ValueError, KeyError, OSError):
                pass
        if result is None:
            attempts = []
            for attempt in range(retries + 1):
                raw = None
                try:
                    raw = backend.judge(row, repair=attempt > 0)
                    judgment = parse_judgment(raw)
                    attempts.append({"raw_response": raw})
                    result = {"status": "ok", "judgment": judgment, "attempts": attempts}
                    break
                except Exception as exc:
                    attempts.append({"raw_response": raw, "error": f"{type(exc).__name__}: {exc}"})
                    result = {"status": "error", "attempts": attempts, "error": attempts[-1]["error"]}
                    if attempt < retries:
                        time.sleep(1)
            write_json(path, result)
        row["details"]["qwen"] = result
        if result["status"] == "ok":
            row["scores"].update(qwen_content=result["judgment"]["content_score"],
                                 qwen_style=result["judgment"]["style_score"])
        print(f"[qwen] {index + 1}/{len(rows)} {row['id']}: {result['status']}", flush=True)


def save_reports(args, rows, config):
    write_json(args.output_dir / "config.json", config)
    write_jsonl(args.output_dir / "scores.jsonl", rows)
    summary = summarize(rows)
    summary.update(run_name=config["run_name"], requested_metrics=args.metrics,
                   protocol_ids=config["protocol_ids"])
    write_json(args.output_dir / "summary.json", summary)
    flat_rows = []
    for row in rows:
        flat = {key: row.get(key) for key in ("id", "style_group", "content", "style", "output")}
        flat.update(row["scores"])
        flat["errors"] = json.dumps({m: d for m, d in row["details"].items() if d["status"] != "ok"}, ensure_ascii=False)
        judgment = row["details"].get("qwen", {}).get("judgment", {})
        flat.update({key: judgment.get(key) for key in ("content_reason", "style_reason", "uncertainty")})
        flat_rows.append(flat)
    write_csv(args.output_dir / "scores.csv", flat_rows,
              ["id", "style_group", *SCORE_NAMES, "content_reason", "style_reason", "uncertainty",
               "content", "style", "output", "errors"])
    return summary


def main(argv=None):
    args = parser().parse_args(argv)
    if args.batch_size < 1 or args.qwen_retries < 0 or args.qwen_max_side < 32 or args.qwen_max_tokens < 1:
        raise ValueError("Invalid batch size, retry count, or Qwen image/token limit")
    args.metrics = list(dict.fromkeys(args.metrics))
    rows = input_rows(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_identity = fingerprint([{key: row.get(key) for key in ("id", "content", "style", "output", "seed", "generation")}
                                for row in rows])
    previous = args.output_dir / "config.json"
    if previous.exists() and json.loads(previous.read_text(encoding="utf-8"))["run_identity"] != run_identity:
        raise ValueError("This report directory belongs to different inputs/generation settings; use another --output-dir")
    files = inspect_inputs(rows)
    write_json(args.output_dir / "input_audit.json", {"samples": len(rows), "files": files})
    if args.check_only:
        failed = sum(bool(row["input_errors"]) for row in rows)
        print(f"Input audit: {len(rows)} samples, {failed} with missing/corrupt inputs")
        return int(failed > 0)
    for row in rows:
        row["scores"], row["details"] = {}, {}
    config = {"schema_version": 1, "run_name": args.run_name or args.output_dir.name,
              "run_identity": run_identity, "runtime": runtime_versions(), "protocols": {}, "protocol_ids": {}}
    cache_root = args.cache_dir or args.output_dir / "cache"
    # Include source hashes so modified preprocessing/rubrics cannot reuse stale scores.
    source_hash = fingerprint({p.name: sha256_file(p) for p in Path(__file__).parent.glob("*.py")})
    for metric in args.metrics:
        required = ("content", "style", "output") if metric == "qwen" else (
            "content" if metric == "dino" else "style", "output")
        if not any(all(role not in row["input_errors"] for role in required) for row in rows):
            for row in rows:
                row["details"][metric] = {"status": "error", "error": row["input_errors"]}
            print(f"[{metric}] no readable input pairs/triplets; skipping model load", file=sys.stderr)
            save_reports(args, rows, config)
            continue
        backend = None
        try:
            from .backends import CsdEncoder, DinoEncoder, QwenJudge
            backend = {"dino": DinoEncoder, "csd": CsdEncoder, "qwen": QwenJudge}[metric](args)
            packages = ("Pillow",) if metric == "qwen" and args.qwen_backend == "api" else (
                "Pillow", "torch", "torchvision", "transformers", "clip", "numpy", "tokenizers", "accelerate")
            protocol = {**backend.protocol, "implementation": source_hash,
                        "packages": {p: config["runtime"].get(p) for p in packages}}
            config["protocols"][metric] = protocol
            config["protocol_ids"][metric] = fingerprint(protocol)
            cache_dir = cache_root / metric / config["protocol_ids"][metric]
            if metric == "qwen":
                qwen_scores(backend, rows, cache_dir, args.qwen_retries)
            else:
                feature_scores(backend, rows, files, metric, cache_dir, args.batch_size)
        except Exception as exc:
            print(f"[{metric}] failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            for row in rows:
                row["details"].setdefault(metric, {"status": "error", "error": f"{type(exc).__name__}: {exc}"})
        finally:
            del backend
            gc.collect()
            if "torch" in sys.modules:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        save_reports(args, rows, config)
    summary = save_reports(args, rows, config)
    for metric in args.metrics:
        for score in METRIC_SCORES[metric]:
            stat = summary["scores"][score]
            print(f"{score}: mean={stat['mean']} coverage={stat['n']}/{stat['total']}")
    print(f"Reports: {args.output_dir.resolve()}")
    return int(any(row["details"][metric]["status"] != "ok" for row in rows for metric in args.metrics))


if __name__ == "__main__":
    sys.exit(main())
