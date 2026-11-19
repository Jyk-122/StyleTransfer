from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import statistics

SCORE_NAMES = ("dino_content", "csd_style", "qwen_content", "qwen_style")
METRIC_SCORES = {"dino": SCORE_NAMES[:1], "csd": SCORE_NAMES[1:2], "qwen": SCORE_NAMES[2:]}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def read_rows(path):
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    rows = json.loads(text) if text.lstrip().startswith("[") else [
        json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(rows, list) or not rows or any(not isinstance(r, dict) for r in rows):
        raise ValueError(f"{path}: expected a nonempty JSON array or JSONL of objects")
    return rows


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as destination:
        for row in rows:
            destination.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def resolved_path(value, root):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Expected a nonempty image path, got {value!r}")
    path = Path(value).expanduser()
    return str((path if path.is_absolute() else Path(root) / path).resolve())


def load_manifest(path):
    """`output` is deliberately required; training metadata `image` is NOT a prediction."""
    path = Path(path).resolve()
    rows, seen = read_rows(path), set()
    for row in rows:
        if "output" not in row:
            raise ValueError("Evaluation requires an explicit 'output' (model prediction). "
                             "Training metadata 'image' is the dataset target. Use eval.generate "
                             "or --metadata-path with --outputs-dir.")
        if not isinstance(row.get("id"), (str, int)) or isinstance(row.get("id"), bool):
            raise ValueError("Each manifest row needs a string/integer id")
        row["id"] = str(row["id"])
        if not row["id"] or row["id"] in seen:
            raise ValueError(f"Empty or duplicate sample id: {row['id']!r}")
        seen.add(row["id"])
        if row.get("style_group") is not None:
            row["style_group"] = str(row["style_group"])
        for role in ("content", "style", "output"):
            row[role] = resolved_path(row.get(role), path.parent)
        if row.get("target"):
            row["target"] = resolved_path(row["target"], path.parent)
            if row["target"] == row["output"]:
                raise ValueError(f"{row['id']}: output points to the dataset target")
    return rows


def metadata_cases(metadata_path, dataset_base_path=None):
    metadata_path = Path(metadata_path).resolve()
    root = Path(dataset_base_path).resolve() if dataset_base_path else metadata_path.parent
    cases = []
    for index, row in enumerate(read_rows(metadata_path)):
        references = row.get("edit_image")
        if not isinstance(references, list) or len(references) != 2:
            raise ValueError(f"metadata[{index}].edit_image must be [content, style]")
        case = {"id": f"{index:06d}", "metadata_index": index,
                "content": resolved_path(references[0], root),
                "style": resolved_path(references[1], root), "prompt": row.get("prompt", "")}
        if not isinstance(case["prompt"], str):
            raise ValueError(f"metadata[{index}].prompt must be a string")
        if row.get("image"):
            case["target"] = resolved_path(row["image"], root)
        if row.get("style_group") is not None:
            case["style_group"] = str(row["style_group"])
        cases.append(case)
    return cases


def read_image(path):
    from PIL import Image, ImageOps
    with Image.open(path) as source:
        source.load()
        image = ImageOps.exif_transpose(source)
        if "A" in image.getbands() or "transparency" in image.info:
            rgba = image.convert("RGBA")
            background = Image.new("RGBA", rgba.size, "white")
            image = Image.alpha_composite(background, rgba)
        return image.convert("RGB")


def inspect_inputs(rows):
    """Hash image bytes once per path; preserve missing/corrupt cases in the denominator."""
    files = {}
    for row in rows:
        row["input_hashes"], row["input_errors"] = {}, {}
        for role in ("content", "style", "output"):
            path = row[role]
            if path not in files:
                try:
                    image = read_image(path)
                    files[path] = {"sha256": sha256_file(path), "size": list(image.size)}
                except Exception as exc:
                    files[path] = {"error": f"{type(exc).__name__}: {exc}"}
            info = files[path]
            if "error" in info:
                row["input_errors"][role] = info["error"]
            else:
                row["input_hashes"][role] = info["sha256"]
    return files


def local_model_identity(path):
    """Local models: config hashes + size/mtime inventory, not a full weight checksum."""
    path = Path(path).expanduser()
    if not path.is_dir():
        return None
    inventory = []
    for file in sorted(path.rglob("*")):
        if file.is_file() and file.suffix in {".json", ".safetensors", ".bin", ".model", ".txt", ".jinja"}:
            stat = file.stat()
            inventory.append([str(file.relative_to(path)), stat.st_size, stat.st_mtime_ns,
                              sha256_file(file) if stat.st_size < 1024 * 1024 else None])
    return {"kind": "local_inventory", "path": str(path.resolve()), "fingerprint": fingerprint(inventory)}


def runtime_versions():
    result = {"python": platform.python_version(), "platform": platform.platform()}
    for package in ("torch", "torchvision", "transformers", "Pillow", "numpy", "clip", "safetensors",
                    "tokenizers", "accelerate", "huggingface_hub"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    return result


def numeric(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def statistics_for(values, total):
    values = list(values)
    return {"n": len(values), "total": total, "coverage": len(values) / total if total else 0,
            "mean": statistics.mean(values) if values else None,
            "median": statistics.median(values) if values else None,
            "std": statistics.stdev(values) if len(values) > 1 else None}


def summarize(rows):
    summary = {"samples": len(rows), "scores": {}, "by_style_group": {}}
    for name in SCORE_NAMES:
        values = [r["scores"][name] for r in rows if numeric(r.get("scores", {}).get(name))]
        summary["scores"][name] = statistics_for(values, len(rows))
    groups = sorted({r["style_group"] for r in rows if r.get("style_group") is not None})
    for group in groups:
        subset = [r for r in rows if r.get("style_group") == group]
        summary["by_style_group"][group] = {name: statistics_for(
            [r["scores"][name] for r in subset if numeric(r.get("scores", {}).get(name))], len(subset))
            for name in SCORE_NAMES}
    summary["style_macro_mean"] = {}
    for name in SCORE_NAMES:
        values = [group[name]["mean"] for group in summary["by_style_group"].values() if group[name]["mean"] is not None]
        summary["style_macro_mean"][name] = statistics.mean(values) if values else None
    judgments = [r["details"]["qwen"]["judgment"] for r in rows
                 if r.get("details", {}).get("qwen", {}).get("status") == "ok"]
    summary["qwen_judged_samples"] = len(judgments)
    if judgments:
        summary["qwen_flag_rates"] = {key: sum(j["flags"][key] for j in judgments) / len(judgments)
                                      for key in judgments[0]["flags"]}
        counted = [j for j in judgments if j["content_people"] is not None and j["output_people"] is not None
                   and (j["content_people"] > 0 or j["output_people"] > 0)]
        summary["qwen_people_count_agreement"] = {
            "n": len(counted), "rate": sum(j["content_people"] == j["output_people"] for j in counted) / len(counted)
            if counted else None, "note": "VLM estimates on both images, excluding pairs with no people in either; "
                                           "not ground-truth count accuracy"}
    return summary


def write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
