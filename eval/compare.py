"""Compare checkpoint/alpha runs on matched cases and identical scoring protocols."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

from .common import METRIC_SCORES, fingerprint, numeric, read_rows, write_csv, write_json


def case_key(row):
    hashes = row.get("input_hashes", {})
    return fingerprint({"id": row["id"], "seed": row.get("seed"),
                        "content": hashes.get("content", row["content"]),
                        "style": hashes.get("style", row["style"]),
                        "prompt": row.get("prompt")})


def compare_runs(directories, intersection=False):
    runs = []
    for directory in directories:
        directory = Path(directory)
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        rows = read_rows(directory / "scores.jsonl")
        indexed = {case_key(row): row for row in rows}
        if len(indexed) != len(rows):
            raise ValueError(f"{directory}: duplicate cases")
        runs.append({"directory": str(directory.resolve()), "config": config, "rows": indexed})
    keys = [set(run["rows"]) for run in runs]
    if not intersection and any(key != keys[0] for key in keys[1:]):
        raise ValueError("Sample/reference/prompt/seed sets differ. Use --intersection only for an intentional subset comparison.")
    common = set.intersection(*keys)
    if not common:
        raise ValueError("No matched cases; check sample ids, references, prompts and seeds")
    metrics = set.intersection(*(set(run["config"]["protocol_ids"]) for run in runs))
    if not metrics:
        raise ValueError("No common successfully initialized scoring protocols")
    table = [{"run": run["config"]["run_name"], "report_dir": run["directory"],
              "total_samples": len(run["rows"]), "matched_samples": len(common)} for run in runs]
    for metric in sorted(metrics):
        protocols = {run["config"]["protocol_ids"][metric] for run in runs}
        if len(protocols) != 1:
            raise ValueError(f"{metric} scorer/preprocessing/prompt versions differ; re-evaluate with one protocol")
        for score in METRIC_SCORES[metric]:
            valid = sorted(k for k in common if all(numeric(run["rows"][k]["scores"].get(score)) for run in runs))
            for row, run in zip(table, runs):
                values = [run["rows"][k]["scores"][score] for k in valid]
                deltas = [run["rows"][k]["scores"][score] - runs[0]["rows"][k]["scores"][score] for k in valid]
                row[score] = statistics.mean(values) if values else None
                row[score + "_n"] = len(valid)
                row[score + "_coverage"] = len(valid) / len(common)
                row[score + "_delta_vs_first"] = statistics.mean(deltas) if deltas else None
    for row, run in zip(table, runs):
        for field in ("lora_path", "alpha", "steps", "cfg", "seed", "height", "width"):
            values = {json.dumps(r.get("generation", {}).get(field), sort_keys=True) for r in run["rows"].values()}
            row[field] = json.loads(next(iter(values))) if len(values) == 1 else "mixed"
    return {"matched_samples": len(common), "metrics": sorted(metrics), "runs": table,
            "note": "Each score uses the same successfully scored cases in ALL runs. Missing scores are never zero. "
                    "Delta is paired against the first run. Different scores may have different valid subsets."}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("reports", nargs="+", type=Path, help="Directories containing config.json and scores.jsonl")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--intersection", action="store_true")
    args = p.parse_args(argv)
    if len(args.reports) < 2:
        p.error("Provide at least two report directories")
    result = compare_runs(args.reports, args.intersection)
    write_json(args.output_dir / "comparison.json", result)
    fields = list(result["runs"][0])
    write_csv(args.output_dir / "comparison.csv", result["runs"], fields)
    scores = [s for m in result["metrics"] for s in METRIC_SCORES[m]]
    print("run\t" + "\t".join(scores))
    for row in result["runs"]:
        print(row["run"] + "\t" + "\t".join("N/A" if row[s] is None else f"{row[s]:.4f} (n={row[s + '_n']})" for s in scores))
    print(f"Comparison: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
