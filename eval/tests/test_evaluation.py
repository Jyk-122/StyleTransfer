from __future__ import annotations

import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from eval import generate, run
from eval.common import (inspect_inputs, load_manifest, metadata_cases, read_image, read_rows,
                         summarize, write_json, write_jsonl)
from eval.compare import compare_runs
from eval.qwen_prompt import FLAGS, parse_judgment


def judgment(**updates):
    value = {"assessable": True, "content_score": 8, "style_score": 7,
             "content_reason": "All three people retained, hands have minor distortions.",
             "style_reason": "Clay material is present but background remains photographic.",
             "content_people": 3, "output_people": 3, "flags": dict.fromkeys(FLAGS, False),
             "uncertainty": ""}
    value.update(updates)
    return value


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        for name, color in (("content", "red"), ("style", "blue"), ("output", "purple")):
            Image.new("RGB", (40, 24), color).save(self.root / (name + ".png"))
        self.row = {"id": "0", "content": "content.png", "style": "style.png", "output": "output.png"}
        self.manifest = self.root / "manifest.jsonl"
        write_jsonl(self.manifest, [self.row])

    def tearDown(self):
        self.temporary.cleanup()

    def test_training_targets_are_not_predictions(self):
        write_jsonl(self.manifest, [{"image": "output.png", "edit_image": ["content.png", "style.png"]}])
        with self.assertRaisesRegex(ValueError, "explicit 'output'"):
            load_manifest(self.manifest)
        write_jsonl(self.manifest, [{**self.row, "target": "output.png"}])
        with self.assertRaisesRegex(ValueError, "dataset target"):
            load_manifest(self.manifest)

    def test_transparent_line_art_is_composited_on_white(self):
        for mode, transparent_black in (("RGBA", (0, 0, 0, 0)), ("LA", (0, 0))):
            path = self.root / (mode + ".png")
            Image.new(mode, (4, 4), transparent_black).save(path)
            self.assertEqual(read_image(path).getpixel((0, 0)), (255, 255, 255))

    def test_duplicate_id_and_reference_order(self):
        write_jsonl(self.manifest, [self.row, self.row])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            load_manifest(self.manifest)
        metadata = self.root / "metadata.json"
        write_json(metadata, [{"image": "target.png", "edit_image": ["content.png", "style.png"], "prompt": "transfer"}])
        cases = metadata_cases(metadata)
        self.assertEqual(cases[0]["content"], str(self.root / "content.png"))
        self.assertEqual(cases[0]["style"], str(self.root / "style.png"))

    def test_check_only_preserves_missing_outputs_in_denominator(self):
        write_jsonl(self.manifest, [self.row, {**self.row, "id": "1", "output": "missing.png"}])
        result = run.main(["--manifest", str(self.manifest), "--output-dir", str(self.root / "report"), "--check-only"])
        self.assertEqual(result, 1)
        audit = json.loads((self.root / "report/input_audit.json").read_text())
        self.assertEqual(audit["samples"], 2)
        self.assertIn("error", audit["files"][str(self.root / "missing.png")])

    def test_judge_parser_rejects_invalid_scores_and_partial_objects(self):
        self.assertEqual(parse_judgment("```json\n" + json.dumps(judgment(content_score=0)) + "\n```")["content_score"], 0)
        for score in (True, "8", None, -1, 11, float("nan")):
            with self.subTest(score=score), self.assertRaises(ValueError):
                parse_judgment(json.dumps(judgment(content_score=score)))
        for update in ({"content_people": True}, {"flags": {}}, {"assessable": False}, {"style_reason": ""}):
            with self.subTest(update=update), self.assertRaises(ValueError):
                parse_judgment(json.dumps(judgment(**update)))
        with self.assertRaises(ValueError):
            parse_judgment('prefix ' + json.dumps(judgment()))

    def test_api_pipeline_cache_image_invalidation_and_error_coverage(self):
        output = self.root / "report"
        args = ["--manifest", str(self.manifest), "--output-dir", str(output), "--metrics", "qwen",
                "--qwen-backend", "api", "--qwen-base-url", "http://localhost:8000/v1", "--qwen-retries", "0",
                "--qwen-disable-thinking"]
        captured = []

        def response(request, timeout):
            payload = json.loads(request.data)
            captured.append(payload)
            return io.StringIO(json.dumps({"choices": [{"message": {"content": json.dumps(judgment())}}]}))

        with patch("urllib.request.urlopen", side_effect=response):
            self.assertEqual(run.main(args), 0)
            self.assertEqual(run.main(args), 0)
        self.assertEqual(len(captured), 1)
        content = captured[0]["messages"][1]["content"]
        self.assertEqual([c["text"] for c in content if c["type"] == "text"][:3], ["CONTENT:", "STYLE:", "OUTPUT:"])
        self.assertEqual(sum(c["type"] == "image_url" for c in content), 3)
        self.assertEqual(captured[0]["temperature"], 0)
        self.assertEqual(captured[0]["chat_template_kwargs"], {"enable_thinking": False})
        Image.new("RGB", (40, 24), "yellow").save(self.root / "output.png")
        with patch("urllib.request.urlopen", side_effect=response):
            self.assertEqual(run.main(args), 0)
        self.assertEqual(len(captured), 2)
        (self.root / "output.png").unlink()
        with patch("urllib.request.urlopen") as api:
            self.assertEqual(run.main(args), 1)
            api.assert_not_called()
        summary = json.loads((output / "summary.json").read_text())
        self.assertIsNone(summary["scores"]["qwen_content"]["mean"])
        self.assertEqual(summary["scores"]["qwen_content"]["coverage"], 0)

    def test_qwen_retries_keep_raw_responses_and_do_not_cache_failures_as_success(self):
        rows = load_manifest(self.manifest)
        inspect_inputs(rows)
        rows[0].update(scores={}, details={})
        class Judge:
            replies = iter(["not JSON", json.dumps(judgment())])
            def judge(self, row, repair=False):
                return next(self.replies)
        with patch("eval.run.time.sleep"):
            run.qwen_scores(Judge(), rows, self.root / "cache", retries=1)
        detail = rows[0]["details"]["qwen"]
        self.assertEqual(detail["status"], "ok")
        self.assertEqual(detail["attempts"][0]["raw_response"], "not JSON")
        self.assertEqual(len(detail["attempts"]), 2)

    def test_feature_cache_and_content_style_roles(self):
        rows = load_manifest(self.manifest)
        files = inspect_inputs(rows)
        rows[0].update(scores={}, details={})
        class Encoder:
            def encode(self, images):
                return [[1.0, 0.0] if image.getpixel((0, 0))[0] > 0 else [0.0, 1.0] for image in images]
        encoder = Encoder()
        run.feature_scores(encoder, rows, files, "dino", self.root / "dino", 2)
        run.feature_scores(encoder, rows, files, "csd", self.root / "csd", 2)
        self.assertEqual(rows[0]["scores"], {"dino_content": 1.0, "csd_style": 0.0})
        with patch.object(encoder, "encode", side_effect=AssertionError("unexpected encode")):
            run.feature_scores(encoder, rows, files, "dino", self.root / "dino", 2)
        self.assertEqual(rows[0]["details"]["dino"]["status"], "ok")

    def test_means_do_not_turn_missing_values_into_zeros(self):
        summary = summarize([{"scores": {"dino_content": 0.8}}, {"scores": {}}])
        self.assertEqual(summary["scores"]["dino_content"]["mean"], 0.8)
        self.assertEqual(summary["scores"]["dino_content"]["coverage"], 0.5)

    def test_compare_uses_paired_valid_cases_and_rejects_protocol_drift(self):
        directories = [self.root / "early", self.root / "late"]
        base = {"id": "0", "content": "c", "style": "s", "input_hashes": {"content": "a", "style": "b"},
                "scores": {"dino_content": 0.9}}
        for i, directory in enumerate(directories):
            write_json(directory / "config.json", {"run_name": directory.name, "protocol_ids": {"dino": "same"}})
            rows = [copy.deepcopy(base), {**copy.deepcopy(base), "id": "1"}]
            rows[0]["scores"] = {"dino_content": 0.9 if i == 0 else 0.7}
            rows[1]["scores"] = {"dino_content": 0.1} if i == 0 else {}
            write_jsonl(directory / "scores.jsonl", rows)
        result = compare_runs(directories)
        self.assertEqual(result["runs"][0]["dino_content"], 0.9)
        self.assertEqual(result["runs"][0]["dino_content_n"], 1)
        self.assertAlmostEqual(result["runs"][1]["dino_content_delta_vs_first"], -0.2)
        write_json(directories[1] / "config.json", {"run_name": "late", "protocol_ids": {"dino": "changed"}})
        with self.assertRaisesRegex(ValueError, "versions differ"):
            compare_runs(directories)
        rows = read_rows(directories[1] / "scores.jsonl")
        rows[0]["input_hashes"]["content"] = "different reference"
        write_jsonl(directories[1] / "scores.jsonl", rows)
        with self.assertRaisesRegex(ValueError, "sets differ"):
            compare_runs(directories)

    def test_generation_reuses_verified_outputs_and_rejects_changed_settings(self):
        metadata = self.root / "metadata.json"
        write_json(metadata, [{"image": "target.png", "edit_image": ["content.png", "style.png"], "prompt": "transfer"}])
        model = self.root / "model"
        model.mkdir()
        args = ["--metadata-path", str(metadata), "--model-path", str(model), "--output-dir", str(self.root / "generated"),
                "--device", "cpu", "--dtype", "float32"]
        with patch("eval.generate.make_pipeline", return_value=lambda **kwargs: Image.new("RGB", (16, 16))) as load:
            self.assertEqual(generate.main(args), 0)
            self.assertEqual(generate.main(args), 0)
            self.assertEqual(load.call_count, 1)
            with self.assertRaisesRegex(ValueError, "settings/inputs changed"):
                generate.main(args + ["--steps", "28"])
        manifest = load_manifest(self.root / "generated/manifest.jsonl")
        self.assertEqual(manifest[0]["generation"]["steps"], 4)
        self.assertEqual(manifest[0]["generation"]["cfg"], 1)


if __name__ == "__main__":
    unittest.main()
