"""Verify the user's Cartesian-grid inference exports the actual prediction for evaluation."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

import infer
from eval.common import inspect_inputs, load_manifest, read_image
from eval.run import input_rows, parser as eval_parser


class InferExportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.references = self.root / "references"
        self.references.mkdir()
        for i in range(2):
            Image.new("RGB", (32, 16), (100 + i, 0, 0)).save(self.references / f"content_{i:02d}.png")
        for j in range(3):
            Image.new("RGB", (16, 32), (0, 100 + j, 0)).save(self.references / f"style_{j:02d}.png")
        self.output = self.root / "outputs"
        self.args = ["--references-dir", str(self.references), "--content-count", "2", "--style-count", "3",
                     "--model-path", str(self.root / "model"), "--no-lora", "--device", "cpu",
                     "--output-dir", str(self.output), "--width", "16", "--height", "16"]

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def prediction(**kwargs):
        content, style = kwargs["edit_image"]
        # Non-square output catches accidental scoring of a resized preview.
        return Image.new("RGB", (13, 19), (content.getpixel((0, 0))[0], style.getpixel((0, 0))[1], 255))

    def test_grid_manifest_matches_lossless_predictions_not_collages(self):
        with patch("infer.make_pipeline", return_value=self.prediction) as loader:
            self.assertEqual(infer.main(self.args), 0)
            self.assertEqual(infer.main(self.args), 0)
            loader.assert_called_once()
        args = eval_parser().parse_args(["--manifest", str(self.output / "manifest.jsonl"),
                                        "--output-dir", str(self.output / "scores")])
        rows = input_rows(args)
        self.assertEqual(len(rows), 6)
        self.assertEqual(len({r["id"] for r in rows}), 6)
        inspect_inputs(rows)
        for i in range(2):
            for j in range(3):
                row = rows[i * 3 + j]
                self.assertEqual(Path(row["content"]).name, f"content_{i:02d}.png")
                self.assertEqual(Path(row["style"]).name, f"style_{j:02d}.png")
                self.assertEqual(Path(row["output"]).name, f"out_{i}_{j}.png")
                result = read_image(row["output"])
                self.assertEqual(result.size, (13, 19))
                self.assertEqual(result.getpixel((0, 0)), (100 + i, 100 + j, 255))
                self.assertEqual(read_image(self.output / row["preview"]).size, (68, 56))
                self.assertFalse(row["input_errors"])
        with self.assertRaisesRegex(ValueError, "settings changed"):
            infer.main(self.args + ["--steps", "28"])
        (self.root / "model").mkdir()
        (self.root / "model/config.json").write_text('{"model": "changed"}')
        with self.assertRaisesRegex(ValueError, "settings changed"):
            infer.main(self.args)

    def test_failed_generation_stays_in_manifest_and_can_resume(self):
        def fail_some(**kwargs):
            if kwargs["edit_image"][1].getpixel((0, 0))[1] == 101:
                raise RuntimeError("simulated generation failure")
            return self.prediction(**kwargs)
        with patch("infer.make_pipeline", return_value=fail_some):
            self.assertEqual(infer.main(self.args), 1)
        rows = load_manifest(self.output / "manifest.jsonl")
        inspect_inputs(rows)
        self.assertEqual(sum(bool(r["input_errors"]) for r in rows), 2)
        with patch("infer.make_pipeline", return_value=self.prediction):
            self.assertEqual(infer.main(self.args), 0)
        state = json.loads((self.output / "infer_state.json").read_text())
        self.assertTrue(all(value["status"] == "ok" for value in state.values()))

    def test_reference_pattern_supports_indices_above_nine(self):
        args = infer.parser().parse_args(["--content-count", "11", "--style-count", "11"])
        row = infer.build_cases(args)[-1]
        self.assertEqual(Path(row["content"]).name, "content_10.png")
        self.assertEqual(Path(row["style"]).name, "style_10.png")
        self.assertEqual(row["output"], "out_10_10.png")


if __name__ == "__main__":
    unittest.main()
