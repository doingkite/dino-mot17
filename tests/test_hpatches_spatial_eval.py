"""Geometry and end-to-end CPU checks for the HPatches evaluation protocol."""

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from hpatches_spatial_eval import (aggregate, aggregate_match_quality, discover_pairs,
                                   geometry_error, load_ground_truth, match_quality,
                                   mutual_patch_matches, prepare_image, project, run)


class TestHPatchesSpatialEvaluation(unittest.TestCase):
    def test_letterbox_transforms_original_ground_truth_into_common_canvas(self):
        source = np.zeros((120, 200, 3), np.uint8)
        target = np.zeros((240, 300, 3), np.uint8)
        _, t0, valid0 = prepare_image(source, 448)
        _, t1, valid1 = prepare_image(target, 448)
        h_original = np.array([[1.3, 0, 8], [0, 1.4, 15], [0, 0, 1.0]])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "H_1_2"
            np.savetxt(path, h_original)
            expected = load_ground_truth(path, t0, t1)
        point = np.array([[65., 40.]])
        self.assertTrue(np.allclose(project(expected, project(t0, point)),
                                    project(t1, project(h_original, point))))
        self.assertAlmostEqual(geometry_error(expected, expected, valid0, valid1), 0., places=7)

    def test_mutual_matching_and_spatial_spread(self):
        features = np.eye(5, dtype=np.float32)
        xy = np.array([[15, 15], [80, 15], [160, 25], [20, 90], [130, 130]], np.float32)
        a, b, scores = mutual_patch_matches(features, xy, features, xy + 4, 224)
        self.assertEqual(len(scores), 5)
        self.assertTrue(np.allclose(b - a, 4))

    def test_missing_registration_is_not_dropped_from_success_denominator(self):
        rows = [{"category": "v", "method": "SIFT", "median_overlap_error_px": 1.0},
                {"category": "v", "method": "SIFT", "median_overlap_error_px": None}]
        self.assertEqual(aggregate(rows)["v/SIFT"]["success_at_5px"], 0.5)

    def test_match_precision_counts_wrong_and_invisible_proposals(self):
        valid = np.ones((32, 32), np.uint8)
        source = np.array([[5, 5], [10, 10], [30, 30]], np.float32)
        target = np.array([[5, 5], [18, 10], [30, 30]], np.float32)
        truth = np.array([[1, 0, 2], [0, 1, 0], [0, 0, 1]], np.float64)
        score = match_quality(source, target, truth, valid)
        self.assertEqual(score["gt_visible_matches"], 2)
        self.assertEqual(score["correct_matches_5px"], 1)
        self.assertEqual(score["correct_matches_10px"], 2)
        self.assertEqual(score["correct_matches_16px"], 2)
        self.assertAlmostEqual(score["match_precision_5px"], 1 / 3)
        self.assertAlmostEqual(score["match_precision_10px"], 2 / 3)
        self.assertIsNone(match_quality(source[:0], target[:0], truth, valid)["match_precision_5px"])

    def test_match_quality_aggregation_does_not_hide_sparse_pairs(self):
        rows = [{"category": "v", "method": "DINOv3", "matches": 10,
                 "gt_visible_matches": 10, "correct_matches_5px": 10,
                 "correct_matches_10px": 10, "match_precision_5px": 1.,
                 "match_precision_10px": 1.},
                {"category": "v", "method": "DINOv3", "matches": 1,
                 "gt_visible_matches": 1, "correct_matches_5px": 0,
                 "correct_matches_10px": 0, "match_precision_5px": 0.,
                 "match_precision_10px": 0.}]
        for row in rows:
            row["correct_matches_16px"] = row["correct_matches_10px"]
            row["match_precision_16px"] = row["match_precision_10px"]
        score = aggregate_match_quality(rows)["v/DINOv3"]
        self.assertAlmostEqual(score["micro_precision_5px"], 10 / 11)
        self.assertAlmostEqual(score["mean_pair_precision_5px"], 0.5)

    def test_sift_full_image_sequence_end_to_end(self):
        rng = np.random.default_rng(42)
        source = rng.integers(0, 256, (320, 360, 3), dtype=np.uint8)
        for i in range(50):
            center = (int(rng.integers(10, 350)), int(rng.integers(10, 310)))
            cv2.circle(source, center, int(rng.integers(3, 14)), (0, 0, 0), -1)
        homography = np.array([[1, 0.016, 11], [-0.012, 1, 8], [0.00003, 0.00005, 1]], np.float64)
        target = cv2.warpPerspective(source, homography, (360, 320))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "hpatches-sequences-release"
            folder = root / "v_synthetic"
            folder.mkdir(parents=True)
            cv2.imwrite(str(folder / "1.ppm"), source)
            for number in range(2, 7):
                cv2.imwrite(str(folder / f"{number}.ppm"), target)
                np.savetxt(folder / f"H_1_{number}", homography)
            self.assertEqual(len(discover_pairs(root)), 5)
            output = Path(tmp) / "results"
            args = SimpleNamespace(data=root, categories=["v"], max_sequences=1,
                                   dry_run=False, methods=["SIFT"], seed=7,
                                   model=None, size=448, ransac_threshold=3.,
                                   output=output, visualize=1)
            run(args)
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["summary"]["v/SIFT"]["pairs"], 5)
            self.assertEqual(summary["summary"]["v/SIFT"]["success_at_5px"], 1.)
            self.assertGreater(summary["match_quality"]["v/SIFT"]["micro_precision_5px"], 0.5)
            self.assertTrue((output / "v_synthetic_1_2_SIFT.jpg").is_file())

    def test_dino_route_uses_same_geometry_protocol(self):
        import hpatches_spatial_eval as experiment

        source = np.full((320, 360, 3), 140, np.uint8)
        homography = np.array([[1, 0, 12], [0, 1, 7], [0, 0, 1]], np.float64)
        _, transform, _ = prepare_image(source)
        model_h = transform @ homography @ np.linalg.inv(transform)
        points = np.array([(x, y) for x in (100., 170., 240., 310.)
                           for y in (100., 180., 260., 340.)], dtype=np.float32)

        class FakeDino:
            def __init__(self, name, size, device):
                self.patch_size = 16
                self.calls = 0

            def encode(self, image, valid):
                self.calls += 1
                features = np.eye(len(points), dtype=np.float32)
                return features, points if self.calls == 1 else project(model_h, points).astype(np.float32)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "full"
            folder = root / "v_mock"
            folder.mkdir(parents=True)
            cv2.imwrite(str(folder / "1.ppm"), source)
            for number in range(2, 7):
                cv2.imwrite(str(folder / f"{number}.ppm"), source)
                np.savetxt(folder / f"H_1_{number}", homography)
            args = SimpleNamespace(data=root, categories=["v"], max_sequences=1,
                                   dry_run=False, methods=["DINOv3"], seed=7,
                                   model="mock", size=448, ransac_threshold=3.,
                                   output=Path(tmp) / "result", visualize=0)
            fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
            with patch.dict(sys.modules, {"torch": fake_torch}), patch.object(experiment, "DinoEncoder", FakeDino):
                run(args)
            summary = json.loads((args.output / "summary.json").read_text())["summary"]["v/DINOv3"]
            self.assertEqual(summary["registered"], 5)
            self.assertEqual(summary["success_at_5px"], 1.)
            quality = json.loads((args.output / "summary.json").read_text())["match_quality"]["v/DINOv3"]
            self.assertGreater(quality["micro_precision_5px"], 0.99)


if __name__ == "__main__":
    unittest.main()
