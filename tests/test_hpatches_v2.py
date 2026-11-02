"""CPU checks for the v2 additions: refinement, perturbation, original-resolution metrics, hybrids."""

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

import hpatches_spatial_eval as experiment
from hpatches_spatial_eval import (apply_perturbation, corner_error, load_ground_truth,
                                   make_perturbation, original_match_quality, parabola_peak,
                                   parse_visualize_pairs, prepare_image, project, refine_targets,
                                   run, stable_rng, to_original_homography)


def textured(seed=42, shape=(320, 360)):
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 256, (*shape, 3), dtype=np.uint8)
    for _ in range(60):
        center = (int(rng.integers(10, shape[1] - 10)), int(rng.integers(10, shape[0] - 10)))
        cv2.circle(image, center, int(rng.integers(3, 14)), (0, 0, 0), -1)
    return image


def write_sequence(root, name, source, target, homography):
    folder = root / name
    folder.mkdir(parents=True)
    cv2.imwrite(str(folder / "1.ppm"), source)
    for number in range(2, 7):
        cv2.imwrite(str(folder / f"{number}.ppm"), target)
        np.savetxt(folder / f"H_1_{number}", homography)


def base_args(root, output, methods, **extra):
    values = dict(data=root, categories=["v"], max_sequences=1, dry_run=False, methods=methods,
                  seed=7, model="mock", size=448, ransac_threshold=3., output=output, visualize=0)
    values.update(extra)
    return SimpleNamespace(**values)


class TestRefinement(unittest.TestCase):
    def test_parabola_recovers_exact_vertex(self):
        x0 = 0.3
        samples = [-(x - x0) ** 2 for x in (-1, 0, 1)]
        self.assertAlmostEqual(parabola_peak(*samples), x0, places=9)
        self.assertEqual(parabola_peak(0, 0, 0), 0.0)          # flat: no move
        self.assertEqual(parabola_peak(np.nan, 1, 0), 0.0)     # missing neighbour

    def test_quadratic_refinement_moves_point_inside_patch(self):
        g, patch_size = 6, 16
        grid = np.eye(g * g, dtype=np.float32).reshape(g, g, g * g)
        valid = np.ones((g, g), bool)
        row, col, oy, ox = 2, 3, -0.2, 0.35
        query = np.array([-((c - (col + ox)) ** 2 + (r - (row + oy)) ** 2)
                          for r in range(g) for c in range(g)], dtype=np.float32)
        start = np.array([[(col + .5) * patch_size, (row + .5) * patch_size]], np.float32)
        refined = refine_targets(query[None], start, (grid, valid), patch_size, "quadratic")
        self.assertTrue(np.allclose(refined[0], start[0] + [ox * patch_size, oy * patch_size], atol=1e-4))
        soft = refine_targets(query[None], start, (grid, valid), patch_size, "soft", temperature=0.05)
        self.assertGreater(soft[0, 0], start[0, 0])
        self.assertLess(soft[0, 1], start[0, 1])


class TestPerturbationAndOriginalMetrics(unittest.TestCase):
    def test_perturbation_is_composed_into_ground_truth(self):
        image = textured()
        perturb = make_perturbation(image.shape, "homography", 12, stable_rng(1, "x", 2))
        warped, mask = apply_perturbation(image, perturb)
        self.assertEqual(mask.shape, image.shape[:2])
        self.assertLess(mask.mean(), 1.0)
        _, t0, _ = prepare_image(image)
        _, t1, valid1 = prepare_image(warped, 448, mask)
        self.assertLess(valid1.mean(), prepare_image(warped)[2].mean())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "H_1_2"
            np.savetxt(path, np.eye(3))
            truth = load_ground_truth(path, t0, t1, perturb)
        point = np.array([[100., 80.]])
        self.assertTrue(np.allclose(project(truth, project(t0, point)), project(t1, project(perturb, point))))
        self.assertIsNone(make_perturbation(image.shape, "none", 7, None))

    def test_shift_rng_is_deterministic_per_pair(self):
        a = make_perturbation((10, 10), "shift", 7, stable_rng(5, "i_x", 3))
        b = make_perturbation((10, 10), "shift", 7, stable_rng(5, "i_x", 3))
        c = make_perturbation((10, 10), "shift", 7, stable_rng(5, "i_x", 4))
        self.assertTrue(np.array_equal(a, b))
        self.assertFalse(np.array_equal(a, c))

    def test_corner_error_and_original_conversion(self):
        source = np.zeros((300, 400, 3), np.uint8)
        target = np.zeros((600, 800, 3), np.uint8)
        _, t0, _ = prepare_image(source)
        _, t1, _ = prepare_image(target)
        h = np.array([[2, 0, 5], [0, 2, 3], [0, 0, 1.]])
        canvas_h = t1 @ h @ np.linalg.inv(t0)
        original = to_original_homography(canvas_h, t0, t1)
        self.assertAlmostEqual(corner_error(original, h, source.shape), 0, places=6)
        shifted = np.array([[1, 0, 4], [0, 1, 0], [0, 0, 1.]]) @ h
        self.assertAlmostEqual(corner_error(shifted, h, source.shape), 4, places=6)
        self.assertIsNone(corner_error(None, h, source.shape))

    def test_original_mma_uses_original_pixels(self):
        source = np.zeros((224, 448, 3), np.uint8)   # canvas scale 1.0 horizontally
        _, t0, _ = prepare_image(source)
        truth = np.eye(3)
        src_canvas = project(t0, np.array([[100., 50.], [200., 60.]]))
        tgt_canvas = src_canvas + [[0, 0], [3, 0]]
        counts = original_match_quality(src_canvas, tgt_canvas, t0, t0, truth, np.ones((224, 448), np.uint8))
        self.assertEqual(counts["correct_orig_1px"], 1)
        self.assertEqual(counts["correct_orig_3px"], 2)

    def test_visualize_pairs_parser(self):
        self.assertEqual(parse_visualize_pairs(["v_bark:3"]), {("v_bark", 3)})
        with self.assertRaises(ValueError):
            parse_visualize_pairs(["v_bark"])


class TestEndToEndV2(unittest.TestCase):
    def test_sift_with_homography_perturbation_still_registers(self):
        source = textured()
        homography = np.array([[1, 0.016, 11], [-0.012, 1, 8], [0.00003, 0.00005, 1]], np.float64)
        target = cv2.warpPerspective(source, homography, (360, 320))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            write_sequence(root, "v_synthetic", source, target, homography)
            output = Path(tmp) / "out"
            run(base_args(root, output, ["SIFT"], model=None, target_perturb="homography", perturb_px=10,
                          visualize_pairs=["v_synthetic:4"]))
            report = json.loads((output / "summary.json").read_text())
            self.assertEqual(report["summary"]["v/SIFT"]["success_at_5px"], 1.)
            original = report["original_resolution"]["v/SIFT"]
            self.assertEqual(original["homography_accuracy_3px"], 1.)
            self.assertGreater(original["mma_micro"]["3"], .5)
            self.assertTrue((output / "v_synthetic_1_4_SIFT.jpg").is_file())

    def test_identity_ground_truth_is_solved_by_grid_coincidence(self):
        """Documents the HPatches i_* artefact: identical patch grids give a perfect identity H."""
        source = np.full((320, 360, 3), 120, np.uint8)
        _, _, valid = prepare_image(source)
        grid = 448 // 16
        ys, xs = np.indices((grid, grid))
        centers = np.column_stack(((xs.ravel() + .5) * 16, (ys.ravel() + .5) * 16)).astype(np.float32)
        idx = centers.astype(int)
        keep = valid[idx[:, 1], idx[:, 0]] > 0

        class GridDino:   # any features that match identical grid cells to each other
            def __init__(self, *a):
                self.patch_size = 16

            def encode(self, image, valid_mask):
                return np.eye(len(centers), dtype=np.float32)[keep], centers[keep]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            write_sequence(root, "i_flat", source, source, np.eye(3))
            args = base_args(root, Path(tmp) / "out", ["DINOv3"], categories=["i"])
            fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
            with patch.dict(sys.modules, {"torch": fake_torch}), patch.object(experiment, "DinoEncoder", GridDino):
                report = run(args)
            self.assertLess(report["summary"]["i/DINOv3"]["median_error_registered_px"], 1e-6)

    def test_dino_guided_sift_and_separate_ransac(self):
        source = textured(3)
        homography = np.array([[1, 0, 12], [0, 1, 7], [0, 0, 1]], np.float64)
        target = cv2.warpPerspective(source, homography, (360, 320))
        _, transform, _ = prepare_image(source)
        model_h = transform @ homography @ np.linalg.inv(transform)
        points = np.array([(x, y) for x in (100., 170., 240., 310.)
                           for y in (100., 180., 260., 340.)], dtype=np.float32)

        class FakeDino:
            def __init__(self, *a):
                self.patch_size = 16
                self.calls = 0
                self.last_grid = None

            def encode(self, image, valid):
                self.calls += 1
                g = 448 // 16
                self.last_grid = (np.ones((g, g, 4), np.float32) / 2, np.ones((g, g), bool))
                features = np.eye(len(points), dtype=np.float32)
                return features, points if self.calls == 1 else project(model_h, points).astype(np.float32)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            write_sequence(root, "v_shift", source, target, homography)
            args = base_args(root, Path(tmp) / "out", ["DINOv3", "DINOguidedSIFT", "SIFTdinoFilter", "SIFT"],
                             dino_ransac_threshold=8.0)
            fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
            with patch.dict(sys.modules, {"torch": fake_torch}), patch.object(experiment, "DinoEncoder", FakeDino):
                report = run(args)
            self.assertEqual(report["dino_ransac_threshold_model_px"], 8.0)
            self.assertEqual(report["summary"]["v/DINOguidedSIFT"]["success_at_5px"], 1.)
            self.assertIn("DINOguidedSIFT_vs_SIFT", report["paired_comparisons"])
            self.assertIn("v/SIFTdinoFilter", report["summary"])


if __name__ == "__main__":
    unittest.main()
