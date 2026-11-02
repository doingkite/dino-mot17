"""CPU end-to-end checks for SPair-71k, TAP-Vid-DAVIS and MOT17-GT tools on synthetic data."""

import json
import pickle
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from dense_correspondence import Geometry, nearest_patch, sample_features

ROOT = Path(__file__).resolve().parents[1]


def textured(seed, shape=(240, 320)):
    rng = np.random.default_rng(seed)
    image = cv2.GaussianBlur(rng.integers(0, 256, (*shape, 3), dtype=np.uint8), (3, 3), 0)
    for _ in range(80):
        c = (int(rng.integers(5, shape[1] - 5)), int(rng.integers(5, shape[0] - 5)))
        cv2.circle(image, c, int(rng.integers(2, 9)), tuple(int(v) for v in rng.integers(0, 256, 3)), -1)
    return image


def run_script(*args):
    out = subprocess.run([sys.executable, str(ROOT / args[0]), *map(str, args[1:])],
                         capture_output=True, text=True, cwd=ROOT)
    if out.returncode:
        raise AssertionError(out.stdout + out.stderr)
    return out.stdout


class TestDenseUtilities(unittest.TestCase):
    def test_geometry_roundtrip(self):
        for mode in ("letterbox", "stretch"):
            geo = Geometry((240, 320), 448, mode)
            xy = np.array([[10.5, 20.0], [319, 239]])
            self.assertTrue(np.allclose(geo.to_original(geo.to_canvas(xy)), xy))

    def test_nearest_patch_finds_matching_feature(self):
        g, c = 4, 16
        rng = np.random.default_rng(0)
        grid = rng.normal(size=(g, g, c))
        grid /= np.linalg.norm(grid, axis=2, keepdims=True)
        valid = np.ones((g, g), bool)
        query = sample_features(grid, np.array([[2.5 * 16, 1.5 * 16]]), 16)   # centre of patch (row 1, col 2)
        xy, _ = nearest_patch(query, grid, valid, 16)
        self.assertTrue(np.allclose(xy[0], [2.5 * 16, 1.5 * 16]))
        region = np.zeros((g, g), bool)
        region[3, 3] = True
        xy, _ = nearest_patch(query, grid, valid, 16, region=region)
        self.assertTrue(np.allclose(xy[0], [3.5 * 16, 3.5 * 16]))


class TestSPair(unittest.TestCase):
    def test_sift_and_bbox_prior_on_translated_instance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "SPair-71k"
            (root / "JPEGImages" / "cat").mkdir(parents=True)
            (root / "PairAnnotation" / "test").mkdir(parents=True)
            src = textured(1)
            shift = (9, 6)
            trg = cv2.warpAffine(src, np.float32([[1, 0, shift[0]], [0, 1, shift[1]]]), (320, 240))
            cv2.imwrite(str(root / "JPEGImages" / "cat" / "a.jpg"), src, [cv2.IMWRITE_JPEG_QUALITY, 98])
            cv2.imwrite(str(root / "JPEGImages" / "cat" / "b.jpg"), trg, [cv2.IMWRITE_JPEG_QUALITY, 98])
            kps = [[60, 60], [150, 100], [220, 150], [100, 180]]
            item = {"src_imname": "a.jpg", "trg_imname": "b.jpg", "category": "cat",
                    "src_kps": kps, "trg_kps": [[x + shift[0], y + shift[1]] for x, y in kps],
                    "src_bndbox": [40, 40, 260, 200], "trg_bndbox": [49, 46, 269, 206],
                    "viewpoint_variation": 0, "scale_variation": 0}
            (root / "PairAnnotation" / "test" / "000001-a-b_cat.json").write_text(json.dumps(item))
            out = Path(tmp) / "out"
            run_script("spair_eval.py", "--data", root, "--methods", "SIFT", "bboxprior", "--output", out)
            overall = json.loads((out / "summary.json").read_text())["overall"]
            self.assertEqual(overall["bboxprior"]["pck_0.05"], 1.0)
            self.assertGreaterEqual(overall["SIFT"]["pck_0.05"], 0.75)


class TestTapVid(unittest.TestCase):
    def test_static_baseline_degrades_while_sift_tracks(self):
        base = textured(2, (300, 300))
        frames, T = [], 6
        for t in range(T):
            m = np.float32([[1, 0, -5 * t], [0, 1, -3 * t]])
            frames.append(cv2.cvtColor(cv2.warpAffine(base, m, (300, 300))[:256, :256], cv2.COLOR_BGR2RGB))
        starts = np.array([[60., 60.], [128., 140.], [190., 90.], [100., 200.]])
        points = np.stack([[(p - [5 * t, 3 * t]) / 256 for t in range(T)] for p in starts])
        data = {"synthetic": {"video": np.stack(frames), "points": points,
                              "occluded": np.zeros(points.shape[:2], bool)},
                "pixels_occlusions_key": {"video": np.stack(frames), "points": points * 256,
                                          "occlusions": np.zeros(points.shape[:2], bool)}}
        with tempfile.TemporaryDirectory() as tmp:
            pkl = Path(tmp) / "tapvid_davis.pkl"
            pkl.write_bytes(pickle.dumps(data))
            out = Path(tmp) / "out"
            run_script("tapvid_davis_eval.py", "--pkl", pkl, "--methods", "SIFT", "static", "--output", out)
            overall = json.loads((out / "summary.json").read_text())["overall"]
            self.assertGreater(overall["SIFT"]["acc_4px"], 0.9)
            self.assertLess(overall["static"]["acc_4px"], 0.3)


class TestMOT17GT(unittest.TestCase):
    def test_roi_pairs_and_reid_baselines(self):
        with tempfile.TemporaryDirectory() as tmp:
            seq = Path(tmp) / "MOT17-09-FRCNN"
            (seq / "img1").mkdir(parents=True)
            (seq / "gt").mkdir()
            colours = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
            lines = []
            for f in range(1, 41):
                image = np.full((240, 320, 3), 128, np.uint8)
                for pid, colour in enumerate(colours, start=1):
                    x, y = 20 + 90 * (pid - 1) + f, 60
                    cv2.rectangle(image, (x, y), (x + 40, y + 100), colour, -1)
                    lines.append(f"{f},{pid},{x},{y},40,100,1,1,1.0")
                cv2.imwrite(str(seq / "img1" / f"{f:06d}.jpg"), image)
            (seq / "gt" / "gt.txt").write_text("\n".join(lines))
            roi = Path(tmp) / "roi_pairs.json"
            run_script("mot17_gt_tools.py", "roi-pairs", "--sequence-dir", seq, "--gaps", "5", "30",
                       "--per-gap", "3", "--output", roi)
            pairs = json.loads(roi.read_text())
            self.assertEqual(len(pairs), 6)
            self.assertEqual(pairs[0]["target_frame"] - pairs[0]["source_frame"], pairs[0]["gap"])
            out = Path(tmp) / "reid"
            run_script("mot17_gt_tools.py", "reid", "--sequence-dir", seq, "--gaps", "5", "--no-dino",
                       "--queries-per-gap", "20", "--output", out)
            gap = json.loads((out / "reid_summary.json").read_text())["gaps"]["5"]
            self.assertEqual(gap["rank1_hsv_histogram"], 1.0)
            self.assertAlmostEqual(gap["rank1_random_expected"], 1 / 3)


if __name__ == "__main__":
    unittest.main()
