import json
import tempfile
import unittest
from pathlib import Path

from mot17_dinov3_point_match import distance_metrics, load_pairs, valid_patch_mask
from mot17_dinov3_batch import summarize


class TestMOT17Pipeline(unittest.TestCase):
    def test_letterbox_mask_excludes_source_padding_for_reverse(self):
        mask = valid_patch_mask((4, 4), (1.0, 16, 0, 32, 64), 16)
        self.assertEqual(mask.sum(), 8)
        self.assertFalse(mask[:, 0].any())
        self.assertFalse(mask[:, 3].any())

    def test_model_pixel_error_uses_uniform_letterbox_scale(self):
        metrics = distance_metrics((10, 60), (2, 54), (100, 50, 3), 4.48, 448)
        self.assertAlmostEqual(metrics["pixel_error_original"], 10)
        self.assertAlmostEqual(metrics["pixel_error_model_input"], 44.8)
        self.assertAlmostEqual(metrics["pck_normalized_error"], .1)
        self.assertTrue(metrics["pck_at_010"])

    def test_rejects_missing_and_noninteger_annotations(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "pairs.json"
            path.write_text(json.dumps([{"source_frame": 1}]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing target_frame"):
                load_pairs(path)
            pair = {"source_frame": 1, "target_frame": 2,
                    "source_roi": [0, 0, 10, 10], "target_roi": [0, 0, 10, 10],
                    "source_point": [2, 2], "target_point": [2, 2]}
            path.write_text(json.dumps([pair]), encoding="utf-8")
            self.assertEqual(load_pairs(path)[0], pair)
            pair["source_point"] = [2.5, 2]
            path.write_text(json.dumps([pair]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source_point"):
                load_pairs(path)

    def test_summary_counts_point_and_frame_pair_separately(self):
        row = {"source_frame": 61, "target_frame": 76, "person_id": "", "source_roi": [0,0,10,10], "target_roi": [1,0,10,10], "pixel_error_original": 10,
               "pck_at_005": False, "pck_at_010": True,
               "within_16px_model_input": True, "cycle_consistent_strict": False,
               "cycle_consistent_within_one_patch": True,
               "baseline_static_pck_at_010": False, "baseline_center_pck_at_010": False}
        result = summarize([row, {**row, "pck_at_010": False}])
        self.assertEqual(result["n_points"], 2)
        self.assertEqual(result["n_frame_pairs"], 1)
        self.assertEqual(result["pck_at_010"], .5)


if __name__ == "__main__":
    unittest.main()
