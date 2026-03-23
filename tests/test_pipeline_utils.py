from __future__ import annotations

import unittest

from sam3_video_poc.targeting import normalize_click_point, pick_target_obj_id


class PipelineUtilsTest(unittest.TestCase):
    def test_normalize_click_point(self) -> None:
        self.assertEqual(normalize_click_point(50, 25, 200, 100), [[0.25, 0.25]])

    def test_pick_target_obj_id_prefers_highest_score(self) -> None:
        outputs = {
            "out_obj_ids": [4, 9, 2],
            "out_probs": [0.6, 0.95, 0.5],
        }
        obj_id, score = pick_target_obj_id(outputs)
        self.assertEqual(obj_id, 9)
        self.assertAlmostEqual(score or 0.0, 0.95, places=6)

    def test_pick_target_obj_id_empty(self) -> None:
        self.assertEqual(pick_target_obj_id({}), (None, None))


if __name__ == "__main__":
    unittest.main()
