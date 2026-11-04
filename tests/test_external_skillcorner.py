import unittest
import numpy as np
from pulsoxi.external_skillcorner import contiguous_runs, frame_inputs, possession_group
class SkillCornerTests(unittest.TestCase):
    def test_unknown_possession_is_not_dead_ball(self): self.assertIsNone(possession_group({"possession":{}}))
    def test_unknown_breaks_reference_runs(self): self.assertEqual(contiguous_runs(["home","home",None,"away"]),[("home",0,1),("away",3,3)])
    def test_features_do_not_use_possession(self):
        base={"frame":1,"period":1,"ball_data":{"x":1,"y":2,"is_detected":1},"player_data":[{"x":3,"y":4,"player_id":7,"is_detected":1}]}
        a=frame_inputs(dict(base,possession={"group":"home"}),{7:"home"}); b=frame_inputs(dict(base,possession={"group":"away"}),{7:"home"})
        np.testing.assert_array_equal(a["ball_xy"],b["ball_xy"]); np.testing.assert_array_equal(a["player_xy"],b["player_xy"])
