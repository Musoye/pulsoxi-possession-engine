import unittest
import numpy as np
from pulsoxi.external_metrica import event_frame, nearest_team
class ExternalMetricaTests(unittest.TestCase):
 def test_event_frame_converts_one_based_frame(self):
  self.assertEqual(event_frame({'Start Frame':'26'}),25)
 def test_nearest_team_uses_coordinates_not_slot_identity(self):
  home={0:(1,np.array([.5,.5]),np.array([[.49,.5]]))}; away={0:(1,np.array([.5,.5]),np.array([[.8,.8]]))}
  team,margin=nearest_team(0,home,away); self.assertEqual(team,'Home'); self.assertGreater(margin,0)
 def test_missing_ball_is_unscored(self):
  home={0:(1,np.array([np.nan,np.nan]),np.array([[.49,.5]]))}; away={0:(1,np.array([.5,.5]),np.array([[.8,.8]]))}
  self.assertIsNone(nearest_team(0,home,away))
if __name__=='__main__': unittest.main()
