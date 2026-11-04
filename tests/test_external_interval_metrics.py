import unittest
from pulsoxi.external_interval_metrics import spells, interval_iou, overlap_scores, best_matches
class IntervalMetricTests(unittest.TestCase):
 def test_perfect(self):
  r=[("H",0,10),("A",10,20)]; self.assertEqual(overlap_scores(r,r)[.5]["f1"],1.0)
 def test_split_reduces_f1(self):
  self.assertLess(overlap_scores([("H",0,20)],[ ("H",0,10),("H",10,20)])[.5]["f1"],1.0)
 def test_team_mismatch(self): self.assertEqual(interval_iou(("H",0,10),("A",0,10)),0.0)
 def test_unknown_breaks_spells(self): self.assertEqual(spells(["H","H","X","H"]),[("H",0,2),("H",3,4)])
 def test_one_to_one(self):
  r=[("H",0,10),("H",10,20)]; p=[("H",0,20)]; self.assertEqual(len(best_matches(r,p,.3)),1)
 def test_threshold(self): self.assertGreater(overlap_scores([("H",0,10)],[ ("H",0,6)])[.3]["f1"],overlap_scores([("H",0,10)],[ ("H",0,6)])[.7]["f1"])
