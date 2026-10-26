import unittest
import numpy as np
from pulsoxi.diagnostics import possession_spans, duration_summary, reappearance_events, assert_duration_invariants
class DiagnosticTests(unittest.TestCase):
 def test_reappearance_only_after_gap_and_not_half_start(self):
  v=np.array([1,1,0,0,1,1,0,1,1],bool); f=np.arange(9)
  self.assertEqual(reappearance_events(v),[(4,2),(7,1)])
 def test_half_boundary_does_not_bridge_gap(self):
  v=np.array([1,0,1,1],bool); f=np.array([90650,90651,110808,110809])
  self.assertEqual(reappearance_events(v,f),[])
 def test_nested_duration_counts(self):
  y=np.array(['H']*10+['A']*20+['D']*2+['H']*5)
  s=duration_summary(y); assert_duration_invariants(s); self.assertEqual(s['denominator'],3)
if __name__=='__main__': unittest.main()