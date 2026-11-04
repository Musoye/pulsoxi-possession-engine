import unittest
import numpy as np
from pulsoxi.turnover_stabilizer import stabilize_turnovers
class StabilizerTests(unittest.TestCase):
 def args(self,n): return dict(periods=np.ones(n),frame_ids=np.arange(n),fps=10)
 def test_one_frame(self): self.assertEqual(stabilize_turnovers(np.array([0,1,0]),max_reversal_seconds=.3,**self.args(3)).tolist(),[0,0,0])
 def test_two_frame(self): self.assertEqual(stabilize_turnovers(np.array([0,1,1,0]),max_reversal_seconds=.3,**self.args(4)).tolist(),[0,0,0,0])
 def test_five_frame(self): self.assertEqual(stabilize_turnovers(np.array([0]+[1]*5+[0]),max_reversal_seconds=.5,**self.args(7)).tolist(),[0]*7)
 def test_ten_frame(self): self.assertEqual(stabilize_turnovers(np.array([0]+[1]*10+[0]),max_reversal_seconds=1.0,**self.args(12)).tolist(),[0]*12)
 def test_just_below(self): self.assertEqual(stabilize_turnovers(np.array([0]+[1]*4+[0]),max_reversal_seconds=.5,**self.args(6)).tolist(),[0]*6)
 def test_exact_threshold(self): self.assertEqual(stabilize_turnovers(np.array([0]+[1]*5+[0]),max_reversal_seconds=.5,**self.args(7)).tolist(),[0]*7)
 def test_above_threshold(self): self.assertEqual(stabilize_turnovers(np.array([0]+[1]*6+[0]),max_reversal_seconds=.5,**self.args(8)).tolist(),[0,1,1,1,1,1,1,0])
 def test_multiple(self): self.assertEqual(stabilize_turnovers(np.array([0,1,0,1,0]),max_reversal_seconds=.2,**self.args(5)).tolist(),[0]*5)
 def test_long_preserved(self): self.assertEqual(stabilize_turnovers(np.array([0,1,1,1,0]),max_reversal_seconds=.2,**self.args(5)).tolist(),[0,1,1,1,0])
 def test_period_boundary(self): self.assertEqual(stabilize_turnovers(np.array([0,1,0]),max_reversal_seconds=.3,periods=np.array([1,1,2]),frame_ids=np.arange(3),fps=10).tolist(),[0,1,0])
 def test_gap(self): self.assertEqual(stabilize_turnovers(np.array([0,1,0]),max_reversal_seconds=.3,periods=np.ones(3),frame_ids=np.array([0,1,3]),fps=10).tolist(),[0,1,0])
 def test_d_x(self): self.assertEqual(stabilize_turnovers(np.array([0,2,0]),max_reversal_seconds=.3,**self.args(3)).tolist(),[0,2,0])
 def test_idempotent(self):
  x=np.array([0,1,1,0]); y=stabilize_turnovers(x, max_reversal_seconds=.3, **self.args(4)); np.testing.assert_array_equal(y,stabilize_turnovers(y,max_reversal_seconds=.3,**self.args(4)))
 def test_25fps(self): self.assertEqual(stabilize_turnovers(np.array([0]+[1]*12+[0]),max_reversal_seconds=.5,periods=np.ones(14),frame_ids=np.arange(14),fps=25).tolist(),[0]*14)
