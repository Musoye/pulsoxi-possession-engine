import unittest
import numpy as np
from pulsoxi.inference import generate_emissions, generate_emissions_e2
class E2SyntheticTests(unittest.TestCase):
 def base(self,n=8):
  x=np.zeros((n,17),float); x[:,0]=1; x[:,3]=5; x[:,4]=5; x[:,9]=2; x[:,10]=6; x[:,16]=1; return x
 def scenario(self, values):
  x=self.base(len(values))
  for i,team in enumerate(values): x[i,9:11]=[2,6] if team=='H' else [6,2]
  return x
 def test_e2_persistence_reduces_isolated_flip(self):
  x=self.base(); x[3,9:11]=[6,2]; e1,_=generate_emissions(x); e2,_=generate_emissions_e2(x); self.assertGreater(e1[3,1],e1[3,0]); self.assertLess(e2[3,1]/e2[3,0],e1[3,1]/e1[3,0])
 def test_missing_ball_is_unknown_and_neutral_dead(self):
  x=self.base(); x[3,0]=0; x[3,9:11]=np.nan; e,k=generate_emissions_e2(x); self.assertFalse(k[3]); self.assertLess(float(e[3,2]),.5)
 def test_invalid_motion_and_discontinuity_are_not_strong_evidence(self):
  x=self.base(); x[:,14]=np.nan; x[:,16]=0; x[3,13]=40; e,_=generate_emissions_e2(x); self.assertLess(abs(np.log(e[3,0]/e[3,1])),2.0)
 def test_half_reset_fixture_has_independent_input(self):
  x=self.scenario(['H','H','A','A']); e,k=generate_emissions_e2(x); self.assertTrue(np.all(k)); self.assertEqual(list(np.argmax(e[:,:2],axis=1)),[0,0,1,1])
 def test_successful_interception_both_models_switch(self):
  x=self.scenario(['H','H','A','A']); e1,_=generate_emissions(x); e2,_=generate_emissions_e2(x); self.assertEqual(list(np.argmax(e1[:,:2],axis=1)),[0,0,1,1]); self.assertEqual(list(np.argmax(e2[:,:2],axis=1)),[0,0,1,1])
 def test_defender_pressure_does_not_change_control(self):
  x=self.scenario(['H']*4); x[2,14]=9; e1,_=generate_emissions(x); e2,_=generate_emissions_e2(x); self.assertTrue(np.all(np.argmax(e1[:,:2],axis=1)==0)); self.assertTrue(np.all(np.argmax(e2[:,:2],axis=1)==0))
 def test_brief_deflection_returns_to_original_control(self):
  x=self.scenario(['H','H','A','H','H']); e1,_=generate_emissions(x); e2,_=generate_emissions_e2(x); self.assertEqual(int(np.argmax(e1[2,:2])),1); self.assertEqual(int(np.argmax(e2[2,:2])),1); self.assertEqual(int(np.argmax(e2[3,:2])),0)
 def test_teammate_pass_remains_same_team(self):
  x=self.scenario(['H']*4); x[:,14]=3; e1,_=generate_emissions(x); e2,_=generate_emissions_e2(x); self.assertTrue(np.all(np.argmax(e1[:,:2],axis=1)==0)); self.assertTrue(np.all(np.argmax(e2[:,:2],axis=1)==0))
 def test_rapid_turnover_and_high_speed_discontinuity(self):
  x=self.scenario(['H','A','A','A']); x[1,13]=12; e1,_=generate_emissions(x); e2,_=generate_emissions_e2(x); self.assertEqual(int(np.argmax(e1[1,:2])),1); self.assertEqual(int(np.argmax(e2[1,:2])),1); x[1,13]=40; e2b,_=generate_emissions_e2(x); self.assertLess(abs(np.log(e2b[1,0]/e2b[1,1])),abs(np.log(e2[1,0]/e2[1,1])))
 def test_dead_ball_restart_and_half_isolation(self):
  x=self.scenario(['H']*4); ev=[{'code':'SET PIECES','start_frame':2,'end_frame':2}]; e1,_=generate_emissions(x,smart_tagging_events=ev); e2,_=generate_emissions_e2(x,smart_tagging_events=ev); self.assertGreater(e1[2,2],e1[2,0]); self.assertGreater(e2[2,2],e2[2,0]); self.assertEqual(int(np.argmax(e1[3,:2])),0); self.assertEqual(int(np.argmax(e2[3,:2])),0)
if __name__=='__main__': unittest.main()
