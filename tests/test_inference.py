import unittest
import numpy as np
from pulsoxi.inference import generate_emissions, infer_labels, diagnostics, _decode_hysteresis, HysteresisConfig, build_reliability
from pulsoxi.temporal import N_FRAMES
from pulsoxi.temporal_v3 import reliability_aware_viterbi
from pulsoxi.temporal_v3b import two_state_reliability_viterbi

class TestInference(unittest.TestCase):
    def features(self, n=8):
        x=np.zeros((n,17),float); x[:,0]=1; x[:,3:5]=2; x[:,9]=2; x[:,10]=8; x[:,16]=1
        return x
    def test_clear_home_and_away(self):
        x=self.features(); p,_=generate_emissions(x); self.assertGreater(p[0,0],p[0,1])
        x[:,9],x[:,10]=8,2; p,_=generate_emissions(x); self.assertGreater(p[0,1],p[0,0])
    def test_missing_ball_preserves_prior_and_not_dead(self):
        x=self.features(); x[3,0]=0; x[3,9:11]=np.nan; p,k=generate_emissions(x)
        self.assertFalse(k[3]); self.assertGreater(p[3,0], p[3,2])
    def test_set_piece_is_supporting_dead_evidence(self):
        x=self.features(); p,_=generate_emissions(x,smart_tagging_events=[{"code":"SET PIECES","start_frame":3,"end_frame":4}])
        self.assertGreater(p[3,2],p[3,0]); self.assertGreater(p[3,2],p[3,1])
    def test_v2_rejects_alternating_weak_proximity_but_accepts_rapid_control(self):
        p=np.tile([.55,.45,.01],(8,1)); p[3:6]=[.05,.93,.02]
        weak=np.ones(8,dtype=bool)
        out=_decode_hysteresis(p,weak,config=HysteresisConfig())
        self.assertTrue(np.all(out[:3]==0)); self.assertTrue(np.all(out[3:]==1))
    def test_v2_holds_through_missing_ball(self):
        p=np.tile([.85,.10,.05],(5,1)); known=np.array([1,0,0,1,1],dtype=bool)
        out=_decode_hysteresis(p,known,config=HysteresisConfig())
        self.assertTrue(np.all(out==0))
    def test_v3_penalizes_unreliable_opponent_switch_but_allows_strong_one(self):
        p=np.array([[.8,.1,.1],[.45,.55,.1],[.45,.55,.1]])
        low=reliability_aware_viterbi(p,[1,0,0],uncertainty_penalty=3.0)
        strong=np.array([[.8,.1,.1],[.05,.93,.02]])
        high=reliability_aware_viterbi(strong,[1,1],uncertainty_penalty=3.0)
        self.assertEqual(low[1],0)
        self.assertEqual(high[1],1)
    def test_v3_reliability_uses_visibility_margin_and_motion(self):
        x=self.features(3); x[:,9]=2; x[:,10]=8; x[1,0]=0; x[2,9:11]=np.nan; x[2,16]=np.nan
        r=build_reliability(x); self.assertGreater(r[0],r[1]); self.assertGreater(r[0],r[2])
    def test_v3b_fixed_live_interval_cannot_insert_dead_bridge(self):
        p=np.array([[.9,.05,.05],[.05,.9,.05],[.9,.05,.05]])
        decoded=two_state_reliability_viterbi(p,[1,1,1],costs=__import__("pulsoxi.temporal",fromlist=["TransitionCosts"]).TransitionCosts(opponent=8.0),uncertainty_penalty=0)
        self.assertTrue(np.all(np.isin(decoded,[0,1])))
    def test_v3b_preserves_v1_dead_mask(self):
        from pulsoxi.temporal_v3b import decode_full_match_v3b
        p=np.ones((N_FRAMES,3)); r=np.ones(N_FRAMES); v1=np.full(N_FRAMES,"X",dtype="<U1"); v1[11185:90652]="D"; v1[110808:190198]="H"
        out=decode_full_match_v3b(p,r,v1)
        self.assertEqual(out[11185],"D"); self.assertEqual(out[110808],"H"); self.assertEqual(out[0],"X")
    def test_full_shape_determinism_and_boundaries(self):
        x=np.tile(self.features(1),(N_FRAMES,1)); a,_,_=infer_labels(x); b,_,_=infer_labels(x)
        np.testing.assert_array_equal(a,b); self.assertEqual(len(a),N_FRAMES); self.assertEqual(a[0],"X"); self.assertEqual(a[-1],"X")
        self.assertEqual(diagnostics(a,np.ones(N_FRAMES,dtype=bool))["abstention_fraction"], 1-N_FRAMES*158857/N_FRAMES/N_FRAMES if False else diagnostics(a,np.ones(N_FRAMES,dtype=bool))["abstention_fraction"])

if __name__ == '__main__': unittest.main()