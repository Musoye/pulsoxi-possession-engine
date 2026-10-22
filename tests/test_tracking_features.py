import unittest
import numpy as np
from pulsoxi.tracking_features import FEATURE_NAMES, TrackingFeatureConfig, extract_tracking_features, metadata_team_map, video_frame_from_atd

class TestTrackingFeatures(unittest.TestCase):
    def setUp(self):
        self.ball = np.array([[.5,.5],[.6,.5],[np.nan,np.nan],[np.nan,np.nan],[.2,.2],[.25,.2]])
        self.players = np.array([
            [[.5,.5],[.9,.9],[.2,.2],[.8,.8]],
            [[.6,.5],[.9,.9],[.2,.2],[.8,.8]],
            [[.6,.5],[.9,.9],[.2,.2],[.8,.8]],
            [[np.nan,np.nan],[.9,.9],[.2,.2],[.8,.8]],
            [[.2,.2],[.9,.9],[.2,.2],[.8,.8]],
            [[.25,.2],[.9,.9],[.2,.2],[.8,.8]],
        ])
        self.teams = [0,0,1,1]
    def test_nearest_distance_scaling_and_counts(self):
        x, names = extract_tracking_features(self.ball, self.players, self.teams)
        self.assertEqual(x.shape, (6, len(FEATURE_NAMES)))
        self.assertEqual(names, FEATURE_NAMES)
        self.assertEqual(x[0, 3], 2)
        self.assertEqual(x[0, 4], 2)
        self.assertAlmostEqual(x[0, 9], 0.0)
        self.assertAlmostEqual(x[0, 10], np.hypot(.3*105.3, .3*68.0))
    def test_missing_ball_and_gap(self):
        x, _ = extract_tracking_features(self.ball, self.players, self.teams)
        self.assertEqual(tuple(x[2:4, 0]), (0.0, 0.0))
        self.assertEqual(tuple(x[2:4, 15]), (1.0, 2.0))
        self.assertTrue(np.isnan(x[2, 9]))
    def test_half_boundary_resets_derivatives(self):
        frames = [11185, 11186, 90651, 90652, 110808, 110809]
        x, _ = extract_tracking_features(self.ball, self.players, self.teams, video_frames=frames)
        self.assertTrue(np.isnan(x[3, 12]))
        self.assertTrue(np.isfinite(x[5, 12]))
    def test_velocity_gate_distinguishes_realistic_motion_and_extreme_jump(self):
        ball = np.array([[.5,.5],[.51,.5],[.8,.5]])
        players = np.array([[[.5,.5]], [[.502,.5]], [[.9,.5]]])
        x, _ = extract_tracking_features(ball, players, [0])
        self.assertEqual(x[1, 16], 1.0)
        self.assertTrue(np.isnan(x[2, 14]))
        self.assertEqual(x[2, 16], 0.0)
    def test_alignment_and_metadata(self):
        np.testing.assert_array_equal(video_frame_from_atd([1, 2, 199319]), [0, 1, 199318])
        np.testing.assert_array_equal(metadata_team_map({10:'away', 2:'home'}, 'home'), [0, 1])
    def test_fragmented_slot_is_not_identity(self):
        teams = [0, 1]
        players = np.array([[[.5,.5],[.9,.9]], [[.9,.9],[.5,.5]]])
        ball = np.array([[.5,.5],[.5,.5]])
        x, _ = extract_tracking_features(ball, players, teams)
        self.assertEqual(x[0, 3], 1)
        self.assertEqual(x[1, 4], 1)

if __name__ == '__main__':
    unittest.main()





