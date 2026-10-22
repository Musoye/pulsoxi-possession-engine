import unittest
import numpy as np
from pulsoxi.data_audit import audit_arrays
from pulsoxi.tracking_features import extract_tracking_features

class TestDataAudit(unittest.TestCase):
    def test_audit_flags_quality_issues_without_repairing(self):
        ids=np.array([1,2,4]); ball=np.array([[.5,.5],[np.nan,np.nan],[2.,.5]])
        players=np.array([[[.1,.1],[.9,.9]],[[.1,.1],[.9,.9]],[[.9,.1],[.9,.9]]])
        features,_=extract_tracking_features(ball,players,[0,1])
        report=audit_arrays(ids,ball,players,[0,1],features)
        self.assertFalse(report['frame_ids_contiguous'])
        self.assertEqual(report['duplicate_frame_ids'],0)
        self.assertGreater(report['coordinate_out_of_range'],0)
        self.assertEqual(report['ball_missing_gap_frames']['max'],1)
        self.assertIn('ball_visible',report['feature_missingness'])

if __name__ == '__main__': unittest.main()
