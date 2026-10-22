"""Synthetic, licence-safe unit tests. NOT benchmark results."""
import unittest
import numpy as np
from pulsoxi.temporal import (STATES, N_FRAMES, HALVES, TransitionCosts,
                              viterbi, decode_full_match)
from pulsoxi.submission import write_submission
from pathlib import Path
import tempfile

class TestPossessionDecoder(unittest.TestCase):
    def test_isolated_false_team_flip_is_suppressed(self):
        p = np.tile([.93, .04, .03], (80, 1))
        p[40] = [.15, .8, .05]
        labels = viterbi(p)
        self.assertTrue(np.all(labels == 0))

    def test_sustained_change_is_allowed(self):
        p = np.vstack((np.tile([.96, .02, .02], (50, 1)),
                       np.tile([.02, .96, .02], (50, 1))))
        labels = viterbi(p)
        self.assertTrue(np.all(labels[:40] == 0))
        self.assertTrue(np.all(labels[60:] == 1))

    def test_missing_observations_keep_prior_context(self):
        p = np.vstack((np.tile([.97, .02, .01], (30, 1)),
                       np.tile([.01, .97, .02], (8, 1)),
                       np.tile([.97, .02, .01], (30, 1))))
        known = np.ones(len(p), dtype=bool)
        known[30:38] = False
        labels = viterbi(p, known=known)
        self.assertTrue(np.all(labels[30:38] == 0))

    def test_per_half_reset_and_ungraded_x(self):
        p = np.tile([.8, .1, .1], (N_FRAMES, 1))
        a,b = HALVES[1]
        p[a:b+1] = [.1, .8, .1]
        labels = decode_full_match(p)
        self.assertEqual(len(labels), N_FRAMES)
        self.assertEqual(labels[0], 'X')
        self.assertEqual(labels[HALVES[0][0]], 'H')
        self.assertEqual(labels[HALVES[0][1] + 1], 'X')
        self.assertEqual(labels[HALVES[1][0]], 'A')
        self.assertEqual(labels[-1], 'X')

    def test_rejects_bad_emission_array(self):
        with self.assertRaises(ValueError):
            viterbi(np.array([[0.2, -0.2, 1.0]]))

    def test_csv_output_contract(self):
        from pulsoxi.temporal import N_FRAMES
        labels = np.full(N_FRAMES, 'X')
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'submission.csv'
            write_submission(path, labels)
            with path.open() as f:
                self.assertEqual(f.readline().strip(), 'frame,label')
                self.assertEqual(f.readline().strip(), '0,X')
                self.assertEqual(sum(1 for _ in f), N_FRAMES - 1)

if __name__ == '__main__':
    unittest.main()
