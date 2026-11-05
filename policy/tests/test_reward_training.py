import unittest
from reward_training.train import balanced_batches, partitions

class RewardTrainingTests(unittest.TestCase):
    def test_global_balance_disjoint_ranks(self):
        rows = [{'label': i % 2} for i in range(64)]
        ranks = [balanced_batches(rows, range(64), 16, r, 4, 42) for r in range(4)]
        consumed = []
        for step in range(4):
            batch = sum([r[step] for r in ranks], [])
            self.assertEqual(len(set(batch)), 16)
            self.assertEqual(sum(rows[i]['label'] for i in batch), 8)
            consumed.extend(batch)
        self.assertEqual(len(set(consumed)), 64)

    def test_episode_holdout_and_duplicate_rejection(self):
        rows = [{'sample_id': s, 'aligned_frame_index': 0, 'label': 0} for s in ['a', 'b']]
        split = {'train_episodes': ['a'], 'val_test_episodes': ['b']}
        self.assertEqual(partitions(rows, split), ([0], [1]))
        with self.assertRaises(ValueError):
            partitions(rows + [rows[0]], split)
        with self.assertRaises(ValueError):
            partitions(rows, {'train_episodes': ['a', 'b'], 'val_test_episodes': ['b']})

if __name__ == '__main__':
    unittest.main()
