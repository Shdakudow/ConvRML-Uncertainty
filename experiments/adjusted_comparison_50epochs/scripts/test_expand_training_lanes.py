import unittest
from scripts.expand_training_lanes import plan


class PlanTests(unittest.TestCase):
    def test_twelve_serial_lanes_and_complete_tuning_barrier(self):
        lanes = [list(range(i*5, i*5+5)) for i in range(6)]
        jobs = []
        for epoch in (2, 4):
            for task in range(30):
                jobs.append(dict(job=f'{epoch}_{task}', stage='train', task=task, epoch=epoch))
        jobs.append(dict(job='tune', stage='tune'))
        live = {j['job']: dict(state='PENDING') for j in jobs}
        for task in range(0, 30, 5):
            live[f'2_{task}']['state'] = 'RUNNING'
        result, changes, tails = plan(dict(lanes=lanes,jobs=jobs), live, {i:0 for i in range(30)})
        self.assertEqual(len(result), 12)
        self.assertEqual(len(tails), 12)
        roots = [c for c in changes if c['stage']=='train' and c['dependency'] is None]
        self.assertEqual(len(roots), 6)  # Six existing running jobs plus six newly ready roots.
        self.assertEqual(set(changes[-1]['dependency'].split(':')), set(tails))
        for change in changes:
            if change['stage']=='train' and change['dependency']:
                self.assertLessEqual(int(change['dependency'].split('_')[0]), int(change['job'].split('_')[0]))

    def test_missing_unfinished_job_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Missing unfinished'):
            plan(dict(lanes=[list(range(i*5,i*5+5)) for i in range(6)],
                      jobs=[dict(job='missing',stage='train',task=0,epoch=2)]), {}, {i:0 for i in range(30)})


if __name__ == '__main__':
    unittest.main()
