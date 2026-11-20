"""Reporting and provenance checks using synthetic files, without training."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import importlib
import json
import unittest
from unittest.mock import patch

import numpy as np

from experiments.world_model_baselines.data import digest, write_json
from experiments.world_model_baselines.model import BaselineModel, save_checkpoint

runner = importlib.import_module('experiments.world_model_baselines.__main__')
reporting = importlib.import_module('experiments.world_model_baselines.report')


def metric(empty=False):
    return dict(one_step_mse=None if empty else .1,
                one_step_r2=None if empty else .9,
                rollout_10_mse=None if empty else .2,
                moving_rows=0 if empty else 1,
                rollout_starts=0 if empty else 1, selected_rows=1)


class ReportingRegressionTest(unittest.TestCase):
    def test_failed_seed_is_not_removed_from_average(self):
        result = reporting.moments([1., 2., None, 3., 4.])
        self.assertEqual((result['n'], result['expected_n']), (4, 5))
        self.assertIsNone(result['mean'])
        self.assertIsNone(result['sd'])
        self.assertEqual(reporting.fmt(result), 'N/A')
        complete = reporting.moments([1., 2., 3., 4., 5.])
        self.assertEqual(complete['mean'], 3.)
        self.assertAlmostEqual(complete['sd'], np.std([1., 2., 3., 4., 5.], ddof=1))

    def test_empty_early_window_stays_missing_in_table_and_plot(self):
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib.axes import Axes

        profile = [dict(step=step, **metric(empty=step < 5)) for step in range(6)]
        windows = dict(early=metric(empty=True), late=metric())
        evaluation = dict(overall=metric(), per_task={name: metric() for name in reporting.TASKS},
            step_profile=dict(overall=profile, per_task={name: profile for name in reporting.TASKS}),
            early_late=dict(overall=windows, per_task={name: windows for name in reporting.TASKS}))
        plotted = []
        original = Axes.plot

        def record_plot(axis, *args, **kwargs):
            if kwargs.get('label') == 'B1_K1':
                plotted.append((np.array(args[0]), np.array(args[1])))
            return original(axis, *args, **kwargs)

        with TemporaryDirectory() as tmp:
            root, results = Path(tmp), Path(tmp) / 'results/baselines'
            write_json(results / 'protocol.json', {'synthetic_test': True})
            write_json(root / 'runs/baselines/B1_K1/seed_0/evaluation.json',
                       dict(model='B1_K1', seed=0, protocol_sha256=digest(results / 'protocol.json'),
                            evaluation=evaluation))
            with patch.object(reporting, 'ROOT', root), patch.object(reporting, 'RESULTS', results), \
                    patch.object(Axes, 'plot', record_plot):
                reporting.report()
            saved = json.loads((results / 'metrics.json').read_text())
            early = saved['b1_early_late']['B1_K1']['lift']['early']
            self.assertEqual(early['moving_rows'], [0])
            for key in reporting.METRICS:
                self.assertIsNone(early[key]['mean'])
            self.assertIn('| B1_K1, steps 1–5 | lift | N/A | N/A | N/A |',
                          (root / 'BASELINE.md').read_text())
            self.assertEqual(len(plotted), 4)
            for x, y in plotted:
                np.testing.assert_array_equal(x, np.arange(1, 7))
                self.assertTrue(np.isnan(y[:5]).all())
                self.assertEqual(y[5], .1)


class ProvenanceRegressionTest(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.results = self.root / 'results/baselines'
        self.folder = self.root / 'runs/baselines/B0/seed_0'
        for source in runner.CORE_SOURCES:
            path = self.root / source
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('# Synthetic source fingerprint fixture\n')
        sources = {source: digest(self.root / source) for source in runner.CORE_SOURCES}
        write_json(self.results / 'implementation.json', {'sha256': sources})
        write_json(self.results / 'protocol.json', {'synthetic': 'protocol'})
        write_json(self.results / 'split.json', {'synthetic': 'split'})
        self.training = dict(kind='B0', history_k=0, seed=0, state_dim=51, action_dim=8)
        self.model = BaselineModel()
        save_checkpoint(self.folder / 'checkpoint.pt', self.model, self.training)
        write_json(self.folder / 'training.json', self.training)
        self.saved = dict(model='B0', kind='B0', history_k=0, seed=0,
                          protocol_sha256=digest(self.results / 'protocol.json'),
                          split_sha256=digest(self.results / 'split.json'),
                          source_sha256=sources,
                          checkpoint_sha256=digest(self.folder / 'checkpoint.pt'),
                          training=self.training,
                          evaluation={'per_task': {name: metric() for name in reporting.TASKS}})
        for name, value in [('ROOT', self.root), ('RESULTS', self.results)]:
            guard = patch.object(runner, name, value)
            guard.start()
            self.addCleanup(guard.stop)

    def verify(self, saved=None):
        runner.verify_completed_run(self.saved if saved is None else saved, self.folder, 'B0', 0, 0)

    def test_rejects_changed_core_source(self):
        self.verify()
        (self.root / runner.CORE_SOURCES[0]).write_text('# Changed core source\n')
        with self.assertRaisesRegex(ValueError, 'implementation changed'):
            self.verify()

    def test_rejects_checkpoint_digest_and_internal_seed_mismatch(self):
        self.verify()
        altered = deepcopy(self.saved)
        altered['checkpoint_sha256'] = 'incorrect'
        with self.assertRaisesRegex(ValueError, 'checksum differs'):
            self.verify(altered)
        # Matching the new file digest is insufficient if its embedded run is wrong.
        wrong_training = dict(self.training, seed=1)
        save_checkpoint(self.folder / 'checkpoint.pt', self.model, wrong_training)
        write_json(self.folder / 'training.json', wrong_training)
        altered['checkpoint_sha256'] = digest(self.folder / 'checkpoint.pt')
        with self.assertRaisesRegex(ValueError, 'training identity'):
            self.verify(altered)


if __name__ == '__main__':
    unittest.main()
