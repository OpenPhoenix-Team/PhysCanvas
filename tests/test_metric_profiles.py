import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evaluation import cli
from evaluation.aggregate import summarize


class MetricProfileTests(unittest.TestCase):
    def args(self, profile):
        return cli.build_parser().parse_args([
            '--mode', 'metrics-128', '--metric-profile', profile, '--complete-suite',
            '--checkpoint', '/checkpoint', '--config-json', '/config', '--output-dir', '/out'])

    def test_reconstruction_needs_no_distribution_or_flow_assets(self):
        args = self.args('reconstruction')
        cli.validate_metric_profile(args)
        args.i3d_weights = Path('/unused')
        with self.assertRaisesRegex(ValueError, 'does not use'):
            cli.validate_metric_profile(args)

    def test_visual_requires_vbench_and_excludes_flow(self):
        args = self.args('visual')
        args.i3d_weights = Path('/i3d')
        with self.assertRaisesRegex(ValueError, 'vbench-repo'):
            cli.validate_metric_profile(args)
        args.vbench_repo = Path('/vbench')
        cli.validate_metric_profile(args)
        args.threshold_json = Path('/tau')
        with self.assertRaisesRegex(ValueError, 'belong to the all'):
            cli.validate_metric_profile(args)

    def test_full_suite_requires_contact_and_tracking(self):
        args = self.args('all')
        for field in ('i3d_weights', 'vbench_repo', 'threshold_json'):
            setattr(args, field, Path('/asset'))
        with self.assertRaisesRegex(ValueError, 'contact-threshold-json.*queries-json'):
            cli.validate_metric_profile(args)
        for field in ('contact_threshold_json', 'cotracker_repo', 'cotracker_checkpoint', 'queries_json'):
            setattr(args, field, Path('/asset'))
        cli.validate_metric_profile(args)

    def test_public_entrypoints_fix_the_selected_suite(self):
        for profile in ('reconstruction', 'visual', 'all'):
            with self.subTest(profile=profile), patch.object(cli, 'run', return_value=0) as run:
                with patch('sys.argv', ['entry', '--metric-profile', 'all']):
                    getattr(cli, 'evaluate_' + profile)()
                self.assertEqual(run.call_args.args[0][-3:], ['--metric-profile', profile, '--complete-suite'])

    def test_runner_routes_each_profile_to_separate_output_and_assets(self):
        for profile in ('reconstruction', 'visual', 'all'):
            with self.subTest(profile=profile), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root / 'PhysCanvas/models').mkdir(parents=True)
                (root / 'checkpoint').mkdir()
                for name in ('prompt', 'i3d', 'evaluate.py', 'hubconf.py', 'tracker'):
                    (root / name).touch()
                (root / 'config.json').write_text(json.dumps({'training_config': {}}))
                (root / 'threshold.json').write_text('{}')
                (root / 'records.json').write_text(json.dumps({'records': [
                    {'episode_id': 'demo', 'object_state_task': 'bottle', 'sample_num_frames': 128}]}))
                (root / 'cohort.json').write_text(json.dumps({'episodes': [
                    {'episode_id': 'demo', 'task': 'bottle', 'source_frames': 128}]}))
                argv = ['--mode', 'metrics-128', '--metric-profile', profile, '--complete-suite',
                        '--checkpoint', str(root / 'checkpoint'), '--config-json', str(root / 'config.json'),
                        '--output-dir', str(root / 'out'), '--runtime-source', str(root),
                        '--prompt-embedding', str(root / 'prompt'), '--allow-external-records',
                        '--records-json', str(root / 'records.json'), '--cohort', str(root / 'cohort.json')]
                if profile != 'reconstruction':
                    argv += ['--i3d-weights', str(root / 'i3d'), '--vbench-repo', str(root)]
                if profile == 'all':
                    argv += ['--threshold-json', str(root / 'threshold.json'),
                             '--contact-threshold-json', str(root / 'threshold.json'),
                             '--cotracker-repo', str(root), '--cotracker-checkpoint', str(root / 'tracker'),
                             '--queries-json', str(root / 'threshold.json')]
                with patch('evaluation.cli.subprocess.run') as process, patch.object(cli, 'run_extra_metrics'):
                    self.assertEqual(cli.run(argv), 0)
                command, aggregate = [call.args[0] for call in process.call_args_list]
                self.assertEqual(command[command.index('--metric-profile') + 1], profile)
                self.assertEqual('--i3d-weights' in command, profile != 'reconstruction')
                self.assertEqual('--threshold-json' in aggregate, profile == 'all')
                directory = 'metrics-128' if profile == 'all' else 'metrics-' + profile
                self.assertTrue((root / 'out' / directory / 'n128/run_spec.json').is_file())

    def test_reconstruction_aggregation_never_reads_distribution_or_flow(self):
        image = {'frames': 127, 'mse_01': 0.1, 'psnr': 10, 'ssim': 0.7, 'lpips': 0.2}
        rows = [{'frames': 128, 'task': 'bottle', 'metrics': {
            'images': {'rgb': image, 'tactile_diff': image}}}]
        with patch('evaluation.aggregate.distribution_summary', side_effect=AssertionError), \
             patch('evaluation.aggregate.flow_summary', side_effect=AssertionError):
            result = summarize(rows, Path('/unused'), True, 'reconstruction')
        self.assertNotIn('flow', result)
        self.assertNotIn('fid', result['images']['rgb'])
        self.assertEqual(result['images']['rgb']['ssim'], 0.7)

    def test_visual_aggregation_includes_distributions_only(self):
        image = {'frames': 127, 'mse_01': 0.1, 'psnr': 10, 'ssim': 0.7, 'lpips': 0.2}
        rows = [{'frames': 128, 'task': 'bottle', 'metrics': {
            'images': {'rgb': image, 'tactile_diff': image}}}]
        with patch('evaluation.aggregate.distribution_summary', return_value={'fid': 1, 'fvd': 2}) as dist:
            result = summarize(rows, Path('/unused'), True, 'visual')
        self.assertEqual(dist.call_count, 2)
        self.assertNotIn('marker_displacement', result)
        self.assertEqual(result['images']['rgb']['fvd'], 2)


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'requires torch; run on the PPU CPU environment')
class MetricComputationTests(unittest.TestCase):
    def test_bfloat16_tactile_event_input_converts_to_float32(self):
        value = torch.ones(2, 3, 672, 672, dtype=torch.bfloat16)
        converted = value[:, :, 224:448, :].float().numpy()
        self.assertEqual(str(converted.dtype), 'float32')
        self.assertEqual(converted.shape, (2, 3, 224, 672))

    def test_reconstruction_does_not_load_i3d_or_write_features(self):
        from evaluation.metrics import EpisodeMetrics
        with tempfile.TemporaryDirectory() as tmp, \
             patch('evaluation.image_metrics.ImageMetricAccumulator._ensure_lpips', return_value=object()), \
             patch('evaluation.image_metrics.ImageMetricAccumulator._lpips_values', side_effect=lambda a, b: [0.0] * len(a)), \
             patch('torch.jit.load', side_effect=AssertionError('I3D must not be loaded')):
            metric = EpisodeMetrics('cpu', None, tmp, None, profile='reconstruction')
            gt = torch.zeros(3, 3, 672, 8)
            pred = gt + 0.1
            result = metric.evaluate({'episode_id': 'example'}, argparse.Namespace(split='evaluation'),
                                     gt, pred, None, None, None, None, None, None, None)
            self.assertEqual(set(result['images']), {'rgb', 'tactile_diff'})
            self.assertEqual(result['images']['rgb']['frames'], 2)
            self.assertGreater(result['images']['rgb']['mse_01'], 0)
            self.assertNotIn('flow', result)
            self.assertNotIn('temporary_features', result)
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_visual_computes_features_without_accessing_flow(self):
        from evaluation.metrics import EpisodeMetrics
        from evaluation.image_metrics import ImageMetricAccumulator
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(ImageMetricAccumulator, '_lpips_values', side_effect=lambda a, b: [0.0] * len(a)):
            metric = EpisodeMetrics.__new__(EpisodeMetrics)
            metric.profile, metric.device, metric.root = 'visual', 'cpu', Path(tmp)
            metric.images = ImageMetricAccumulator(['rgb', 'tactile_diff'], 'cpu')
            metric.inception = lambda x: torch.ones(len(x), 4)
            metric.i3d = lambda x, **kw: torch.ones(1, 4)
            gt = torch.zeros(17, 3, 672, 8)
            result = metric.evaluate({'episode_id': 'example'}, argparse.Namespace(split='evaluation'),
                                     gt, gt + 0.1, None, None, None, None, None, None, None)
            self.assertNotIn('flow', result)
            self.assertNotIn('task_success', result)
            self.assertTrue(Path(result['temporary_features']).is_file())
            self.assertEqual(result['fid_feature_counts']['rgb_fvd_gt'], 1)


if __name__ == '__main__':
    unittest.main()
