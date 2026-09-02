"""Tests for the metrics writer and the dashboard's loader.

Deliberately standard-library only (``unittest``, no torch, no cshogi) so they
run anywhere::

    python -m unittest discover -s tests
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, 'dashboard'))

import metrics_store  # noqa: E402
from pydlshogi2.metrics import MetricsWriter, append_record, new_run_id  # noqa: E402


def write_train_run(directory, name='train.jsonl', run_id=None, steps=3, checkpoint=True):
    """Write a small but complete training run and return its writer."""
    writer = MetricsWriter(os.path.join(directory, name), kind='train',
                           args={'lr': 0.01, 'batchsize': 1024, 'blocks': 20},
                           run_id=run_id, extra={'gpu_name': 'test-gpu'})
    writer.event('data_loaded', train_positions=100, test_positions=10,
                 network={'blocks': 20, 'channels': 256, 'fcl': 256, 'se': True},
                 start_epoch=0, start_step=0)
    for step in range(1, steps + 1):
        writer.metric(scope='interval', epoch=1, step=step * 100,
                      train_loss_total=4.0 - step * 0.1,
                      test_loss_total=4.1 - step * 0.1,
                      test_accuracy_policy=0.1 + step * 0.01,
                      test_accuracy_value=0.5 + step * 0.01)
    if checkpoint:
        writer.event('checkpoint', path='checkpoints/checkpoint-001.pth',
                     epoch=1, step=steps * 100)
    return writer


class NewRunIdTest(unittest.TestCase):
    def test_ids_are_unique_and_sortable(self):
        ids = [new_run_id() for _ in range(5)]
        self.assertEqual(len(set(ids)), 5)
        self.assertTrue(all(len(i) == len(ids[0]) for i in ids))


class MetricsWriterTest(unittest.TestCase):
    def test_disabled_writer_is_a_noop(self):
        writer = MetricsWriter(None, kind='train', args={'lr': 0.01})
        writer.metric(step=1, train_loss_total=1.0)
        writer.event('checkpoint', path='x')
        writer.close()
        self.assertTrue(writer.run_id)

    def test_writes_run_metric_and_event_records(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'train.jsonl')
            writer = MetricsWriter(path, kind='train', args={'lr': 0.01})
            writer.metric(scope='epoch', epoch=1, step=10, train_loss_total=3.0)
            writer.close(status='completed')

            with open(path, encoding='utf-8') as f:
                records = [json.loads(line) for line in f]
            self.assertEqual([r['type'] for r in records],
                             ['run', 'metric', 'event'])
            self.assertEqual(records[0]['kind'], 'train')
            self.assertEqual(records[0]['args']['lr'], 0.01)
            self.assertEqual(records[-1]['event'], 'run_end')
            self.assertTrue(all(r['run_id'] == writer.run_id for r in records))

    def test_close_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'train.jsonl')
            writer = MetricsWriter(path, kind='train')
            writer.close()
            writer.close()
            with open(path, encoding='utf-8') as f:
                ends = [line for line in f if '"run_end"' in line]
            self.assertEqual(len(ends), 1)

    def test_non_serialisable_values_are_stringified(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'train.jsonl')
            writer = MetricsWriter(path, kind='train', args={'device': object()})
            writer.close()
            records = metrics_store.load_records([path])
            self.assertIsInstance(records[0]['args']['device'], str)

    def test_creates_parent_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'nested', 'deeper', 'train.jsonl')
            MetricsWriter(path, kind='train').close()
            self.assertTrue(os.path.exists(path))

    def test_resumed_run_can_reuse_a_run_id(self):
        with tempfile.TemporaryDirectory() as directory:
            first = write_train_run(directory, run_id='fixed-id')
            first.close()
            second = write_train_run(directory, name='resume.jsonl', run_id='fixed-id')
            second.close()
            records = metrics_store.load(directory)
            summaries = metrics_store.summarize_runs(records)
            self.assertEqual([s['run_id'] for s in summaries], ['fixed-id'])


class MetricsCliTest(unittest.TestCase):
    def run_cli(self, *args):
        subprocess.check_call([sys.executable, '-m', 'pydlshogi2.metrics'] + list(args),
                              cwd=REPO_ROOT, stdout=subprocess.DEVNULL)

    def test_appends_records_from_the_shell(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'rl.jsonl')
            self.run_cli(path, '--type', 'run', '--kind', 'rl',
                         '--run-id', 'rl-1', '--set', 'iterations=20')
            self.run_cli(path, '--type', 'metric', '--run-id', 'rl-1',
                         '--set', 'scope=iteration', '--set', 'iteration=1',
                         '--set', 'checkpoint=rl/checkpoint-001.pth')
            self.run_cli(path, '--type', 'event', '--event', 'run_end',
                         '--run-id', 'rl-1', '--set', 'status=completed')

            records = metrics_store.load_records([path])
            self.assertEqual([r['type'] for r in records], ['run', 'metric', 'event'])
            # 数値はJSONとして、パスは文字列としてデコードされる
            self.assertEqual(records[0]['iterations'], 20)
            self.assertEqual(records[1]['iteration'], 1)
            self.assertEqual(records[1]['checkpoint'], 'rl/checkpoint-001.pth')
            self.assertEqual(records[2]['status'], 'completed')

            iterations = metrics_store.rl_iterations(records)
            self.assertEqual(len(iterations), 1)


class MetricsStoreTest(unittest.TestCase):
    def test_finds_files_recursively(self):
        with tempfile.TemporaryDirectory() as directory:
            write_train_run(directory).close()
            os.makedirs(os.path.join(directory, 'rl'))
            write_train_run(os.path.join(directory, 'rl'), name='train-001.jsonl').close()
            self.assertEqual(len(metrics_store.find_metric_files(directory)), 2)

    def test_skips_malformed_lines(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'train.jsonl')
            writer = MetricsWriter(path, kind='train')
            writer.metric(scope='epoch', step=1, train_loss_total=1.0)
            writer._file.write('{"type": "metric", "step": 2\n')  # 途中で切れた行
            writer._file.write('\n')
            writer._file.write('"not an object"\n')
            writer.close()
            records = metrics_store.load_records([path])
            self.assertEqual([r['type'] for r in records], ['run', 'metric', 'event'])

    def test_missing_directory_yields_nothing(self):
        self.assertEqual(metrics_store.load('/nonexistent-metrics-dir'), [])

    def test_summarize_runs_reports_progress_and_status(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = write_train_run(directory, steps=3)
            writer.close(status='interrupted')
            summaries = metrics_store.summarize_runs(metrics_store.load(directory))
            self.assertEqual(len(summaries), 1)
            summary = summaries[0]
            self.assertEqual(summary['kind'], 'train')
            self.assertEqual(summary['status'], 'interrupted')
            self.assertEqual(summary['samples'], 3)
            self.assertEqual(summary['last_step'], 300)
            self.assertEqual(summary['lr'], 0.01)
            self.assertEqual(summary['train_positions'], 100)
            self.assertEqual(summary['last_checkpoint'],
                             'checkpoints/checkpoint-001.pth')
            self.assertEqual(summary['network']['blocks'], 20)

    def test_run_without_a_run_record_is_still_listed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'orphan.jsonl')
            append_record(path, {'type': 'metric', 'run_id': 'orphan',
                                 'step': 1, 'train_loss_total': 1.0})
            summaries = metrics_store.summarize_runs(metrics_store.load(directory))
            self.assertEqual([s['run_id'] for s in summaries], ['orphan'])
            self.assertEqual(summaries[0]['status'], 'running')

    def test_curve_points_filter_by_run_and_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            a = write_train_run(directory, name='a.jsonl', run_id='a', steps=2)
            a.metric(scope='epoch', epoch=1, step=200, test_loss_total=3.0)
            a.close()
            b = write_train_run(directory, name='b.jsonl', run_id='b', steps=4)
            b.close()
            records = metrics_store.load(directory)

            self.assertEqual(len(metrics_store.curve_points(records, scope='interval')), 6)
            self.assertEqual(
                len(metrics_store.curve_points(records, run_ids={'a'}, scope='interval')), 2)
            self.assertEqual(len(metrics_store.curve_points(records, scope='epoch')), 1)

    def test_curve_points_are_sorted_by_run_then_step(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = MetricsWriter(os.path.join(directory, 'a.jsonl'), kind='train', run_id='a')
            for step in (300, 100, 200):
                writer.metric(scope='interval', step=step, train_loss_total=1.0)
            writer.close()
            points = metrics_store.curve_points(metrics_store.load(directory), scope='interval')
            self.assertEqual([p['step'] for p in points], [100, 200, 300])

    def test_selfplay_workers_are_aggregated_per_iteration(self):
        with tempfile.TemporaryDirectory() as directory:
            for iteration in (1, 2):
                for worker in range(3):
                    writer = MetricsWriter(
                        os.path.join(directory, 'selfplay-%d-w%d.jsonl' % (iteration, worker)),
                        kind='selfplay', args={'games': 4, 'playouts': 400},
                        extra={'iteration': iteration, 'worker': worker})
                    writer.metric(scope='summary', iteration=iteration, worker=worker,
                                  games=4, positions=400, black_wins=2, white_wins=1,
                                  draws=1, mean_moves=100.0, seconds=60.0 + worker)
                    writer.close()

            rows = metrics_store.selfplay_by_iteration(metrics_store.load(directory))
            self.assertEqual([r['iteration'] for r in rows], [1, 2])
            first = rows[0]
            self.assertEqual(first['workers'], 3)
            self.assertEqual(first['games'], 12)
            self.assertEqual(first['positions'], 1200)
            self.assertAlmostEqual(first['mean_moves'], 100.0)
            self.assertAlmostEqual(first['black_win_rate'], 0.5)
            self.assertAlmostEqual(first['draw_rate'], 0.25)
            # ワーカーは並列に走るので、経過時間は合計ではなく最大値を取る
            self.assertAlmostEqual(first['seconds'], 62.0)

    def test_selfplay_aggregation_survives_zero_games(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = MetricsWriter(os.path.join(directory, 'sp.jsonl'), kind='selfplay')
            writer.metric(scope='summary', iteration=1, games=0, positions=0,
                          black_wins=0, white_wins=0, draws=0, seconds=1.0)
            writer.close()
            rows = metrics_store.selfplay_by_iteration(metrics_store.load(directory))
            self.assertEqual(rows[0]['mean_moves'], 0.0)
            self.assertEqual(rows[0]['black_win_rate'], 0.0)

    def test_list_checkpoints_orders_by_mtime(self):
        with tempfile.TemporaryDirectory() as directory:
            old = os.path.join(directory, 'old.pth')
            new = os.path.join(directory, 'nested', 'new.onnx')
            os.makedirs(os.path.dirname(new))
            for path in (old, new):
                with open(path, 'wb') as f:
                    f.write(b'0' * 1024)
            os.utime(old, (1_600_000_000, 1_600_000_000))
            rows = metrics_store.list_checkpoints(directory)
            self.assertEqual([os.path.basename(r['path']) for r in rows],
                             ['new.onnx', 'old.pth'])
            self.assertGreater(rows[0]['size_mb'], 0)

    def test_read_checkpoint_network_returns_none_for_junk(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'broken.pth')
            with open(path, 'wb') as f:
                f.write(b'not a checkpoint')
            self.assertIsNone(metrics_store.read_checkpoint_network(path))


class DashboardLauncherTest(unittest.TestCase):
    """The launcher and its systemd unit have to agree with each other."""

    RUN_SH = os.path.join(REPO_ROOT, 'dashboard', 'run.sh')
    UNIT = os.path.join(REPO_ROOT, 'dashboard', 'dlshogi-dashboard.service')

    def test_launcher_is_executable(self):
        self.assertTrue(os.access(self.RUN_SH, os.X_OK))

    def test_launcher_rejects_an_unknown_subcommand(self):
        result = subprocess.run([self.RUN_SH, 'bogus'], cwd=REPO_ROOT,
                                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('usage:', result.stderr)

    def test_status_reports_not_running_without_a_pid_file(self):
        # PIDファイルの場所を差し替えられないので、既存のPIDファイルがある環境では
        # このテストは意味を持たない。その場合はスキップする。
        if os.path.exists(os.path.join(REPO_ROOT, 'logs', 'dashboard.pid')):
            self.skipTest('a dashboard is running on this machine')
        result = subprocess.run([self.RUN_SH, 'status'], cwd=REPO_ROOT,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn('not running', result.stdout)

    def test_systemd_unit_points_at_the_launcher(self):
        with open(self.UNIT, encoding='utf-8') as f:
            unit = f.read()
        self.assertIn('ExecStart=__REPO__/dashboard/run.sh run', unit)
        # sed で置換する前提のプレースホルダが両方の行に残っていること
        self.assertIn('WorkingDirectory=__REPO__', unit)

    def test_documented_defaults_match_the_script(self):
        with open(self.RUN_SH, encoding='utf-8') as f:
            script = f.read()
        self.assertIn('PORT="${PORT:-8501}"', script)
        self.assertIn('ADDRESS="${ADDRESS:-127.0.0.1}"', script)

        with open(os.path.join(REPO_ROOT, 'wiki', 'Metrics-and-Dashboard.md'),
                  encoding='utf-8') as f:
            page = f.read()
        self.assertIn('`8501`', page)
        self.assertIn('`127.0.0.1`', page)


if __name__ == '__main__':
    unittest.main()
