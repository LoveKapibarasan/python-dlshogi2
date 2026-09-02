"""Load the JSONL metrics written by training, self-play and the RL loop.

This module is deliberately dependency-free (standard library only) so it can be
imported from a test, a notebook or a throwaway script without pulling in
Streamlit or pandas.  :mod:`dashboard.app` builds its tables on top of it.

Layout on disk — one ``.jsonl`` per run, anywhere under a metrics directory::

    metrics/
      train-sl.jsonl                 # supervised training run
      rl/
        rl.jsonl                     # the RL loop itself
        train-001.jsonl              # training inside iteration 1
        selfplay-001-w0.jsonl        # self-play worker 0 of iteration 1
        ...

See :mod:`pydlshogi2.metrics` for the record schema.
"""
import glob
import json
import os

#: Record types understood by :func:`load_records`.
RECORD_TYPES = ('run', 'metric', 'event')


def find_metric_files(directory):
    """Return every ``.jsonl`` file under ``directory``, sorted by name.

    :param directory: root directory to scan (searched recursively).
    """
    pattern = os.path.join(directory, '**', '*.jsonl')
    return sorted(glob.glob(pattern, recursive=True))


def load_records(paths):
    """Parse the given JSONL files into a flat list of records.

    Malformed lines are skipped rather than raising, so a run that was killed
    mid-write still loads.  Each record gains a ``source_file`` field.

    :param paths: iterable of ``.jsonl`` paths.
    :returns: list of record dicts.
    """
    records = []
    for path in paths:
        try:
            with open(path, encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except ValueError:
                        continue
                    if not isinstance(record, dict):
                        continue
                    record['source_file'] = path
                    records.append(record)
        except OSError:
            continue
    return records


def split_records(records):
    """Split records into ``(runs, metrics, events)`` by their ``type`` field."""
    runs = [r for r in records if r.get('type') == 'run']
    metrics = [r for r in records if r.get('type') == 'metric']
    events = [r for r in records if r.get('type') == 'event']
    return runs, metrics, events


def summarize_runs(records):
    """Build one summary row per run.

    Combines the opening ``run`` record with whatever the run produced: the
    number of metric samples, the last step/epoch reached, the final test
    accuracy, the last checkpoint written and the terminating status.

    A run that has metric records but no ``run`` record (an interrupted or
    hand-edited file) still gets a row, so nothing is silently dropped.

    :param records: records from :func:`load_records`.
    :returns: list of summary dicts, newest first.
    """
    runs, metrics, events = split_records(records)

    summaries = {}
    for run in runs:
        run_id = run.get('run_id') or run.get('source_file')
        args = run.get('args') or {}
        summaries[run_id] = {
            'run_id': run_id,
            'kind': run.get('kind', 'unknown'),
            'started_at': run.get('started_at'),
            'timestamp': run.get('timestamp'),
            'git_commit': (run.get('git_commit') or '')[:8],
            'git_dirty': run.get('git_dirty'),
            'hostname': run.get('hostname'),
            'gpu_name': run.get('gpu_name'),
            'parent_run_id': run.get('parent_run_id'),
            'source_file': run.get('source_file'),
            'lr': args.get('lr'),
            'batchsize': args.get('batchsize'),
            'val_lambda': args.get('val_lambda'),
            'blocks': args.get('blocks'),
            'channels': args.get('channels'),
            'playouts': args.get('playouts'),
            'games': args.get('games'),
            'resume': args.get('resume') or None,
            'args': args,
            'samples': 0,
            'status': 'running',
        }

    def slot(record):
        """Return the summary row a record belongs to, creating it if needed."""
        run_id = record.get('run_id') or record.get('source_file')
        if run_id not in summaries:
            summaries[run_id] = {
                'run_id': run_id, 'kind': 'unknown', 'samples': 0,
                'status': 'running', 'source_file': record.get('source_file'),
                'timestamp': record.get('timestamp'), 'args': {},
            }
        return summaries[run_id]

    for metric in metrics:
        row = slot(metric)
        row['samples'] += 1
        for key in ('epoch', 'step'):
            if metric.get(key) is not None:
                row['last_' + key] = metric[key]
        for key in ('test_accuracy_policy', 'test_accuracy_value',
                    'test_loss_total', 'train_loss_total'):
            if metric.get(key) is not None:
                row[key] = metric[key]
        if metric.get('scope') == 'summary':
            for key in ('positions', 'mean_moves', 'games_per_hour',
                        'black_wins', 'white_wins', 'draws'):
                if metric.get(key) is not None:
                    row[key] = metric[key]

    for event in events:
        row = slot(event)
        if event.get('event') == 'checkpoint':
            row['last_checkpoint'] = event.get('path')
        elif event.get('event') == 'run_end':
            row['status'] = event.get('status', 'completed')
            row['ended_at'] = event.get('timestamp')
        elif event.get('event') == 'data_loaded':
            row['train_positions'] = event.get('train_positions')
            row['test_positions'] = event.get('test_positions')
            row['network'] = event.get('network')

    for row in summaries.values():
        if row.get('ended_at') and row.get('timestamp'):
            row['duration_sec'] = row['ended_at'] - row['timestamp']

    return sorted(summaries.values(),
                  key=lambda r: r.get('timestamp') or 0, reverse=True)


def curve_points(records, run_ids=None, scope=None):
    """Return the metric samples usable as learning-curve points.

    :param records: records from :func:`load_records`.
    :param run_ids: keep only these runs (``None`` keeps all).
    :param scope: keep only this ``scope`` (``'interval'`` or ``'epoch'``).
    """
    _, metrics, _ = split_records(records)
    points = []
    for metric in metrics:
        if run_ids is not None and metric.get('run_id') not in run_ids:
            continue
        if scope is not None and metric.get('scope') != scope:
            continue
        if metric.get('step') is None:
            continue
        points.append(metric)
    return sorted(points, key=lambda m: (m.get('run_id') or '', m.get('step') or 0))


def selfplay_by_iteration(records):
    """Aggregate self-play worker summaries into per-iteration totals.

    Each parallel worker writes its own file; the RL loop cares about the
    iteration as a whole, so games, positions and results are summed and the
    mean game length is recomputed from the totals.

    :returns: list of dicts sorted by iteration.
    """
    _, metrics, _ = split_records(records)
    buckets = {}
    for metric in metrics:
        if metric.get('scope') != 'summary':
            continue
        iteration = metric.get('iteration')
        bucket = buckets.setdefault(iteration, {
            'iteration': iteration, 'workers': 0, 'games': 0, 'positions': 0,
            'black_wins': 0, 'white_wins': 0, 'draws': 0, 'seconds': 0.0,
        })
        bucket['workers'] += 1
        for key in ('games', 'positions', 'black_wins', 'white_wins', 'draws'):
            bucket[key] += metric.get(key) or 0
        bucket['seconds'] = max(bucket['seconds'], metric.get('seconds') or 0.0)

    rows = []
    for bucket in buckets.values():
        games = bucket['games']
        bucket['mean_moves'] = (bucket['positions'] / games) if games else 0.0
        bucket['black_win_rate'] = (bucket['black_wins'] / games) if games else 0.0
        bucket['white_win_rate'] = (bucket['white_wins'] / games) if games else 0.0
        bucket['draw_rate'] = (bucket['draws'] / games) if games else 0.0
        rows.append(bucket)
    return sorted(rows, key=lambda r: (r['iteration'] is None, r['iteration']))


def rl_iterations(records):
    """Return the RL loop's per-iteration records (``scope == 'iteration'``)."""
    _, metrics, _ = split_records(records)
    rows = [m for m in metrics if m.get('scope') == 'iteration']
    return sorted(rows, key=lambda r: r.get('iteration') or 0)


def list_checkpoints(directory):
    """List ``.pth`` / ``.onnx`` files under ``directory`` with size and mtime.

    The architecture stored inside a checkpoint is *not* read here — that needs
    torch, which the dashboard loads lazily only when asked.
    """
    rows = []
    for pattern in ('**/*.pth', '**/*.onnx'):
        for path in glob.glob(os.path.join(directory, pattern), recursive=True):
            try:
                stat = os.stat(path)
            except OSError:
                continue
            rows.append({
                'path': path,
                'size_mb': stat.st_size / (1024 * 1024),
                'modified': stat.st_mtime,
            })
    return sorted(rows, key=lambda r: r['modified'], reverse=True)


def read_checkpoint_network(path):
    """Return the ``network`` config embedded in a ``.pth`` checkpoint.

    Imports torch lazily and loads onto CPU with ``weights_only=False`` disabled
    where possible, so opening a large GPU checkpoint on a laptop still works.

    :returns: the config dict, or ``None`` when it cannot be read (legacy
        checkpoints saved before the field existed, or torch unavailable).
    """
    try:
        import torch
        checkpoint = torch.load(path, map_location='cpu')
    except Exception:
        return None
    if not isinstance(checkpoint, dict):
        return None
    return checkpoint.get('network')


def load(directory):
    """Convenience wrapper: read every ``.jsonl`` under ``directory``."""
    return load_records(find_metric_files(directory))
