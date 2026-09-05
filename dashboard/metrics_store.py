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
import re
import sys

# リポジトリルートを通して pydlshogi2.rating を使う (標準ライブラリのみのモジュール)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pydlshogi2 import rating as rating_math  # noqa: E402

#: Record types understood by :func:`load_records`.
RECORD_TYPES = ('run', 'metric', 'event')

#: Default location of the improvement backlog, relative to the repository root.
BACKLOG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'wiki', 'Improvement-Backlog.md')


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


def match_summaries(records):
    """Build one row per engine-versus-engine match.

    A match writes a ``run`` record (who played whom, which experiment it
    belongs to) and, when it finishes, a ``scope='summary'`` metric with the
    final counts.  A match still in progress — or one killed part-way — has no
    summary, so the newest per-game record stands in and the row is marked
    ``running``.  Nothing is dropped: a match interrupted after 60 of 100 games
    is still evidence.

    :param records: records from :func:`load_records`.
    :returns: list of match dicts, newest first.
    """
    runs, metrics, events = split_records(records)

    matches = {}
    for run in runs:
        if run.get('kind') != 'match':
            continue
        run_id = run.get('run_id') or run.get('source_file')
        args = run.get('args') or {}
        matches[run_id] = {
            'run_id': run_id,
            'started_at': run.get('started_at'),
            'timestamp': run.get('timestamp'),
            'git_commit': (run.get('git_commit') or '')[:8],
            'git_dirty': run.get('git_dirty'),
            'hostname': run.get('hostname'),
            'experiment': run.get('experiment'),
            'issue': run.get('issue'),
            'note': run.get('note'),
            'player_a': run.get('player_a'),
            'player_b': run.get('player_b'),
            'byoyomi': args.get('byoyomi'),
            'playouts': args.get('playouts'),
            'opening': args.get('opening'),
            'source_file': run.get('source_file'),
            'status': 'running',
            'wins': 0, 'losses': 0, 'draws': 0, 'games': 0,
        }

    def slot(record):
        run_id = record.get('run_id') or record.get('source_file')
        return matches.get(run_id)

    latest_game = {}
    for metric in metrics:
        row = slot(metric)
        if row is None:
            continue
        if metric.get('scope') == 'summary':
            row.update({key: metric[key] for key in metric
                        if key not in ('type', 'run_id', 'timestamp', 'scope',
                                       'source_file')})
        elif metric.get('scope') == 'game':
            latest_game[metric.get('run_id')] = metric

    for run_id, metric in latest_game.items():
        row = matches.get(run_id)
        if row is None or row.get('games'):
            continue
        # summary がない (実行中/中断) 場合は最新の1局記録で代用する
        for key in ('wins', 'losses', 'draws', 'score', 'elo', 'error_margin',
                    'los', 'llr', 'sprt_decision', 'pairs', 'pentanomial'):
            if metric.get(key) is not None:
                row[key] = metric[key]
        row['games'] = (row.get('wins') or 0) + (row.get('losses') or 0) + (row.get('draws') or 0)

    for event in events:
        row = slot(event)
        if row is not None and event.get('event') == 'run_end':
            row['status'] = event.get('status', 'completed')
            row['ended_at'] = event.get('timestamp')

    return sorted(matches.values(),
                  key=lambda r: r.get('timestamp') or 0, reverse=True)


def rating_table(matches, anchor=None):
    """Fit one Elo rating per engine across every recorded match.

    Individual matches only give differences between two engines.  Fitting them
    all at once puts every checkpoint and every branch on a single scale, which
    is what makes "is this better than what we had three iterations ago?"
    answerable without replaying those games.

    :param matches: rows from :func:`match_summaries`.
    :param anchor: engine pinned to 0 Elo (default: the most-played one).
    :returns: list of rating rows, strongest first.
    """
    usable = [m for m in matches
              if m.get('player_a') and m.get('player_b') and m.get('games')]
    return rating_math.bradley_terry_ratings(usable, anchor=anchor)


def _split_table_row(line):
    """Split one Markdown table row into stripped cell strings."""
    return [cell.strip() for cell in line.strip().strip('|').split('|')]


def load_backlog(path=None):
    """Parse the improvement backlog table out of its wiki page.

    The backlog lives in ``wiki/Improvement-Backlog.md`` so that a proposal is
    reviewed in a pull request like everything else, and so the page stays
    readable on GitHub without this dashboard.  Reading it here — rather than
    keeping a second copy in a database — is what keeps the two from drifting.

    The first Markdown table with an ``ID`` column is used; its header cells
    become the keys of each row.

    :param path: path to the backlog page (default: :data:`BACKLOG_PATH`).
    :returns: list of row dicts, or ``[]`` when the page or table is missing.
    """
    path = path or BACKLOG_PATH
    try:
        with open(path, encoding='utf-8') as f:
            lines = f.read().splitlines()
    except OSError:
        return []

    header = None
    rows = []
    for line in lines:
        stripped = line.strip()
        if not stripped.startswith('|'):
            if header is not None:
                break  # 表が終わった
            continue
        cells = _split_table_row(stripped)
        if header is None:
            if not any(cell.lower() == 'id' for cell in cells):
                continue
            header = cells
            continue
        if all(re.fullmatch(r':?-{2,}:?', cell or '') for cell in cells):
            continue  # 区切り行
        row = dict(zip(header, cells))
        if row.get('ID'):
            rows.append(row)
    return rows


def backlog_with_results(records, path=None):
    """Join the backlog proposals with the matches that measured them.

    :param records: records from :func:`load_records`.
    :param path: path to the backlog page.
    :returns: list of backlog rows, each with a ``matches`` list attached and
        the measured Elo of its most recent match.
    """
    matches = match_summaries(records)
    by_experiment = {}
    for match in matches:
        experiment = match.get('experiment')
        if experiment:
            by_experiment.setdefault(experiment, []).append(match)

    rows = []
    for row in load_backlog(path):
        row = dict(row)
        experiment = row.get('ID')
        measured = by_experiment.get(experiment, [])
        row['matches'] = measured
        row['match_count'] = len(measured)
        if measured:
            newest = measured[0]
            row['measured_elo'] = newest.get('elo')
            row['error_margin'] = newest.get('error_margin')
            row['los'] = newest.get('los')
            row['games'] = newest.get('games')
            row['sprt_decision'] = newest.get('sprt_decision')
        rows.append(row)

    # backlog に載っていない実験も取りこぼさない
    known = {row.get('ID') for row in rows}
    for experiment, measured in sorted(by_experiment.items()):
        if experiment in known:
            continue
        newest = measured[0]
        rows.append({
            'ID': experiment,
            '改善案': '(backlog に記載なし)',
            'matches': measured,
            'match_count': len(measured),
            'measured_elo': newest.get('elo'),
            'error_margin': newest.get('error_margin'),
            'los': newest.get('los'),
            'games': newest.get('games'),
            'sprt_decision': newest.get('sprt_decision'),
        })
    return rows


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
