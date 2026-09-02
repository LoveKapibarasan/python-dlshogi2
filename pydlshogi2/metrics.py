"""Structured metrics logging for training and self-play runs.

The training loop and the self-play generator both emit human-readable text
logs, which are convenient to tail but painful to compare across runs.  This
module writes the same information a second time as **JSON Lines** so that the
dashboard (``dashboard/app.py``) — or any ad-hoc script — can load a whole
experiment history with a one-line parser.

Every run appends to its own ``.jsonl`` file.  The first record describes the
run, the rest are metric samples and events:

.. code-block:: json

    {"type": "run", "run_id": "20260902-101500-1a2b3c4d", "kind": "train", ...}
    {"type": "metric", "run_id": "...", "step": 100, "train_loss_total": 3.1, ...}
    {"type": "event", "run_id": "...", "event": "checkpoint", "path": "..."}
    {"type": "event", "run_id": "...", "event": "run_end", "status": "completed"}

Records are flushed on every write, so a run that is preempted mid-epoch still
leaves a readable history behind.

Example
-------

.. code-block:: python

    writer = MetricsWriter('metrics/train.jsonl', kind='train', args=vars(args))
    writer.metric(step=t, epoch=epoch, train_loss_total=loss)
    writer.close(status='completed')
"""
import json
import os
import platform
import socket
import subprocess
import time
import uuid

#: Version of the record schema, stored on the ``run`` record so the dashboard
#: can stay backwards compatible when fields are added later.
SCHEMA_VERSION = 1


def new_run_id():
    """Return a fresh run identifier, e.g. ``20260902-101500-1a2b3c4d``.

    The timestamp prefix makes the ids sort chronologically, and the random
    suffix keeps parallel runs started in the same second distinct.
    """
    return '{}-{}'.format(time.strftime('%Y%m%d-%H%M%S'), uuid.uuid4().hex[:8])


def git_revision(cwd=None):
    """Return ``(commit, dirty)`` for the repository containing this file.

    :param cwd: directory to run ``git`` in (defaults to the package directory).
    :returns: a ``(commit_hash_or_None, is_dirty)`` tuple.  Both are ``None`` /
        ``False`` when git is unavailable or the code is not in a checkout.
    """
    if cwd is None:
        cwd = os.path.dirname(os.path.abspath(__file__))
    try:
        commit = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=cwd,
            stderr=subprocess.DEVNULL).decode().strip()
        status = subprocess.check_output(
            ['git', 'status', '--porcelain'], cwd=cwd,
            stderr=subprocess.DEVNULL).decode().strip()
        return commit, bool(status)
    except (OSError, subprocess.CalledProcessError):
        return None, False


def gpu_name(gpu_id=0):
    """Return the CUDA device name for ``gpu_id``, or ``None`` on CPU.

    Importing torch lazily keeps this module usable from the dashboard, which
    has no reason to pull in a deep learning framework.
    """
    if gpu_id is None or gpu_id < 0:
        return None
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        return torch.cuda.get_device_name(gpu_id)
    except Exception:
        return None


def _jsonable(value):
    """Coerce ``value`` into something :func:`json.dumps` accepts."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return str(value)


class MetricsWriter:
    """Append run metadata, metric samples and events to a JSONL file.

    :param path: output ``.jsonl`` path.  Parent directories are created.  Pass
        ``None`` to get a no-op writer, so callers never need to branch on
        whether metrics are enabled.
    :param kind: what produced the records — ``'train'``, ``'selfplay'`` or
        ``'rl'``.
    :param args: the run's parsed arguments (``vars(args)``), stored verbatim.
    :param run_id: reuse an existing id (e.g. to continue a run after a
        preemption); a new one is generated when omitted.
    :param parent_run_id: id of the run this one continues or was spawned by.
    :param extra: additional fields to store on the ``run`` record.
    """

    def __init__(self, path, kind, args=None, run_id=None, parent_run_id=None, extra=None):
        self.path = path
        self.kind = kind
        self.run_id = run_id or new_run_id()
        self._file = None
        self._closed = False

        if path is None:
            return

        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        self._file = open(path, 'a', encoding='utf-8')

        commit, dirty = git_revision()
        record = {
            'type': 'run',
            'schema_version': SCHEMA_VERSION,
            'run_id': self.run_id,
            'parent_run_id': parent_run_id,
            'kind': kind,
            'timestamp': time.time(),
            'started_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'git_commit': commit,
            'git_dirty': dirty,
            'hostname': socket.gethostname(),
            'platform': platform.platform(),
            'python': platform.python_version(),
            'args': _jsonable(args or {}),
        }
        if extra:
            record.update(_jsonable(extra))
        self._write(record)

    def _write(self, record):
        """Serialise one record and flush it to disk."""
        if self._file is None:
            return
        self._file.write(json.dumps(record, ensure_ascii=False) + '\n')
        self._file.flush()

    def metric(self, **fields):
        """Append a ``metric`` record.

        All keyword arguments are stored as-is, so callers can log whatever they
        have (``step``, ``epoch``, losses, accuracies, self-play counters …).
        """
        record = {'type': 'metric', 'run_id': self.run_id,
                  'timestamp': time.time()}
        record.update(_jsonable(fields))
        self._write(record)

    def event(self, event, **fields):
        """Append an ``event`` record such as a checkpoint save or run end.

        :param event: short event name, e.g. ``'checkpoint'``.
        """
        record = {'type': 'event', 'run_id': self.run_id,
                  'timestamp': time.time(), 'event': event}
        record.update(_jsonable(fields))
        self._write(record)

    def close(self, status='completed', **fields):
        """Write a terminating ``run_end`` event and close the file.

        :param status: ``'completed'``, ``'interrupted'`` or ``'failed'``.
        """
        if self._closed:
            return
        self._closed = True
        self.event('run_end', status=status, **fields)
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close(status='completed' if exc_type is None else 'failed')
        return False


def append_record(path, record):
    """Append a single already-built record to a JSONL file.

    Used by shell drivers (``rl_loop.sh``) that want to log one event without
    opening a long-lived :class:`MetricsWriter`.

    :param path: target ``.jsonl`` file; parent directories are created.
    :param record: a JSON-serialisable mapping.
    """
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(_jsonable(record), ensure_ascii=False) + '\n')


def _parse_field(text):
    """Split a ``key=value`` CLI field, decoding the value as JSON when possible."""
    key, _, raw = text.partition('=')
    try:
        return key, json.loads(raw)
    except ValueError:
        return key, raw


def main():
    """CLI entry point: append one record to a metrics file.

    .. code-block:: bash

        python -m pydlshogi2.metrics rl/metrics/rl.jsonl \\
            --type event --event iteration_end --run-id rl-20260902 \\
            --set iteration=3 --set checkpoint=rl/checkpoint-003.pth
    """
    import argparse

    parser = argparse.ArgumentParser(description=main.__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('path', help='JSONL file to append to')
    parser.add_argument('--type', default='event', choices=['run', 'metric', 'event'],
                        help='record type')
    parser.add_argument('--run-id', dest='run_id', default=None, help='run id')
    parser.add_argument('--kind', default=None, help="run kind, e.g. 'rl' (type=run only)")
    parser.add_argument('--event', default=None, help='event name (type=event only)')
    parser.add_argument('--set', dest='fields', action='append', default=[],
                        metavar='KEY=VALUE',
                        help='extra field; VALUE is decoded as JSON when it parses')
    args = parser.parse_args()

    record = {'type': args.type, 'timestamp': time.time()}
    if args.type == 'run':
        record['schema_version'] = SCHEMA_VERSION
        record['kind'] = args.kind or 'rl'
        record['started_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
        commit, dirty = git_revision()
        record['git_commit'] = commit
        record['git_dirty'] = dirty
        record['hostname'] = socket.gethostname()
    if args.run_id:
        record['run_id'] = args.run_id
    if args.event:
        record['event'] = args.event
    for field in args.fields:
        key, value = _parse_field(field)
        record[key] = value

    append_record(args.path, record)
    print(args.run_id or '')


if __name__ == '__main__':
    main()
