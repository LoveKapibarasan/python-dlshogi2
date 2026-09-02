Metrics and the dashboard
=========================

Training and self-play both write human-readable text logs, which are fine to
tail but awkward to compare across runs. Passing ``--metrics`` makes them write
the same information a second time as **JSON Lines**, which the Streamlit
dashboard reads back as a browsable development history.

Recording metrics
-----------------

Supervised training:

.. code-block:: bash

   python -m pydlshogi2.train train.hcpe test.hcpe \
       --gpu 0 --epoch 10 --checkpoint checkpoints/checkpoint-{epoch:03}.pth \
       --metrics metrics/train-sl.jsonl

Self-play:

.. code-block:: bash

   python -m pydlshogi2.selfplay checkpoints/checkpoint.pth selfplay.hcpe \
       --games 1000 --metrics metrics/selfplay.jsonl

The reinforcement-learning loop needs no extra flags — it writes everything
under ``$WORKDIR/metrics`` (override with ``METRICS_DIR``):

.. code-block:: text

   rl/metrics/
     rl.jsonl                    # the loop itself: one record per iteration
     selfplay-001-w0.jsonl       # self-play worker 0 of iteration 1
     selfplay-001-w1.jsonl
     train-001.jsonl             # the training run inside iteration 1
     ...

Record schema
-------------

Each file holds one run. The first record describes it, the rest are samples
and events:

.. code-block:: json

   {"type": "run", "run_id": "20260902-101500-1a2b3c4d", "kind": "train",
    "git_commit": "fc71d53...", "git_dirty": false, "hostname": "vast-1",
    "gpu_name": "RTX 4090", "args": {"lr": 0.01, "batchsize": 1024, "...": "..."}}
   {"type": "metric", "run_id": "...", "scope": "interval", "epoch": 1, "step": 100,
    "train_loss_total": 4.31, "test_accuracy_policy": 0.12, "...": "..."}
   {"type": "event", "run_id": "...", "event": "checkpoint", "path": "checkpoints/..."}
   {"type": "event", "run_id": "...", "event": "run_end", "status": "completed"}

``scope`` separates the granularities: ``interval`` samples come from every
``--eval_interval`` steps and use a single test mini-batch, ``epoch`` samples
come from the full test set at the end of an epoch, and self-play uses ``game``
and ``summary``.

Every record is flushed as it is written, so a run killed by a preemption still
leaves a usable history behind.

Runs, resumes and run ids
-------------------------

A run id looks like ``20260902-101500-1a2b3c4d``: a sortable timestamp plus a
random suffix. By default each process gets a fresh one, so a training run that
is preempted and resumed appears as **two** runs — the second one carrying the
``resume`` path in its arguments. Pass ``--run_id`` to keep them merged into one
logical run instead:

.. code-block:: bash

   python -m pydlshogi2.train train.hcpe test.hcpe --gpu 0 --epoch 10 \
       --save_interval 1000 --checkpoint checkpoints/latest.pth \
       --metrics metrics/train-sl.jsonl --run_id sl-2026-09

   # after a preemption: same id, so the dashboard shows one continuous curve
   python -m pydlshogi2.train train.hcpe test.hcpe --gpu 0 --epoch 10 \
       --resume checkpoints/latest.pth --checkpoint checkpoints/latest.pth \
       --metrics metrics/train-sl.jsonl --run_id sl-2026-09

Because the step counter is restored from the checkpoint, the resumed samples
continue on the same step axis either way.

Running the dashboard
---------------------

.. code-block:: bash

   pip install -r dashboard/requirements.txt
   streamlit run dashboard/app.py

The metrics and checkpoint directories are editable in the sidebar and can be
preset with ``DLSHOGI_METRICS_DIR`` and ``DLSHOGI_CHECKPOINT_DIR``. The metrics
directory is scanned recursively for ``*.jsonl``, so pointing it at a directory
of downloaded remote runs works without any preprocessing.

The four tabs are:

Runs
    Every run with its start time, git commit (and whether the tree was dirty),
    host and GPU, hyper-parameters, last step reached and final accuracy.
    Selecting a run shows its full argument list.

学習曲線
    Train/test loss and policy/value accuracy for several runs overlaid on a
    step axis, switchable between the ``interval`` and ``epoch`` granularities.

RL ループ
    Self-play statistics per iteration with the parallel workers summed —
    games, win/draw split, mean game length — plus the artifacts each iteration
    produced.

チェックポイント
    Model files with size and mtime, and on request the architecture embedded
    in a ``.pth``.

Reading the metrics yourself
----------------------------

``dashboard/metrics_store.py`` is standard-library only, so a notebook or a
throwaway script can use it without installing Streamlit:

.. code-block:: python

   import sys; sys.path.insert(0, 'dashboard')
   import metrics_store

   records = metrics_store.load('rl/metrics')
   for run in metrics_store.summarize_runs(records):
       print(run['run_id'], run['kind'], run['status'], run.get('last_step'))

Shell scripts can append a record without any Python of their own:

.. code-block:: bash

   python -m pydlshogi2.metrics rl/metrics/rl.jsonl \
       --type event --event iteration_end --run-id rl-20260902 \
       --set iteration=3 --set checkpoint=rl/checkpoint-003.pth
