Supervised training
===================

Train the policy-value network on game records converted to the HCPE format.

Data preparation
----------------

Convert CSA game records to HCPE with ``utils/csa_to_hcpe.py``:

.. code-block:: bash

   python utils/csa_to_hcpe.py <csa_dir> train.hcpe test.hcpe \
       --filter_moves 50 --filter_rating 3500 --test_ratio 0.1

Training
--------

.. code-block:: bash

   python -m pydlshogi2.train train.hcpe test.hcpe \
       --gpu 0 --epoch 10 --batchsize 1024 --lr 0.01 \
       --checkpoint checkpoints/checkpoint-{epoch:03}.pth

Key options
-----------

``--blocks`` / ``--channels`` / ``--fcl`` / ``--no_se``
    Network architecture. The default is a Squeeze-and-Excitation ResNet with
    20 blocks of 256 channels. The chosen architecture is **embedded in the
    checkpoint**, so the players and the ONNX exporter reconstruct it
    automatically. Resuming with ``--resume`` ignores these flags and uses the
    checkpoint's own architecture. Legacy checkpoints (saved before the config
    was embedded) load as a ``10 x 192`` SE-free network.

``--val_lambda`` (default ``0.333``)
    Weight on the game outcome in the value target. The remaining weight is
    placed on the stored search evaluation, mapped to a win rate by
    :func:`pydlshogi2.features.make_eval_winrate`. ``1.0`` reproduces the
    outcome-only target.

``--eval_coef`` (default ``600``)
    Sigmoid temperature for the centipawn-to-win-rate mapping; matches the
    engine's PV ``cp`` conversion.

``--amp``
    Enable bfloat16 autocast (mixed precision) on CUDA.

``--compile``
    Wrap the model with :func:`torch.compile`.

``--save_interval`` (default ``0``)
    Save a checkpoint every N steps in addition to the epoch-end save. Use it
    on preemptible instances so an interruption mid-epoch loses at most N
    steps.

Interrupting and resuming
-------------------------

Training is preemption-safe. On ``SIGTERM`` or ``SIGINT`` it checkpoints after
the current step and exits with status ``0``; combined with ``--save_interval``
this makes spot/preemptible instances safe. Resume by pointing ``--resume`` and
``--checkpoint`` at the same path:

.. code-block:: bash

   # initial run
   python -m pydlshogi2.train train.hcpe test.hcpe --gpu 0 --epoch 10 \
       --save_interval 1000 --checkpoint checkpoints/latest.pth

   # continue after a preemption
   python -m pydlshogi2.train train.hcpe test.hcpe --gpu 0 --epoch 10 \
       --resume checkpoints/latest.pth --checkpoint checkpoints/latest.pth

``--epoch`` counts *additional* epochs on resume. The model weights, optimizer
state, global step and architecture are all restored from the checkpoint.

Checkpoint contents
-------------------

Each checkpoint is a dict with ``epoch``, ``t`` (total steps), ``model``
(weights), ``optimizer`` (state) and ``network`` (architecture config). See
:func:`pydlshogi2.network.policy_value_resnet.load_network`.
