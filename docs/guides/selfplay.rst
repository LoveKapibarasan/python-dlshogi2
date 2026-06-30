Self-play reinforcement learning
================================

After a supervised warm start, the engine can improve itself by playing games
against itself and training on the results — an expert-iteration / AlphaZero
style loop.

Generating games
----------------

.. code-block:: bash

   python -m pydlshogi2.selfplay checkpoints/checkpoint.pth selfplay.hcpe \
       --games 1000 --playouts 800 --gpu 0

Each played position is written as an HCPE record with the most-visited move as
the policy target, the MCTS root win rate as ``eval`` and the final game result
as ``gameResult``. The move actually played is sampled from the visit counts
with a temperature, and Dirichlet noise is mixed into the root prior, so games
stay diverse. See :mod:`pydlshogi2.selfplay`.

Parallel generation
-------------------

A single self-play process is CPU-bound on the MCTS tree and leaves the GPU
mostly idle. ``selfplay_parallel.sh`` runs several workers (each with a distinct
``--seed``) that share the GPU and concatenates their output, multiplying
throughput roughly by the worker count:

.. code-block:: bash

   WORKERS=8 GAMES=1000 PLAYOUTS=400 GPU=0 \
       ./selfplay_parallel.sh checkpoints/checkpoint.pth selfplay.hcpe

The output is a plain concatenation (no de-duplication): in self-play the same
position reached in different games carries different outcomes, and those
repeated samples are exactly the value signal training averages over.

Useful options
--------------

``--temperature`` / ``--temp_cutoff``
    Sampling temperature and the ply after which moves become greedy.

``--dirichlet_alpha`` / ``--noise_eps``
    Root exploration noise (``noise_eps = 0`` disables it).

``--playouts``
    MCTS playouts per move; higher gives stronger targets but slower
    generation.

The full loop
-------------

``rl_loop.sh`` automates *generate -> train -> promote* for many iterations:

.. code-block:: bash

   ./rl_loop.sh checkpoints/checkpoint.pth

Tune it with environment variables (``ITERATIONS``, ``GAMES``, ``PLAYOUTS``,
``EPOCHS``, ``LR``, ``VAL_LAMBDA``, ``GPU``, ``WORKDIR``). Training during the
loop uses a lower learning rate and ``val_lambda`` below ``1.0`` to blend the
game result with the bootstrapped MCTS value.

Cleaning the data
-----------------

Self-play data over-represents opening positions. De-duplicate before training
on the accumulated set:

.. code-block:: bash

   python utils/hcpe_dedup.py merged.hcpe rl/selfplay-*.hcpe
