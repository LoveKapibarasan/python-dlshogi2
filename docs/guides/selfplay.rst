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
