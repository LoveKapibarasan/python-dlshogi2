python-dlshogi2 documentation
==============================

python-dlshogi2 is a deep-learning shogi engine: a policy-value ResNet trained
on game records (supervised learning) and optionally improved through
self-play reinforcement learning, played via Monte Carlo Tree Search over the
USI protocol.

This documentation combines hand-written guides with an API reference generated
from the in-source docstrings.

.. toctree::
   :maxdepth: 2
   :caption: Guides

   guides/training
   guides/selfplay
   guides/playing

.. toctree::
   :maxdepth: 2
   :caption: API reference

   api/features
   api/network
   api/dataloader
   api/selfplay
   api/players
   api/uct
   api/utils

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
