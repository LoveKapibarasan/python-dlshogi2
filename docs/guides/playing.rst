Playing (USI engine)
====================

The engine speaks the USI protocol and connects to any USI-compatible GUI
(ShogiGUI, Shogidokoro, ...).

PyTorch backend
---------------

.. code-block:: bash

   python -m pydlshogi2.player.mcts_player

ONNX backend
------------

Export a trained checkpoint to ONNX first (the exported graph uses the same
feature representation as the PyTorch engine):

.. code-block:: bash

   python utils/export_onnx.py checkpoints/checkpoint.pth model/model.onnx
   python -m pydlshogi2.player.onnx_player

Search options
--------------

In addition to ``modelfile``, ``gpu_id``, ``batchsize``, ``resign_threshold``,
``temperature`` and the time-control margins, the engine exposes:

``c_puct`` / ``c_base``
    PUCT exploration constant and the base of its slow-growing log term.

``fpu_reduction``
    First Play Urgency reduction applied to unvisited children. See
    :meth:`pydlshogi2.player.mcts_player.MCTSPlayer.select_max_ucb_child`.

``mate_root_ply``
    Depth of the root mate search run once at the start of each ``go`` (odd
    number of plies; ``1`` keeps only the built-in 1-ply check).

``tensorrt`` (ONNX backend only)
    Enable the TensorRT execution provider, falling back to CUDA then CPU.
