"""Export a trained :class:`~pydlshogi2.network.policy_value_resnet.PolicyValueNetwork`
checkpoint to the ONNX format.

The exported model uses the **same single-input feature representation** as
:mod:`pydlshogi2.features` (the representation used by the PyTorch
:class:`~pydlshogi2.player.mcts_player.MCTSPlayer`), so the resulting ``.onnx``
file is directly consumable by
:class:`~pydlshogi2.player.onnx_player.OnnxPlayer`.

Graph signature
---------------

* input ``input``: ``float32[batch, FEATURES_NUM, 9, 9]``
* output ``output_policy``: ``float32[batch, MOVE_LABELS_NUM]`` (raw logits)
* output ``output_value``: ``float32[batch, 1]`` (win probability, ``sigmoid``
  applied inside the graph)

The ``batch`` axis is dynamic so a single exported model can serve any batch
size at inference time.

Example
-------

.. code-block:: bash

    python utils/export_onnx.py checkpoints/checkpoint.pth model/model.onnx
"""
import argparse

import torch
import torch.nn as nn

from pydlshogi2.features import FEATURES_NUM
from pydlshogi2.network.policy_value_resnet import PolicyValueNetwork, load_network


class InferenceModel(nn.Module):
    """Wrap a :class:`PolicyValueNetwork` so the value head emits a win
    probability rather than a raw logit.

    Baking the ``sigmoid`` into the graph keeps the ONNX inference contract
    identical to :meth:`pydlshogi2.player.mcts_player.MCTSPlayer.infer`, which
    returns ``(policy_logits, sigmoid(value_logits))``.

    :param model: the trained policy-value network to wrap.
    """

    def __init__(self, model: PolicyValueNetwork):
        super().__init__()
        self.model = model

    def forward(self, x):
        """Run a forward pass.

        :param x: input feature tensor of shape ``(batch, FEATURES_NUM, 9, 9)``.
        :returns: a ``(policy_logits, value_probability)`` tuple.
        """
        policy, value = self.model(x)
        return policy, torch.sigmoid(value)


def export(modelfile: str, onnxfile: str, device: str = "cpu") -> None:
    """Load a checkpoint and write an ONNX model to disk.

    :param modelfile: path to a ``.pth`` checkpoint produced by
        :mod:`pydlshogi2.train` (must contain a ``'model'`` state dict).
    :param onnxfile: destination path for the exported ``.onnx`` file.
    :param device: torch device string used while tracing the graph.
    """
    model, _ = load_network(modelfile, device)
    model.eval()

    wrapper = InferenceModel(model)
    wrapper.eval()

    # A batch of 1 is enough to trace the graph; the batch axis is marked
    # dynamic below so the exported model serves any batch size.
    dummy_input = torch.zeros((1, FEATURES_NUM, 9, 9), dtype=torch.float32, device=device)

    torch.onnx.export(
        wrapper,
        dummy_input,
        onnxfile,
        input_names=["input"],
        output_names=["output_policy", "output_value"],
        dynamic_axes={
            "input": {0: "batch"},
            "output_policy": {0: "batch"},
            "output_value": {0: "batch"},
        },
        opset_version=17,
    )
    print(f"Exported {modelfile} -> {onnxfile}")


def main() -> None:
    """Parse command-line arguments and run :func:`export`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("modelfile", help="input PyTorch checkpoint (.pth)")
    parser.add_argument("onnxfile", help="output ONNX model (.onnx)")
    parser.add_argument("--device", default="cpu", help="torch device for tracing")
    args = parser.parse_args()
    export(args.modelfile, args.onnxfile, args.device)


if __name__ == "__main__":
    main()
