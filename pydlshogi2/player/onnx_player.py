"""ONNX inference backend for the MCTS engine.

:class:`OnnxPlayer` subclasses :class:`~pydlshogi2.player.mcts_player.MCTSPlayer`
and swaps only the model loading, feature buffer and inference steps so the
network runs through :mod:`onnxruntime` instead of PyTorch. Everything else
(search, time control, USI handling) is inherited unchanged.

The player consumes models exported by :mod:`utils.export_onnx`, which use the
same single-input feature representation as :mod:`pydlshogi2.features`. This
keeps the ONNX and PyTorch backends bit-for-bit consistent: a checkpoint
trained with :mod:`pydlshogi2.train` can be exported and played without any
feature mismatch.
"""
import numpy as np
import onnxruntime

from pydlshogi2.features import FEATURES_NUM, make_input_features
from pydlshogi2.player.mcts_player import MCTSPlayer


class OnnxPlayer(MCTSPlayer):
    """MCTS player that performs network inference with onnxruntime."""

    #: USI engine name reported to the GUI.
    name = "python-dlshogi-onnx"
    #: Default model path used when ``modelfile`` is not overridden via USI.
    DEFAULT_MODELFILE = "model/model.onnx"

    def __init__(self):
        super().__init__()
        #: Whether to enable the TensorRT execution provider (fastest, requires
        #: a TensorRT-enabled onnxruntime build).
        self.tensorrt = False

    def usi(self):
        """Print USI options, adding the ONNX-specific ``tensorrt`` toggle."""
        super().usi()
        print("option name tensorrt type check default false")

    def setoption(self, args):
        """Handle the ``tensorrt`` option, delegating the rest to the base."""
        if args[1] == "tensorrt":
            self.tensorrt = args[3] == "true"
        else:
            super().setoption(args)

    def load_model(self):
        """Create the onnxruntime inference session."""
        if self.gpu_id >= 0:
            providers = []
            if self.tensorrt:
                providers.append(("TensorrtExecutionProvider", {"device_id": self.gpu_id}))
            providers.append(("CUDAExecutionProvider", {"device_id": self.gpu_id}))
            providers.append("CPUExecutionProvider")
        else:
            providers = ["CPUExecutionProvider"]
        self.session = onnxruntime.InferenceSession(self.modelfile, providers=providers)

    def init_features(self):
        """Allocate the single-input feature buffer of shape
        ``(batch_size, FEATURES_NUM, 9, 9)``."""
        self.features = np.empty((self.batch_size, FEATURES_NUM, 9, 9), dtype=np.float32)

    def make_input_features(self, board):
        """Write ``board`` into the current batch slot of the feature buffer."""
        make_input_features(board, self.features[self.current_batch_index])

    def infer(self):
        """Run inference over the queued positions."""
        x = self.features[0:self.current_batch_index]
        policy, value = self.session.run(["output_policy", "output_value"], {"input": x})
        return policy, value


if __name__ == "__main__":
    player = OnnxPlayer()
    player.run()
