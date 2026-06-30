import onnxruntime
import numpy as np

from pydlshogi2.player.mcts_player import MCTSPlayer
from pydlshogi2.features import FEATURES_NUM, make_input_features, make_move_label

class OnnxPlayer(MCTSPlayer):
    # USIエンジンの名前
    name = 'python-dlshogi-onnx'
    # デフォルトモデル
    DEFAULT_MODELFILE = 'model/rl-020.onnx'

    # モデルのロード
    def load_model(self):
        self.session = onnxruntime.InferenceSession(self.modelfile, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

    # 入力特徴量の初期化
    def init_features(self):
        self.features = np.empty((self.batch_size, FEATURES_NUM, 9, 9), dtype=np.float32)

    # 入力特徴量の作成
    def make_input_features(self, board):
        make_input_features(board, self.features[self.current_batch_index])

    # 推論
    def infer(self):
        x = self.features[0:self.current_batch_index]
        policy, value = self.session.run(
            ['output_policy', 'output_value'],
            {'input': x}
        )
        return policy, value

    # 着手を表すラベル作成
    def make_move_label(self, move, color):
        return make_move_label(move, color)

if __name__ == '__main__':
    player = OnnxPlayer()
    player.run()
