import math

from cshogi import *

# 移動方向を表す定数
MOVE_DIRECTION = [
    UP, UP_LEFT, UP_RIGHT, LEFT, RIGHT, DOWN, DOWN_LEFT, DOWN_RIGHT,
    UP2_LEFT, UP2_RIGHT,
    UP_PROMOTE, UP_LEFT_PROMOTE, UP_RIGHT_PROMOTE, LEFT_PROMOTE, RIGHT_PROMOTE, DOWN_PROMOTE, DOWN_LEFT_PROMOTE, DOWN_RIGHT_PROMOTE,
    UP2_LEFT_PROMOTE, UP2_RIGHT_PROMOTE
] = range(20)

# 入力特徴量の数
FEATURES_NUM = len(PIECE_TYPES) * 2 + sum(MAX_PIECES_IN_HAND) * 2

# 移動を表すラベルの数
MOVE_PLANES_NUM = len(MOVE_DIRECTION) + len(HAND_PIECES)
MOVE_LABELS_NUM = MOVE_PLANES_NUM * 81

# 入力特徴量を作成
def make_input_features(board, features):
    # 入力特徴量を0に初期化
    features.fill(0)

    # 盤上の駒
    if board.turn == BLACK:
        board.piece_planes(features)
        pieces_in_hand = board.pieces_in_hand
    else:
        board.piece_planes_rotate(features)
        pieces_in_hand = reversed(board.pieces_in_hand)
    # 持ち駒
    i = 28
    for hands in pieces_in_hand:
        for num, max_num in zip(hands, MAX_PIECES_IN_HAND):
            features[i:i+num].fill(1)
            i += max_num

# 移動を表すラベルを作成
def make_move_label(move, color):
    if not move_is_drop(move):  # 駒の移動
        to_sq = move_to(move)
        from_sq = move_from(move)

        # 後手の場合盤を回転
        if color == WHITE:
            to_sq = 80 - to_sq
            from_sq = 80 - from_sq

        # 移動方向
        to_x, to_y = divmod(to_sq, 9)
        from_x, from_y = divmod(from_sq, 9)
        dir_x = to_x - from_x
        dir_y = to_y - from_y
        if dir_y < 0:
            if dir_x == 0:
                move_direction = UP
            elif dir_y == -2 and dir_x == -1:
                move_direction = UP2_RIGHT
            elif dir_y == -2 and dir_x == 1:
                move_direction = UP2_LEFT
            elif dir_x < 0:
                move_direction = UP_RIGHT
            else:  # dir_x > 0
                move_direction = UP_LEFT
        elif dir_y == 0:
            if dir_x < 0:
                move_direction = RIGHT
            else:  # dir_x > 0
                move_direction = LEFT
        else:  # dir_y > 0
            if dir_x == 0:
                move_direction = DOWN
            elif dir_x < 0:
                move_direction = DOWN_RIGHT
            else:  # dir_x > 0
                move_direction = DOWN_LEFT

        # 成り
        if move_is_promotion(move):
            move_direction += 10
    else:  # 駒打ち
        to_sq = move_to(move)
        # 後手の場合盤を回転
        if color == WHITE:
            to_sq = 80 - to_sq

        # 駒打ちの移動方向
        move_direction = len(MOVE_DIRECTION) + move_drop_hand_piece(move)

    return move_direction * 81 + to_sq

# 評価値を勝率に変換するデフォルト係数
# MCTSの評価値変換 (cp = -log(1/p - 1) * 600) と整合する
DEFAULT_EVAL_COEF = 600.0

# 対局結果から報酬を作成
def make_result(game_result, color):
    if color == BLACK:
        if game_result == BLACK_WIN:
            return 1
        if game_result == WHITE_WIN:
            return 0
    else:
        if game_result == BLACK_WIN:
            return 0
        if game_result == WHITE_WIN:
            return 1
    return 0.5

# 評価値を手番側から見た勝率に変換
def make_eval_winrate(eval_value, color, eval_coef=DEFAULT_EVAL_COEF):
    """Convert a stored centipawn-like evaluation into a side-to-move win rate.

    The evaluation stored in an HCPE record is from **black's perspective**
    (see ``utils/csa_to_hcpe.py``); it is flipped here when white is to move so
    the returned probability is always from the perspective of the side to move.

    :param eval_value: the raw ``eval`` field from the HCPE record.
    :param color: side to move at the position (``cshogi.BLACK`` / ``WHITE``).
    :param eval_coef: temperature of the sigmoid mapping centipawns to a
        probability.
    :returns: win probability in ``[0, 1]`` for the side to move.
    """
    eval_stm = eval_value if color == BLACK else -eval_value
    return 1.0 / (1.0 + math.exp(-eval_stm / eval_coef))

# 対局結果と評価値をブレンドした価値教師を作成
def make_value_target(game_result, eval_value, color, val_lambda=1.0, eval_coef=DEFAULT_EVAL_COEF):
    """Blend the game outcome with the search evaluation into a value target.

    :param game_result: the ``gameResult`` field from the HCPE record.
    :param eval_value: the ``eval`` field from the HCPE record.
    :param color: side to move at the position.
    :param val_lambda: weight on the game outcome; ``1.0`` reproduces the
        outcome-only target, ``0.0`` uses the evaluation only.
    :param eval_coef: temperature passed to :func:`make_eval_winrate`.
    :returns: blended win-probability target in ``[0, 1]``.
    """
    z = make_result(game_result, color)
    if val_lambda >= 1.0:
        return z
    p = make_eval_winrate(eval_value, color, eval_coef)
    return val_lambda * z + (1.0 - val_lambda) * p
