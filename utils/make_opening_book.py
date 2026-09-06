"""Generate an opening book for the match arena.

The MCTS in this repository is deterministic: same position, same playout
budget, same move.  Two engines started from the initial position therefore
play the *same game* every time, and a "100 game match" is one game counted a
hundred times.  An opening book fixes that by handing each game a different
starting position — and because :mod:`pydlshogi2.match` replays each line twice
with the colours swapped, the resulting pairs also cancel out most of the
opening's own bias.

Lines are sampled from the policy head alone (one forward pass per ply, no
search), which keeps generation cheap and the positions plausible: a book of
uniformly random moves would mostly test how each engine handles nonsense.
``--temperature`` trades diversity against sanity, and ``--top-k`` keeps the
sampler away from the tail of moves the network considers absurd.

Output is the format ``cshogi.cli`` expects — one ``startpos moves ...`` line
per opening.

Example
-------

.. code-block:: bash

    python utils/make_opening_book.py checkpoints/checkpoint.pth openings.txt \\
        --lines 64 --plies 12 --temperature 1.2 --top-k 8
"""
import argparse
import os
import sys

import numpy as np
import torch
from cshogi import Board, move_to_usi

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pydlshogi2.features import (  # noqa: E402
    FEATURES_NUM, make_input_features, make_move_label,
)
from pydlshogi2.network.policy_value_resnet import load_network  # noqa: E402


def policy_probabilities(model, device, board, features, temperature, top_k):
    """Return the legal moves of ``board`` and a sampling distribution over them.

    :param model: the policy-value network in eval mode.
    :param device: torch device the model lives on.
    :param board: position to evaluate.
    :param features: reusable ``(1, FEATURES_NUM, 9, 9)`` numpy buffer.
    :param temperature: softmax temperature; higher is more uniform.
    :param top_k: keep only this many highest-prior moves (``0`` keeps all).
    :returns: a ``(moves, probabilities)`` tuple, or ``(None, None)`` when the
        position has no legal moves.
    """
    moves = list(board.legal_moves)
    if not moves:
        return None, None

    make_input_features(board, features[0])
    with torch.no_grad():
        x = torch.from_numpy(features).to(device)
        policy_logits, _ = model(x)
    logits = policy_logits[0].cpu().numpy()

    legal_logits = np.array(
        [logits[make_move_label(move, board.turn)] for move in moves],
        dtype=np.float64)

    if top_k and 0 < top_k < len(moves):
        keep = np.argpartition(legal_logits, -top_k)[-top_k:]
        moves = [moves[i] for i in keep]
        legal_logits = legal_logits[keep]

    legal_logits /= max(temperature, 1e-6)
    legal_logits -= legal_logits.max()
    probabilities = np.exp(legal_logits)
    probabilities /= probabilities.sum()
    return moves, probabilities


def sample_line(model, device, features, plies, temperature, top_k):
    """Sample one opening line.

    :param plies: number of plies to play out.
    :returns: list of USI move strings, or ``None`` when the line ran into a
        finished game (which would make a useless opening).
    """
    board = Board()
    usi_moves = []
    for _ in range(plies):
        if board.is_game_over() or board.is_draw():
            return None
        moves, probabilities = policy_probabilities(
            model, device, board, features, temperature, top_k)
        if moves is None:
            return None
        move = moves[int(np.random.choice(len(moves), p=probabilities))]
        usi_moves.append(move_to_usi(move))
        board.push(move)
    # 開始局面がすでに終わっている、あるいは詰んでいる本は使い物にならない
    if board.is_game_over() or board.is_draw():
        return None
    return usi_moves


def main():
    """Parse arguments, sample the book and write it out."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('modelfile', help='checkpoint (.pth) to sample the policy from')
    parser.add_argument('output', help='output book file')
    parser.add_argument('--lines', type=int, default=64,
                        help='number of distinct opening lines to produce')
    parser.add_argument('--plies', type=int, default=12,
                        help='plies per opening line')
    parser.add_argument('--temperature', type=float, default=1.2,
                        help='policy softmax temperature; higher is more diverse')
    parser.add_argument('--top-k', type=int, default=8,
                        help='sample only among the k highest-prior moves '
                             '(0 = all legal moves)')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id (-1 for CPU)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument('--max-attempts', type=int, default=None,
                        help='give up after this many samples (default: 20x lines)')
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    device = torch.device('cuda:{}'.format(args.gpu)) if args.gpu >= 0 else torch.device('cpu')
    model, _ = load_network(args.modelfile, device)
    model.eval()

    features = np.zeros((1, FEATURES_NUM, 9, 9), dtype=np.float32)
    max_attempts = args.max_attempts or args.lines * 20

    # 同じ手順を2度入れても対局が増えないので重複は捨てる
    seen = set()
    lines = []
    attempts = 0
    while len(lines) < args.lines and attempts < max_attempts:
        attempts += 1
        usi_moves = sample_line(model, device, features, args.plies,
                                args.temperature, args.top_k)
        if not usi_moves:
            continue
        key = ' '.join(usi_moves)
        if key in seen:
            continue
        seen.add(key)
        lines.append(key)

    directory = os.path.dirname(os.path.abspath(args.output))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write('startpos moves {}\n'.format(line))

    print('wrote {} opening lines to {} ({} samples, {} duplicates or dead lines '
          'discarded)'.format(len(lines), args.output, attempts, attempts - len(lines)))
    if len(lines) < args.lines:
        print('warning: only {} of the requested {} lines were distinct; raise '
              '--temperature, --top-k or --plies for more diversity'.format(
                  len(lines), args.lines), file=sys.stderr)


if __name__ == '__main__':
    main()
