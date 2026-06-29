"""Convert human CSA game records into rating-bucketed HCPE training data.

This is part of the **human-like** AI effort (a separate goal from the strong
self-play engine): instead of maximising strength, we imitate how humans of a
given skill level play, in the spirit of `Maia Chess
<https://maiachess.com/>`_. Each position is bucketed by the rating of the
**player to move**, and the training target is the move that player actually
chose.

Output layout (one sub-directory per rating band)::

    <out_dir>/
        0000-1499/ {train,test}.hcpe
        1500-1799/ {train,test}.hcpe
        1800-2099/ {train,test}.hcpe
        ...
        unrated/   {train,test}.hcpe   # only with --allow_unrated

Train per band with the normal trainer, e.g.::

    python -m pydlshogi2.train out/1800-2099/train.hcpe out/1800-2099/test.hcpe \\
        --gpu 0 --val_lambda 1.0

Use ``--val_lambda 1.0`` (outcome-only value) since for human imitation the
policy (move prediction), not the value, is what matters; evaluate with policy
move-matching accuracy rather than playing strength.

Example
-------

.. code-block:: bash

    python human/csa_to_hcpe_by_rating.py ~/kif out \\
        --bands 1500,1800,2100,2400 --filter_moves 20 --test_ratio 0.05
"""
import argparse
import bisect
import glob
import os

import numpy as np
from cshogi import CSA, Board, BLACK, HuffmanCodedPosAndEval, move16

# 16bit格納のための評価値クリップ
EVAL_CLIP = 32767


def band_labels(edges):
    """Build human-readable band directory names from the rating edges.

    :param edges: sorted list of rating boundaries, e.g. ``[1500, 1800, 2100]``.
    :returns: list of ``len(edges)+1`` labels, one per band.
    """
    labels = []
    lo = 0
    for e in edges:
        labels.append('{:04d}-{:04d}'.format(lo, e - 1))
        lo = e
    labels.append('{:04d}-up'.format(lo))
    return labels


def mover_rating(ratings, turn):
    """Return the rating of the side to move.

    :param ratings: ``(black_rating, white_rating)`` from the CSA record.
    :param turn: side to move (``cshogi.BLACK`` / ``WHITE``).
    :returns: the mover's rating, or ``0`` when unavailable.
    """
    if ratings is None or len(ratings) < 2:
        return 0
    return ratings[0] if turn == BLACK else ratings[1]


class BandWriter:
    """Lazily-opened per-band, per-split HCPE output files.

    :param out_dir: root output directory.
    :param labels: band labels (sub-directory names).
    """

    def __init__(self, out_dir, labels):
        self.out_dir = out_dir
        self.labels = labels
        self.files = {}            # (band_index, split) -> file handle
        self.counts = {}           # (band_index, split) -> records written

    def write(self, band_index, split, hcpes):
        """Append a block of HCPE records for a band/split.

        :param band_index: index into ``labels``.
        :param split: ``'train'`` or ``'test'``.
        :param hcpes: structured ndarray of HuffmanCodedPosAndEval records.
        """
        key = (band_index, split)
        f = self.files.get(key)
        if f is None:
            d = os.path.join(self.out_dir, self.labels[band_index])
            os.makedirs(d, exist_ok=True)
            f = open(os.path.join(d, split + '.hcpe'), 'wb')
            self.files[key] = f
            self.counts[key] = 0
        hcpes.tofile(f)
        self.counts[key] += len(hcpes)

    def close(self):
        """Close all open files."""
        for f in self.files.values():
            f.close()

    def summary(self):
        """Return a sorted list of ``(label, split, count)`` rows."""
        rows = []
        for (band_index, split), count in self.counts.items():
            rows.append((self.labels[band_index], split, count))
        return sorted(rows)


def convert(csa_dir, out_dir, edges, filter_moves, test_ratio, allow_unrated):
    """Convert a directory of CSA files into rating-bucketed HCPE data.

    :param csa_dir: directory searched recursively for ``*.csa`` files.
    :param out_dir: output root directory.
    :param edges: sorted rating boundaries defining the bands.
    :param filter_moves: skip games shorter than this many plies.
    :param test_ratio: fraction of games assigned to the test split.
    :param allow_unrated: keep games without ratings in an ``unrated`` band
        instead of skipping them.
    """
    labels = band_labels(edges)
    unrated_index = None
    if allow_unrated:
        unrated_index = len(labels)
        labels = labels + ['unrated']

    writer = BandWriter(out_dir, labels)
    board = Board()
    # 1局分のレコードを貯めるバッファ (上限は手数次第で拡張)
    buffer = np.zeros(2048, HuffmanCodedPosAndEval)

    csa_files = glob.glob(os.path.join(csa_dir, '**', '*.csa'), recursive=True)
    games = 0
    positions = 0
    skipped = 0

    for filepath in csa_files:
        for kif in CSA.Parser.parse_file(filepath):
            if kif.endgame not in ('%TORYO', '%SENNICHITE', '%KACHI'):
                continue
            if len(kif.moves) < filter_moves:
                continue

            has_rating = kif.ratings is not None and len(kif.ratings) >= 2 \
                and min(kif.ratings) > 0
            if not has_rating and not allow_unrated:
                skipped += 1
                continue

            split = 'test' if np.random.random() < test_ratio else 'train'

            board.set_sfen(kif.sfen)
            # scoresは無い場合がある (KIF由来のCSAなど)。無ければ0扱い。
            scores = getattr(kif, 'scores', None) or []
            # band_index -> list of buffer rows for this game/band
            per_band = {}
            ok = True
            p = 0
            try:
                for i, move in enumerate(kif.moves):
                    if not board.is_legal(move):
                        raise ValueError('illegal move')

                    if has_rating:
                        r = mover_rating(kif.ratings, board.turn)
                        band_index = bisect.bisect_right(edges, r)
                    else:
                        band_index = unrated_index

                    rec = buffer[p]
                    p += 1
                    board.to_hcp(rec['hcp'])
                    score = scores[i] if i < len(scores) else 0
                    eval_value = min(EVAL_CLIP, max(int(score), -EVAL_CLIP))
                    rec['eval'] = eval_value if board.turn == BLACK else -eval_value
                    rec['bestMove16'] = move16(move)
                    rec['gameResult'] = kif.win
                    per_band.setdefault(band_index, []).append(p - 1)

                    board.push(move)
            except Exception:
                ok = False

            if not ok or p == 0:
                skipped += 1
                continue

            for band_index, rows in per_band.items():
                writer.write(band_index, split, buffer[rows])
            games += 1
            positions += p

    writer.close()

    print('games={} positions={} skipped_games={}'.format(games, positions, skipped))
    print('per-band record counts:')
    for label, split, count in writer.summary():
        print('  {:>10}  {:<5}  {}'.format(label, split, count))


def main():
    """Parse arguments and run :func:`convert`."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('csa_dir', help='directory of CSA records (searched recursively)')
    parser.add_argument('out_dir', help='output directory for per-band HCPE files')
    parser.add_argument('--bands', default='1500,1800,2100,2400',
                        help='comma-separated rating boundaries (e.g. 1500,1800,2100,2400)')
    parser.add_argument('--filter_moves', type=int, default=20,
                        help='skip games shorter than this many plies')
    parser.add_argument('--test_ratio', type=float, default=0.05,
                        help='fraction of games assigned to the test split')
    parser.add_argument('--allow_unrated', action='store_true',
                        help='keep games without ratings in an "unrated" band')
    parser.add_argument('--seed', type=int, default=0, help='random seed for the train/test split')
    args = parser.parse_args()

    np.random.seed(args.seed)
    edges = sorted(int(x) for x in args.bands.split(',') if x.strip())
    convert(args.csa_dir, args.out_dir, edges, args.filter_moves,
            args.test_ratio, args.allow_unrated)


if __name__ == '__main__':
    main()
