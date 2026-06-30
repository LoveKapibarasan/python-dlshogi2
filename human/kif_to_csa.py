"""Convert Shogi Wars KIF game records into CSA, preserving dan/kyu ranks.

Shogi Wars exports games as KIF with a few non-standard quirks that ``cshogi``
does not handle on its own:

* the file is UTF-8 (``cshogi.KIF.parse_file`` assumes CP932);
* the player **rank** lives on dedicated ``先手段級：`` / ``後手段級：`` lines and
  is *not* exposed by the parser;
* the game ends with a bare ``投了`` / ``切れ負け`` / ``千日手`` token rather than
  the ``まで○手で△手の勝ち`` line ``cshogi`` recognises, so ``win`` / ``endgame``
  come back as ``None``.

This converter reads each KIF (encoding auto-detected), parses the rank lines
itself, infers the result from the final token, and writes standard CSA. The
dan/kyu rank is mapped to an **ordinal** and emitted as a floodgate-style
``'black_rate:`` / ``'white_rate:`` comment line, so the existing
:mod:`human.csa_to_hcpe_by_rating` converter can bucket positions by skill.

Rank ordinal (higher = stronger)::

    30級=1, 29級=2, ..., 1級=30, 初段=31, 二段=32, ..., 九段=39

Example
-------

.. code-block:: bash

    python human/kif_to_csa.py ~/kifs/kif_data converted     # -> converted/shogiwars.csa
    python human/csa_to_hcpe_by_rating.py converted out --bands 30,33  # kyu / 1-3dan / 4dan+
"""
import argparse
import glob
import os
import re

from cshogi import KIF, CSA, Board, BLACK, WHITE, BLACK_WIN, WHITE_WIN, DRAW

# 段級行 (例: 先手段級：六段, 後手段級：2級)
RANK_RE = {'先手': re.compile(r'先手段級[：:]\s*(\S+)'),
           '後手': re.compile(r'後手段級[：:]\s*(\S+)')}

# 漢数字 (将棋ウォーズの段級に出る範囲)
_KANJI = {'初': 1, '一': 1, '二': 2, '三': 3, '四': 4, '五': 5,
          '六': 6, '七': 7, '八': 8, '九': 9}

# 終局トークン -> (手番側が勝ちか/負けか/引き分け, CSA終局符号)
#   LOSE: 手番側が負け(相手の勝ち), WIN: 手番側の勝ち, DRAW: 引き分け
_RESIGN_TOKENS = ('投了', '切れ負け', '反則負け', '詰み', '詰', 'time-up')
_WIN_TOKENS = ('入玉勝ち', '宣言勝ち', 'トライ', '反則勝ち')
_DRAW_TOKENS = ('千日手', '持将棋')


def kanji_number(s):
    """Parse a (possibly kanji / full-width) numeral used in a rank.

    :param s: numeral string such as ``'六'``, ``'2'``, ``'十五'`` or ``'初'``.
    :returns: the integer value, or ``None`` if unparseable.
    """
    s = s.translate(str.maketrans('０１２３４５６７８９', '0123456789'))
    if s.isdigit():
        return int(s)
    if '十' in s:
        tens, _, ones = s.partition('十')
        t = _KANJI.get(tens, 1) if tens else 1
        o = _KANJI.get(ones, 0) if ones else 0
        return t * 10 + o
    return _KANJI.get(s)


def rank_to_ordinal(rank_str):
    """Map a dan/kyu rank string to an ordinal (higher = stronger).

    :param rank_str: e.g. ``'六段'``, ``'1級'``, ``'初段'``.
    :returns: ordinal in ``1..39``, or ``None`` if it cannot be parsed.
    """
    if not rank_str:
        return None
    if rank_str.endswith('級'):
        n = kanji_number(rank_str[:-1])
        return None if n is None else max(1, 31 - n)
    if rank_str.endswith('段'):
        n = kanji_number(rank_str[:-1])
        return None if n is None else 30 + n
    return None


def parse_ranks(text):
    """Extract ``(black_ordinal, white_ordinal)`` from raw KIF text."""
    out = []
    for side in ('先手', '後手'):
        m = RANK_RE[side].search(text)
        out.append(rank_to_ordinal(m.group(1)) if m else None)
    return out[0], out[1]


def infer_result(text, num_moves):
    """Infer the CSA endgame token and winner from the final KIF line.

    :param text: raw KIF text.
    :param num_moves: number of plies parsed (to find the side to move).
    :returns: ``(endgame_token, game_result)`` or ``(None, None)`` if the game
        did not end decisively / is unrecognised.
    """
    # 末尾の非空行を見る
    last = ''
    for line in reversed(text.splitlines()):
        if line.strip():
            last = line.strip()
            break

    # 手番側 (num_moves手指した後の手番)
    side_to_move = BLACK if num_moves % 2 == 0 else WHITE
    opponent = WHITE if side_to_move == BLACK else BLACK

    if any(tok in last for tok in _DRAW_TOKENS):
        return '%SENNICHITE', DRAW
    if any(tok in last for tok in _WIN_TOKENS):
        win = BLACK_WIN if side_to_move == BLACK else WHITE_WIN
        return '%KACHI', win
    if any(tok in last for tok in _RESIGN_TOKENS):
        win = BLACK_WIN if opponent == BLACK else WHITE_WIN
        return '%TORYO', win
    return None, None


def read_text(path):
    """Read a KIF file, trying UTF-8 (with BOM) then CP932."""
    data = open(path, 'rb').read()
    for enc in ('utf-8-sig', 'utf-8', 'cp932'):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode('utf-8', errors='ignore')


def convert(kif_dir, out_dir, shards):
    """Convert every KIF under ``kif_dir`` into CSA shard files.

    :param kif_dir: directory searched recursively for ``*.kif``.
    :param out_dir: output directory for the ``shogiwars-*.csa`` shards.
    :param shards: number of output CSA files to spread games across (keeps any
        single file from getting unwieldy).
    :returns: ``(converted, skipped)`` game counts.
    """
    os.makedirs(out_dir, exist_ok=True)
    exporters = [CSA.Exporter(os.path.join(out_dir, 'shogiwars-{:02d}.csa'.format(s)))
                 for s in range(shards)]

    kif_files = glob.glob(os.path.join(kif_dir, '**', '*.kif'), recursive=True)
    converted = 0
    skipped = 0

    for idx, path in enumerate(kif_files):
        try:
            text = read_text(path)
            g = KIF.Parser.parse_str(text)
            g = g[0] if isinstance(g, list) else g
        except Exception:
            skipped += 1
            continue

        if not g.moves:
            skipped += 1
            continue

        black_ord, white_ord = parse_ranks(text)
        endgame, _ = infer_result(text, len(g.moves))
        if endgame is None:
            skipped += 1
            continue

        # 段級を順序値としてfloodgate形式のrate行に注入する
        comment = None
        if black_ord is not None and white_ord is not None:
            comment = ("'black_rate:wars:{:.1f}\n"
                       "'white_rate:wars:{:.1f}\n").format(float(black_ord), float(white_ord))

        exp = exporters[idx % shards]
        exp.info(init_board=g.sfen, names=g.names, version='V2.2', comment=comment)
        board = Board(sfen=g.sfen)
        for i, mv in enumerate(g.moves):
            t = g.times[i] if getattr(g, 'times', None) and i < len(g.times) else None
            exp.move(mv, time=t)
            board.push(mv)
        exp.endgame(endgame)
        converted += 1

    for exp in exporters:
        exp.close()

    print('converted={} skipped={} shards={}'.format(converted, skipped, shards))
    return converted, skipped


def main():
    """Parse arguments and run :func:`convert`."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('kif_dir', help='directory of Shogi Wars KIF files (searched recursively)')
    parser.add_argument('out_dir', help='output directory for CSA shards')
    parser.add_argument('--shards', type=int, default=8, help='number of output CSA files')
    args = parser.parse_args()
    convert(args.kif_dir, args.out_dir, args.shards)


if __name__ == '__main__':
    main()
