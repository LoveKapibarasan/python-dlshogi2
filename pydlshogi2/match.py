"""Play two USI engines against each other and score the result.

Training metrics say whether a model fits its data.  They do not say whether it
*plays better*, and neither does a change to the search, which touches no loss
at all.  The only instrument for that is a match, so this module is what closes
the loop: a branch is judged by putting its engine across the board from the
engine on ``main``.

The game loop, the USI protocol and the rule handling come from
:mod:`cshogi.cli`, which already does that job well.  What is added here is the
part that makes a match an *experiment*:

* **Structured results.** Every game appends a record to a JSONL metrics file,
  tagged with the experiment id and issue number it belongs to, so the
  dashboard can line a match up against the proposal that motivated it.
* **A stopping rule.** ``--sprt`` runs a sequential probability ratio test after
  every game and stops as soon as the result is conclusive in either direction.
  A regression is usually obvious within 30 games; paying for 100 to find that
  out is a waste of a GPU that is already shared with something else.
* **Honest error bars.** The summary reports the Elo difference with its 95 %
  interval, not just a win rate. See :mod:`pydlshogi2.rating`.

Both engines must be launchable as a single executable path — use the
``usi_engine.sh`` in each checkout.

Example
-------

.. code-block:: bash

    # a branch against main, one second per move, stopping early once decided
    python -m pydlshogi2.match \\
        --engine1 ../wt-branch/usi_engine.sh --name1 branch \\
        --engine2 ./usi_engine.sh            --name2 main \\
        --options1 modelfile=checkpoints/checkpoint.pth \\
        --options2 modelfile=checkpoints/checkpoint.pth \\
        --games 100 --byoyomi 1000 --opening openings.txt \\
        --sprt --elo1 20 \\
        --metrics metrics/match-EXP-001.jsonl --experiment EXP-001 --issue 7
"""
import argparse
import os
import sys

from pydlshogi2 import rating
from pydlshogi2.metrics import MetricsWriter

#: Result codes stored on each per-game record.
WIN, LOSS, DRAW = 'win', 'loss', 'draw'


def parse_options(text):
    """Parse a ``name=value,name=value`` engine option string into a dict.

    :param text: comma-separated assignments; empty or ``None`` gives ``{}``.
    :returns: mapping of USI option name to value (kept as a string).
    """
    options = {}
    if not text:
        return options
    for item in text.split(','):
        item = item.strip()
        if not item:
            continue
        name, separator, value = item.partition('=')
        if not separator:
            raise ValueError(
                'engine option {!r} is not in name=value form'.format(item))
        options[name.strip()] = value.strip()
    return options


def default_name(engine_path, options):
    """Derive a readable engine name from its launcher path and options.

    The name is what the rating table keys on, so it should identify the thing
    under test: usually the checkout it was launched from, plus the model when
    two engines share a checkout.

    :param engine_path: path to the engine launcher.
    :param options: parsed USI options for that engine.
    :returns: a short name.
    """
    checkout = os.path.basename(os.path.dirname(os.path.abspath(engine_path)))
    modelfile = options.get('modelfile')
    if not modelfile:
        return checkout
    model = os.path.splitext(os.path.basename(modelfile))[0]
    return '{}:{}'.format(checkout, model)


class MatchRecorder:
    """Turn ``cshogi.cli``'s running totals into per-game records and a verdict.

    ``cshogi.cli.main`` reports *cumulative* counts through its callback.  This
    class differences them to recover what each individual game did, appends it
    to the metrics file, and — when a sequential test is configured — decides
    whether there is any point playing on.

    :param metrics: a :class:`~pydlshogi2.metrics.MetricsWriter` (may be a
        no-op writer).
    :param sprt_config: ``None`` for a fixed-length match, otherwise a dict with
        ``elo0``, ``elo1``, ``alpha`` and ``beta``.
    :param games: the requested number of games, for progress output.
    :param paired: score colour-swapped pairs rather than individual games; see
        :class:`~pydlshogi2.rating.PairedMatchStats`.
    :param quiet: suppress the per-game progress line.
    """

    def __init__(self, metrics, sprt_config=None, games=None, paired=True,
                 quiet=False):
        self.metrics = metrics
        self.sprt_config = sprt_config
        self.games = games
        self.paired = paired
        self.quiet = quiet
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.results = []
        self.stopped_early = False
        self.sprt_result = None

    @property
    def stats(self):
        """Statistics for the games played so far.

        Paired matches are scored per colour-swapped pair, which is both more
        accurate and considerably more efficient — see
        :class:`~pydlshogi2.rating.PairedMatchStats`.
        """
        if self.paired:
            return rating.PairedMatchStats(self.results)
        return rating.MatchStats(self.wins, self.losses, self.draws)

    def _classify(self, wins, losses, draws):
        """Return the result of the game that just finished.

        :param wins: cumulative wins reported by the callback.
        :param losses: cumulative losses.
        :param draws: cumulative draws.
        :returns: one of :data:`WIN`, :data:`LOSS`, :data:`DRAW`, or ``None``
            when nothing changed (which should not happen, but a callback that
            fires twice must not corrupt the counts).
        """
        if wins > self.wins:
            return WIN
        if losses > self.losses:
            return LOSS
        if draws > self.draws:
            return DRAW
        return None

    def __call__(self, status):
        """Callback invoked by ``cshogi.cli.main`` after every game.

        :param status: cumulative counts from the arena.
        :returns: ``False`` to stop the match, ``True`` to continue.
        """
        wins = status['engine1_won']
        losses = status['engine2_won']
        draws = status['draw']
        result = self._classify(wins, losses, draws)
        self.wins, self.losses, self.draws = wins, losses, draws
        if result is not None:
            self.results.append(result)

        stats = self.stats
        record = {
            'scope': 'game',
            'game': len(self.results),
            'result': result,
            'player_a': status['engine1_name'],
            'player_b': status['engine2_name'],
            'wins': wins,
            'losses': losses,
            'draws': draws,
            'score': stats.score,
            'elo': stats.elo,
            'error_margin': stats.error_margin,
            'los': stats.los,
            'black_won': status['black_won'],
            'white_won': status['white_won'],
        }
        if self.paired:
            record['pairs'] = stats.pairs
            record['pentanomial'] = stats.pentanomial

        keep_going = True
        if self.sprt_config is not None:
            self.sprt_result = rating.sprt(stats, **self.sprt_config)
            record['llr'] = self.sprt_result.llr
            record['sprt_decision'] = self.sprt_result.decision
            if self.sprt_result.finished:
                self.stopped_early = True
                keep_going = False

        self.metrics.metric(**record)

        if not self.quiet:
            line = '[{}/{}] {} {}-{}-{} score={:.3f} elo={:+.1f}±{:.1f} los={:.1f}%'.format(
                len(self.results), self.games if self.games else '?', result or '-',
                wins, losses, draws, stats.score, stats.elo,
                stats.error_margin, stats.los)
            if self.sprt_result is not None:
                line += ' llr={:+.2f} [{:.2f}, {:.2f}] {}'.format(
                    self.sprt_result.llr, self.sprt_result.lower,
                    self.sprt_result.upper, self.sprt_result.decision)
            print(line, flush=True)

        return keep_going


def format_summary(name_a, name_b, stats, sprt_result=None, stopped_early=False):
    """Render the end-of-match report as a block of text.

    :param name_a: name of the first engine.
    :param name_b: name of the second engine.
    :param stats: final :class:`~pydlshogi2.rating.MatchStats`.
    :param sprt_result: the last :class:`~pydlshogi2.rating.SprtResult`, if any.
    :param stopped_early: whether the sequential test ended the match.
    :returns: a multi-line string.
    """
    low, high = stats.elo_interval(0.95)
    lines = [
        '',
        '=== match result: {} vs {} ==='.format(name_a, name_b),
        'games      : {}{}'.format(stats.games,
                                   ' (stopped early by SPRT)' if stopped_early else ''),
        'W-L-D      : {}-{}-{}'.format(stats.wins, stats.losses, stats.draws),
        'score      : {:.4f}  (draw ratio {:.1%})'.format(stats.score, stats.draw_ratio),
    ]
    if isinstance(stats, rating.PairedMatchStats):
        lines.append('pairs      : {} scored as [0, 1/2, 1, 1 1/2, 2] = {}'.format(
            stats.pairs, stats.pentanomial))
    lines += [
        'Elo        : {:+.1f} +/- {:.1f}   95% CI [{:+.1f}, {:+.1f}]'.format(
            stats.elo, stats.error_margin, low, high),
        'LOS        : {:.1f}%'.format(stats.los),
    ]
    if sprt_result is not None:
        lines.append('SPRT       : llr={:+.3f} bounds=[{:.3f}, {:.3f}] '
                     'H0={:.0f} H1={:.0f} -> {}'.format(
                         sprt_result.llr, sprt_result.lower, sprt_result.upper,
                         sprt_result.elo0, sprt_result.elo1, sprt_result.decision))
        verdict = {
            'accept': 'engine1 is stronger; adopt the change',
            'reject': 'no improvement worth adopting',
            'continue': 'inconclusive; play more games',
        }[sprt_result.decision]
        lines.append('verdict    : {}'.format(verdict))
    elif low > 0.0:
        lines.append('verdict    : engine1 is stronger (95% interval excludes 0)')
    elif high < 0.0:
        lines.append('verdict    : engine1 is weaker (95% interval excludes 0)')
    else:
        lines.append('verdict    : inconclusive (95% interval contains 0)')
    lines.append('')
    return '\n'.join(lines)


def build_parser():
    """Return the command-line parser for ``python -m pydlshogi2.match``."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--engine1', required=True,
                        help='launcher of the engine under test (e.g. a branch '
                             'checkout\'s usi_engine.sh)')
    parser.add_argument('--engine2', required=True,
                        help='launcher of the baseline engine')
    parser.add_argument('--name1', default=None,
                        help='name recorded for engine1 (default: derived from '
                             'its checkout and model)')
    parser.add_argument('--name2', default=None, help='name recorded for engine2')
    parser.add_argument('--options1', default='',
                        help='USI options for engine1 as name=value,name=value')
    parser.add_argument('--options2', default='', help='USI options for engine2')
    parser.add_argument('--games', type=int, default=100,
                        help='maximum number of games (colours alternate every game)')

    time_control = parser.add_argument_group(
        'time control',
        'Pick one. Fixed playouts compare two *models* independently of speed; '
        'fixed time is the only way a search speed-up can show up as strength.')
    time_control.add_argument('--byoyomi', type=int, default=None,
                              help='milliseconds per move')
    time_control.add_argument('--time', type=int, default=None,
                              help='total milliseconds per side')
    time_control.add_argument('--inc', type=int, default=None,
                              help='increment in milliseconds per move')
    time_control.add_argument('--playouts', type=int, default=None,
                              help='fixed playouts per move; sets the engines\' '
                                   '`playouts` USI option and sends a bare `go`')

    parser.add_argument('--opening', default=None,
                        help='opening book (one "startpos moves ..." line per '
                             'position). Without it every game is identical, '
                             'because the search is deterministic — see '
                             'utils/make_opening_book.py')
    parser.add_argument('--opening-moves', type=int, default=24,
                        help='maximum plies replayed from each opening line')
    parser.add_argument('--opening-seed', type=int, default=None,
                        help='seed for shuffling the opening book')
    parser.add_argument('--draw', type=int, default=256,
                        help='declare a draw after this many plies')
    parser.add_argument('--resign', type=int, default=None,
                        help='adjudicate a loss when an engine reports a score '
                             'below -N centipawns')
    parser.add_argument('--mate-win', action='store_true',
                        help='end the game as soon as an engine reports a mate score')

    parser.add_argument('--unpaired', action='store_true',
                        help='score each game independently instead of scoring '
                             'colour-swapped pairs; only correct when the games '
                             'within a pair are genuinely independent')

    sprt_group = parser.add_argument_group('sequential test')
    sprt_group.add_argument('--sprt', action='store_true',
                            help='stop as soon as the result is conclusive')
    sprt_group.add_argument('--elo0', type=float, default=rating.DEFAULT_ELO0,
                            help='null hypothesis Elo (default: %(default)s)')
    sprt_group.add_argument('--elo1', type=float, default=rating.DEFAULT_ELO1,
                            help='alternative hypothesis Elo (default: %(default)s)')
    sprt_group.add_argument('--alpha', type=float, default=rating.DEFAULT_ALPHA,
                            help='type-I error rate (default: %(default)s)')
    sprt_group.add_argument('--beta', type=float, default=rating.DEFAULT_BETA,
                            help='type-II error rate (default: %(default)s)')

    record_group = parser.add_argument_group('recording')
    record_group.add_argument('--metrics', default=None,
                              help='JSONL file to append the match to; read by '
                                   'dashboard/app.py')
    record_group.add_argument('--experiment', default=None,
                              help='experiment id from wiki/Improvement-Backlog.md '
                                   '(e.g. EXP-001), used to join the match to the '
                                   'proposal it tests')
    record_group.add_argument('--issue', type=int, default=None,
                              help='GitHub issue number tracking this experiment')
    record_group.add_argument('--note', default=None,
                              help='free-text note stored on the run record')
    record_group.add_argument('--csa', default=None,
                              help='directory to write the games to as CSA')
    record_group.add_argument('--quiet', action='store_true',
                              help='only print the final summary')
    record_group.add_argument('--debug', action='store_true',
                              help='echo the USI conversation')
    return parser


def run_match(args):
    """Play the match described by ``args`` and return its statistics.

    :param args: parsed arguments from :func:`build_parser`.
    :returns: a ``(stats, sprt_result, stopped_early, names)`` tuple.
    """
    # cshogi は対局と USI プロトコルを既に持っているので、そこは再実装しない
    from cshogi import cli

    options1 = parse_options(args.options1)
    options2 = parse_options(args.options2)

    if args.playouts is not None:
        # 時間制御なしの `go` は固定プレイアウト探索になる (mcts_player の playouts オプション)
        options1.setdefault('playouts', str(args.playouts))
        options2.setdefault('playouts', str(args.playouts))

    name1 = args.name1 or default_name(args.engine1, options1)
    name2 = args.name2 or default_name(args.engine2, options2)

    sprt_config = None
    if args.sprt:
        sprt_config = {'elo0': args.elo0, 'elo1': args.elo1,
                       'alpha': args.alpha, 'beta': args.beta}

    metrics = MetricsWriter(
        args.metrics, kind='match', args=vars(args),
        extra={'experiment': args.experiment, 'issue': args.issue,
               'note': args.note, 'player_a': name1, 'player_b': name2})

    recorder = MatchRecorder(metrics, sprt_config=sprt_config,
                             games=args.games, paired=not args.unpaired,
                             quiet=args.quiet)

    try:
        cli.main(
            args.engine1, args.engine2,
            options1=options1, options2=options2,
            names=[name1, name2],
            games=args.games,
            byoyomi=args.byoyomi, time=args.time, inc=args.inc,
            draw=args.draw, resign=args.resign, mate_win=args.mate_win,
            opening=args.opening, opening_moves=args.opening_moves,
            opening_seed=args.opening_seed,
            keep_process=True,
            csa=args.csa, multi_csa=bool(args.csa),
            print_summary=False, debug=args.debug,
            callback=recorder)
    except Exception:
        metrics.close(status='failed', **recorder.stats.summary())
        raise

    stats = recorder.stats
    summary = {'scope': 'summary', 'player_a': name1, 'player_b': name2,
               'stopped_early': recorder.stopped_early}
    summary.update(stats.summary())
    if recorder.sprt_result is not None:
        summary['sprt'] = recorder.sprt_result.summary()
        summary['sprt_decision'] = recorder.sprt_result.decision
        summary['llr'] = recorder.sprt_result.llr
    metrics.metric(**summary)
    metrics.close(status='completed', games=stats.games)

    return stats, recorder.sprt_result, recorder.stopped_early, (name1, name2)


def main():
    """Entry point: play a match and print the verdict."""
    args = build_parser().parse_args()

    if args.playouts is None and args.byoyomi is None and args.time is None:
        print('warning: no time control given; the engines will use their own '
              'default playout count. Pass --playouts or --byoyomi to make the '
              'match reproducible.', file=sys.stderr)
    if not args.opening:
        print('warning: no --opening book; MCTS is deterministic, so every game '
              'may be a replay of the first one. Generate one with '
              'utils/make_opening_book.py.', file=sys.stderr)

    stats, sprt_result, stopped_early, (name1, name2) = run_match(args)
    print(format_summary(name1, name2, stats, sprt_result, stopped_early))

    # SPRT で「改善なし」と判定された場合のみ非ゼロ終了 (CIやスクリプトから使える)
    if sprt_result is not None and sprt_result.decision == 'reject':
        sys.exit(1)


if __name__ == '__main__':
    main()
