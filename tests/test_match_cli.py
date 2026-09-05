"""Tests for the match runner's own logic in :mod:`pydlshogi2.match`.

The module imports ``cshogi`` lazily — only once a match actually starts — so
the argument handling, the naming and the opening-book check are all testable
with the standard library alone.

Run with::

    python -m unittest discover -s tests
"""
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pydlshogi2 import match, rating  # noqa: E402


class ParseOptionsTest(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(match.parse_options(''), {})
        self.assertEqual(match.parse_options(None), {})

    def test_pairs(self):
        self.assertEqual(
            match.parse_options('modelfile=a.pth,gpu_id=0'),
            {'modelfile': 'a.pth', 'gpu_id': '0'})

    def test_whitespace_and_trailing_comma(self):
        self.assertEqual(match.parse_options(' a = 1 , b=2 , '),
                         {'a': '1', 'b': '2'})

    def test_value_may_contain_a_path(self):
        self.assertEqual(match.parse_options('modelfile=/tmp/a/b.pth'),
                         {'modelfile': '/tmp/a/b.pth'})

    def test_rejects_a_bare_name(self):
        with self.assertRaises(ValueError):
            match.parse_options('modelfile')


class DefaultNameTest(unittest.TestCase):
    def test_uses_the_checkout_directory(self):
        self.assertEqual(match.default_name('/x/wt-main/usi_engine.sh', {}),
                         'wt-main')

    def test_appends_the_model(self):
        self.assertEqual(
            match.default_name('/x/branch/usi_engine.sh',
                               {'modelfile': 'checkpoints/checkpoint-003.pth'}),
            'branch:checkpoint-003')


class OpeningSupplyTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.directory, ignore_errors=True)

    def book(self, lines):
        path = os.path.join(self.directory, 'openings.txt')
        with open(path, 'w', encoding='utf-8') as f:
            for i in range(lines):
                f.write('startpos moves 7g7f 3c3d {}\n'.format(i))
        return path

    def test_counts_lines(self):
        self.assertEqual(match.count_openings(self.book(7)), 7)
        self.assertEqual(match.count_openings(None), 0)
        self.assertEqual(match.count_openings('/nope/openings.txt'), 0)

    def test_enough_openings_is_silent(self):
        self.assertIsNone(match.check_opening_supply(32, self.book(16)))
        self.assertIsNone(match.check_opening_supply(10, self.book(16)))

    def test_too_few_openings_warns(self):
        # 16定跡は32局までしか賄えない。それ以降は同じ将棋の焼き直しになる
        warning = match.check_opening_supply(100, self.book(16))
        self.assertIn('33+', warning)
        self.assertIn('50', warning)

    def test_missing_book_warns(self):
        self.assertIn('deterministic', match.check_opening_supply(10, None))

    def test_empty_book_warns(self):
        path = os.path.join(self.directory, 'empty.txt')
        open(path, 'w').close()
        self.assertIn('empty', match.check_opening_supply(10, path))


class MatchRecorderTest(unittest.TestCase):
    class NullMetrics:
        def __init__(self):
            self.records = []

        def metric(self, **fields):
            self.records.append(fields)

    def status(self, wins, losses, draws):
        return {'engine1_won': wins, 'engine2_won': losses, 'draw': draws,
                'engine1_name': 'branch', 'engine2_name': 'main',
                'black_won': 0, 'white_won': 0, 'total': wins + losses + draws}

    def test_derives_each_game_from_cumulative_counts(self):
        metrics = self.NullMetrics()
        recorder = match.MatchRecorder(metrics, games=4, quiet=True)
        recorder(self.status(1, 0, 0))
        recorder(self.status(1, 1, 0))
        recorder(self.status(1, 1, 1))
        self.assertEqual(recorder.results, ['win', 'loss', 'draw'])
        self.assertEqual([r['result'] for r in metrics.records],
                         ['win', 'loss', 'draw'])

    def test_paired_by_default(self):
        recorder = match.MatchRecorder(self.NullMetrics(), games=2, quiet=True)
        recorder(self.status(1, 0, 0))
        recorder(self.status(1, 1, 0))
        self.assertIsInstance(recorder.stats, rating.PairedMatchStats)
        self.assertEqual(recorder.stats.pairs, 1)
        self.assertAlmostEqual(recorder.stats.score, 0.5)

    def test_unpaired_on_request(self):
        recorder = match.MatchRecorder(self.NullMetrics(), games=2, paired=False,
                                       quiet=True)
        recorder(self.status(1, 0, 0))
        self.assertNotIsInstance(recorder.stats, rating.PairedMatchStats)
        self.assertEqual(recorder.stats.wins, 1)

    def test_sprt_stops_the_match(self):
        recorder = match.MatchRecorder(
            self.NullMetrics(),
            sprt_config={'elo0': 0.0, 'elo1': 20.0, 'alpha': 0.05, 'beta': 0.05},
            games=200, quiet=True)
        wins = draws = 0
        for game in range(200):
            if game % 2 == 0:
                wins += 1
            else:
                draws += 1
            if not recorder(self.status(wins, 0, draws)):
                break
        self.assertTrue(recorder.stopped_early)
        self.assertEqual(recorder.sprt_result.decision, 'accept')
        # 200局まで行かずに打ち切れていること
        self.assertLess(len(recorder.results), 200)

    def test_records_how_long_each_game_took(self):
        recorder = match.MatchRecorder(self.NullMetrics(), games=4, quiet=True)
        recorder(self.status(1, 0, 0))
        recorder(self.status(1, 1, 0))
        self.assertEqual(len(recorder.seconds), 2)
        for seconds in recorder.seconds:
            self.assertGreaterEqual(seconds, 0.0)

    def test_estimates_the_remaining_time(self):
        metrics = self.NullMetrics()
        recorder = match.MatchRecorder(metrics, games=10, quiet=True)
        recorder(self.status(1, 0, 0))
        record = metrics.records[-1]
        self.assertIn('seconds', record)
        # 10局中1局終わったので、残り9局分の見積もりが出る
        self.assertAlmostEqual(record['eta_seconds'], record['seconds'] * 9,
                               delta=record['seconds'] * 9 + 1e-6)

    def test_no_sprt_means_no_early_stop(self):
        recorder = match.MatchRecorder(self.NullMetrics(), games=4, quiet=True)
        self.assertTrue(recorder(self.status(1, 0, 0)))
        self.assertFalse(recorder.stopped_early)
        self.assertIsNone(recorder.sprt_result)


class FormatSummaryTest(unittest.TestCase):
    def test_reports_an_inconclusive_match_as_inconclusive(self):
        text = match.format_summary(
            'branch', 'main', rating.MatchStats(55, 45, 0))
        self.assertIn('inconclusive', text)

    def test_reports_a_clear_win(self):
        text = match.format_summary(
            'branch', 'main', rating.MatchStats(700, 300, 0))
        self.assertIn('stronger', text)

    def test_includes_the_pentanomial_for_a_paired_match(self):
        stats = rating.PairedMatchStats(['win', 'draw'] * 10)
        text = match.format_summary('branch', 'main', stats)
        self.assertIn('pairs', text)
        self.assertIn('[0, 0, 0, 10, 0]', text)

    def test_timing_line_appears_when_durations_are_known(self):
        text = match.format_summary('branch', 'main', rating.MatchStats(5, 5),
                                    seconds=[30.0, 40.0, 50.0])
        self.assertIn('time', text)
        self.assertIn('40 s per game', text)

    def test_sprt_verdict_is_spelled_out(self):
        stats = rating.PairedMatchStats(['win', 'draw'] * 20)
        text = match.format_summary('branch', 'main', stats,
                                    rating.sprt(stats), stopped_early=True)
        self.assertIn('stopped early by SPRT', text)
        self.assertIn('adopt the change', text)


class ParserTest(unittest.TestCase):
    def test_minimal_invocation(self):
        args = match.build_parser().parse_args(
            ['--engine1', 'a.sh', '--engine2', 'b.sh'])
        self.assertEqual(args.games, 100)
        self.assertFalse(args.sprt)
        self.assertFalse(args.unpaired)

    def test_sprt_defaults_match_the_rating_module(self):
        args = match.build_parser().parse_args(
            ['--engine1', 'a.sh', '--engine2', 'b.sh', '--sprt'])
        self.assertEqual(args.elo0, rating.DEFAULT_ELO0)
        self.assertEqual(args.elo1, rating.DEFAULT_ELO1)


if __name__ == '__main__':
    unittest.main()
