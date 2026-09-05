"""Tests for the match arithmetic in :mod:`pydlshogi2.rating`.

The module is standard library only by design, so these run anywhere — no
torch, no cshogi, no GPU.

Run with::

    python -m unittest discover -s tests
"""
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pydlshogi2 import rating  # noqa: E402


class EloConversionTest(unittest.TestCase):
    def test_even_score_is_zero_elo(self):
        self.assertAlmostEqual(rating.elo_from_score(0.5), 0.0)
        self.assertAlmostEqual(rating.score_from_elo(0.0), 0.5)

    def test_round_trip(self):
        for elo in (-400.0, -50.0, 0.0, 17.5, 200.0):
            self.assertAlmostEqual(
                rating.elo_from_score(rating.score_from_elo(elo)), elo, places=6)

    def test_known_value(self):
        # 76 % のスコアはおよそ +200 Elo (Eloの定義そのもの)
        self.assertAlmostEqual(rating.score_from_elo(200.0), 0.7597469, places=6)

    def test_extremes_are_finite(self):
        self.assertTrue(math.isfinite(rating.elo_from_score(0.0)))
        self.assertTrue(math.isfinite(rating.elo_from_score(1.0)))
        self.assertGreater(rating.elo_from_score(1.0), 0.0)
        self.assertLess(rating.elo_from_score(0.0), 0.0)


class PhiInvTest(unittest.TestCase):
    def test_inverts_phi(self):
        for p in (0.001, 0.025, 0.1, 0.5, 0.9, 0.975, 0.999):
            self.assertAlmostEqual(rating.phi(rating.phi_inv(p)), p, places=10)

    def test_known_quantile(self):
        self.assertAlmostEqual(rating.phi_inv(0.975), 1.9599639845, places=8)

    def test_rejects_out_of_range(self):
        with self.assertRaises(ValueError):
            rating.phi_inv(0.0)
        with self.assertRaises(ValueError):
            rating.phi_inv(1.0)


class MatchStatsTest(unittest.TestCase):
    def test_score_counts_draws_as_half(self):
        stats = rating.MatchStats(wins=40, losses=40, draws=20)
        self.assertEqual(stats.games, 100)
        self.assertAlmostEqual(stats.score, 0.5)
        self.assertAlmostEqual(stats.elo, 0.0)
        self.assertAlmostEqual(stats.draw_ratio, 0.2)

    def test_error_margin_shrinks_with_more_games(self):
        small = rating.MatchStats(wins=55, losses=45)
        large = rating.MatchStats(wins=550, losses=450)
        self.assertAlmostEqual(small.score, large.score)
        self.assertLess(large.error_margin, small.error_margin)
        # 10倍の対局数で誤差はおよそ 1/sqrt(10)
        self.assertAlmostEqual(large.error_margin / small.error_margin,
                               1.0 / math.sqrt(10.0), places=1)

    def test_hundred_games_cannot_resolve_small_gains(self):
        # 100局で55%勝ったところで、95%区間は0をまたぐ
        stats = rating.MatchStats(wins=55, losses=45)
        low, high = stats.elo_interval(0.95)
        self.assertLess(low, 0.0)
        self.assertGreater(high, 0.0)

    def test_draws_reduce_variance(self):
        decisive = rating.MatchStats(wins=50, losses=50)
        drawish = rating.MatchStats(wins=25, losses=25, draws=50)
        self.assertAlmostEqual(decisive.score, drawish.score)
        self.assertLess(drawish.error_margin, decisive.error_margin)

    def test_los_is_symmetric(self):
        self.assertAlmostEqual(rating.MatchStats(50, 50).los, 50.0)
        self.assertAlmostEqual(
            rating.MatchStats(60, 40).los + rating.MatchStats(40, 60).los, 100.0)

    def test_los_ignores_draws(self):
        self.assertAlmostEqual(rating.MatchStats(60, 40).los,
                               rating.MatchStats(60, 40, draws=99).los)

    def test_no_games_is_neutral(self):
        stats = rating.MatchStats(0, 0, 0)
        self.assertEqual(stats.games, 0)
        self.assertAlmostEqual(stats.score, 0.5)
        self.assertAlmostEqual(stats.los, 50.0)

    def test_negative_counts_rejected(self):
        with self.assertRaises(ValueError):
            rating.MatchStats(-1, 0, 0)

    def test_summary_is_jsonable(self):
        summary = rating.MatchStats(57, 38, 5).summary()
        self.assertEqual(summary['games'], 100)
        for value in summary.values():
            self.assertIsInstance(value, (int, float))


class SprtTest(unittest.TestCase):
    def test_bounds(self):
        lower, upper = rating.sprt_bounds(0.05, 0.05)
        self.assertAlmostEqual(upper, math.log(0.95 / 0.05))
        self.assertAlmostEqual(lower, -upper)

    def test_strong_improvement_is_accepted(self):
        result = rating.sprt(rating.MatchStats(wins=180, losses=100, draws=20),
                             elo0=0.0, elo1=20.0)
        self.assertEqual(result.decision, 'accept')
        self.assertTrue(result.finished)

    def test_clear_regression_is_rejected(self):
        result = rating.sprt(rating.MatchStats(wins=100, losses=180, draws=20),
                             elo0=0.0, elo1=20.0)
        self.assertEqual(result.decision, 'reject')

    def test_small_sample_is_inconclusive(self):
        result = rating.sprt(rating.MatchStats(wins=6, losses=4),
                             elo0=0.0, elo1=20.0)
        self.assertEqual(result.decision, 'continue')
        self.assertFalse(result.finished)

    def test_zero_variance_does_not_divide_by_zero(self):
        # 全勝・全引き分けはどちらも分散0。正規近似は使えないので保留になる
        for stats in (rating.MatchStats(10, 0, 0), rating.MatchStats(0, 0, 10)):
            self.assertEqual(rating.sprt(stats).decision, 'continue')

    def test_llr_grows_with_evidence(self):
        small = rating.sprt(rating.MatchStats(60, 40)).llr
        large = rating.sprt(rating.MatchStats(600, 400)).llr
        self.assertGreater(large, small)


class BradleyTerryTest(unittest.TestCase):
    def test_empty_input(self):
        self.assertEqual(rating.bradley_terry_ratings([]), [])

    def test_anchor_is_zero(self):
        rows = rating.bradley_terry_ratings(
            [{'player_a': 'new', 'player_b': 'base',
              'wins': 60, 'losses': 40, 'draws': 0}],
            anchor='base')
        by_player = {row['player']: row for row in rows}
        self.assertAlmostEqual(by_player['base']['elo'], 0.0)
        self.assertGreater(by_player['new']['elo'], 0.0)
        self.assertTrue(by_player['base']['is_anchor'])

    def test_recovers_a_transitive_chain(self):
        # a > b > c を、a と c は直接対戦させずに推定する
        matches = [
            {'player_a': 'a', 'player_b': 'b', 'wins': 300, 'losses': 200, 'draws': 0},
            {'player_a': 'b', 'player_b': 'c', 'wins': 300, 'losses': 200, 'draws': 0},
        ]
        rows = rating.bradley_terry_ratings(matches, anchor='c')
        elo = {row['player']: row['elo'] for row in rows}
        self.assertGreater(elo['a'], elo['b'])
        self.assertGreater(elo['b'], elo['c'])
        # 2区間分の差がおよそ足し合わさる
        self.assertAlmostEqual(elo['a'] - elo['b'], elo['b'] - elo['c'], delta=8.0)

    def test_equal_engines_get_equal_ratings(self):
        rows = rating.bradley_terry_ratings(
            [{'player_a': 'x', 'player_b': 'y', 'wins': 50, 'losses': 50, 'draws': 100}])
        self.assertAlmostEqual(rows[0]['elo'], rows[1]['elo'], places=6)

    def test_undefeated_engine_stays_finite(self):
        rows = rating.bradley_terry_ratings(
            [{'player_a': 'perfect', 'player_b': 'base',
              'wins': 30, 'losses': 0, 'draws': 0}], anchor='base')
        for row in rows:
            self.assertTrue(math.isfinite(row['elo']))

    def test_sorted_strongest_first(self):
        matches = [
            {'player_a': 'a', 'player_b': 'b', 'wins': 70, 'losses': 30, 'draws': 0},
            {'player_a': 'b', 'player_b': 'c', 'wins': 70, 'losses': 30, 'draws': 0},
        ]
        rows = rating.bradley_terry_ratings(matches)
        self.assertEqual([row['player'] for row in rows], ['a', 'b', 'c'])

    def test_counts_are_aggregated_per_player(self):
        matches = [
            {'player_a': 'a', 'player_b': 'b', 'wins': 6, 'losses': 3, 'draws': 1},
            {'player_a': 'a', 'player_b': 'c', 'wins': 5, 'losses': 5, 'draws': 0},
        ]
        rows = {row['player']: row for row in rating.bradley_terry_ratings(matches)}
        self.assertEqual(rows['a']['games'], 20)
        self.assertEqual(rows['a']['wins'], 11)
        self.assertEqual(rows['a']['losses'], 8)
        self.assertEqual(rows['a']['draws'], 1)

    def test_self_play_and_empty_matches_ignored(self):
        matches = [
            {'player_a': 'a', 'player_b': 'a', 'wins': 5, 'losses': 5, 'draws': 0},
            {'player_a': 'a', 'player_b': 'b', 'wins': 0, 'losses': 0, 'draws': 0},
            {'player_a': 'a', 'player_b': 'b', 'wins': 5, 'losses': 5, 'draws': 0},
        ]
        rows = rating.bradley_terry_ratings(matches)
        self.assertEqual({row['player'] for row in rows}, {'a', 'b'})
        self.assertEqual(rows[0]['games'], 10)


class VarianceRegularisationTest(unittest.TestCase):
    def test_raw_variance_matches_the_textbook_formula(self):
        stats = rating.MatchStats(wins=60, losses=40)
        # 勝率0.6のベルヌーイ分布の分散は 0.6*0.4 = 0.24
        self.assertAlmostEqual(stats.raw_variance, 0.24, places=10)

    def test_empty_buckets_do_not_give_zero_variance(self):
        # 全勝は経験分散0だが、10局で確信できるはずがない
        stats = rating.MatchStats(wins=10, losses=0, draws=0)
        self.assertAlmostEqual(stats.raw_variance, 0.0)
        self.assertGreater(stats.variance, 0.0)
        self.assertGreater(stats.error_margin, 0.0)

    def test_prior_washes_out_as_the_match_grows(self):
        small = rating.MatchStats(wins=6, losses=4)
        large = rating.MatchStats(wins=6000, losses=4000)
        self.assertGreater(abs(small.variance - small.raw_variance),
                           abs(large.variance - large.raw_variance))
        self.assertAlmostEqual(large.variance, large.raw_variance, places=4)


class PairedMatchStatsTest(unittest.TestCase):
    def test_pairs_are_consecutive_games(self):
        stats = rating.PairedMatchStats(['win', 'loss', 'draw', 'draw'])
        self.assertEqual(stats.pairs, 2)
        self.assertEqual(stats.games, 4)
        self.assertEqual(stats.pair_scores, [1.0, 1.0])
        self.assertEqual(stats.pentanomial, [0, 0, 2, 0, 0])

    def test_counts_still_available(self):
        stats = rating.PairedMatchStats(['win', 'win', 'loss', 'draw'])
        self.assertEqual((stats.wins, stats.losses, stats.draws), (2, 1, 1))
        self.assertAlmostEqual(stats.score, 0.625)

    def test_trailing_game_is_excluded(self):
        stats = rating.PairedMatchStats(['win', 'loss', 'win'])
        self.assertEqual(stats.pairs, 1)
        self.assertEqual(stats.unpaired, 1)
        self.assertEqual(stats.games, 2)

    def test_identical_engines_have_no_paired_variance(self):
        # 同じ開始局面を先後入れ替えて指せば、互角のエンジンは必ず1-1になる
        stats = rating.PairedMatchStats(['win', 'loss'] * 25)
        self.assertAlmostEqual(stats.score, 0.5)
        self.assertAlmostEqual(stats.raw_variance, 0.0)

    def test_pairing_gives_tighter_bars_than_counting_games(self):
        results = ['win', 'draw'] * 25
        paired = rating.PairedMatchStats(results)
        unpaired = rating.MatchStats(wins=25, losses=0, draws=25)
        self.assertAlmostEqual(paired.score, unpaired.score)
        self.assertLess(paired.error_margin, unpaired.error_margin)

    def test_consistent_pairs_reach_a_verdict(self):
        # 全ペアで勝ち越していれば、経験分散が0でも結論を出せなければならない
        stats = rating.PairedMatchStats(['win', 'draw'] * 20)
        self.assertEqual(rating.sprt(stats, elo0=0.0, elo1=20.0).decision, 'accept')

    def test_unknown_labels_ignored(self):
        stats = rating.PairedMatchStats(['win', None, 'loss', 'draw', 'draw'])
        self.assertEqual(stats.games, 4)
        self.assertEqual(stats.pairs, 2)

    def test_empty(self):
        stats = rating.PairedMatchStats([])
        self.assertEqual(stats.pairs, 0)
        self.assertAlmostEqual(stats.score, 0.5)
        self.assertEqual(rating.sprt(stats).decision, 'continue')

    def test_summary_reports_the_pentanomial(self):
        summary = rating.PairedMatchStats(['win', 'loss', 'win', 'win']).summary()
        self.assertTrue(summary['paired'])
        self.assertEqual(summary['pairs'], 2)
        self.assertEqual(summary['pentanomial'], [0, 0, 1, 0, 1])


if __name__ == '__main__':
    unittest.main()
