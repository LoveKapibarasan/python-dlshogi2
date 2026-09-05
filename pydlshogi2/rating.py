"""Rating arithmetic for engine-versus-engine matches.

A win rate on its own does not say whether a change helped: 55 % over 100 games
is well inside the noise of two identical engines.  This module turns match
counts into the numbers that *do* answer the question — an Elo difference with
a confidence interval, a likelihood of superiority, and a sequential test that
says "accept", "reject" or "keep playing".

It is deliberately **standard library only**, so the dashboard, the tests and
throwaway scripts can import it without torch or cshogi.  :mod:`pydlshogi2.match`
plays the games; everything here only counts them.

Two levels are provided:

* :class:`MatchStats` — one pairing, e.g. *new branch vs. main*.
* :func:`bradley_terry_ratings` — many pairings at once, collapsed into a single
  rating per engine so a whole checkpoint history can be put on one scale.

Example
-------

.. code-block:: python

    stats = MatchStats(wins=57, losses=38, draws=5)
    print(stats.elo, stats.error_margin, stats.los)
    print(sprt(stats, elo0=0, elo1=20).decision)
"""
import math

#: Default null hypothesis for :func:`sprt` — "the change is not an improvement".
DEFAULT_ELO0 = 0.0
#: Default alternative hypothesis for :func:`sprt` — "the change is worth this much".
DEFAULT_ELO1 = 20.0
#: Default type-I error rate (probability of accepting a change that does nothing).
DEFAULT_ALPHA = 0.05
#: Default type-II error rate (probability of rejecting a change worth ``elo1``).
DEFAULT_BETA = 0.05

#: Scores this close to 0 or 1 are clamped before taking a logit, so a clean
#: sweep reports a large finite Elo instead of raising.
_SCORE_EPS = 1e-9

#: Pseudo-count added to every outcome bucket before estimating the variance.
#:
#: A sample can easily contain no draws at all, or — with colour-swapped pairs
#: — every pair ending the same way.  The *empirical* variance is then exactly
#: zero, which would claim perfect certainty from a handful of games and make
#: the normal approximation behind :func:`sprt` divide by zero.  Half a
#: pseudo-observation per bucket (a symmetric Dirichlet(½) prior) keeps the
#: estimate positive and its influence vanishes as the match grows.
VARIANCE_PRIOR = 0.5


def elo_from_score(score):
    """Convert an expected score in ``(0, 1)`` to an Elo difference.

    :param score: expected score (win rate with draws counted as a half).
    :returns: Elo difference; ``±inf`` is avoided by clamping the score.
    """
    score = min(max(score, _SCORE_EPS), 1.0 - _SCORE_EPS)
    return -400.0 * math.log10(1.0 / score - 1.0)


def score_from_elo(elo):
    """Convert an Elo difference to the expected score it predicts.

    :param elo: Elo difference.
    :returns: expected score in ``(0, 1)``.
    """
    return 1.0 / (1.0 + 10.0 ** (-elo / 400.0))


def phi(x):
    """Standard normal cumulative distribution function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def phi_inv(p):
    """Inverse of :func:`phi` (the standard normal quantile function).

    Uses Acklam's rational approximation refined by one Halley step, which is
    accurate to roughly machine precision over the range this module needs.

    :param p: probability in ``(0, 1)``.
    """
    if not 0.0 < p < 1.0:
        raise ValueError('p must be in (0, 1), got {!r}'.format(p))

    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]

    p_low, p_high = 0.02425, 1.0 - 0.02425
    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        x = ((((( c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
            (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    else:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        x = -((((( c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
             ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)

    # 1回のHalley法で精度を上げる
    e = phi(x) - p
    u = e * math.sqrt(2.0 * math.pi) * math.exp(x * x / 2.0)
    return x - u / (1.0 + x * u / 2.0)


class MatchStats:
    """Counts of one pairing, plus everything derivable from them.

    All values are from the **first** engine's point of view: ``wins`` are games
    it won, ``losses`` games it lost.

    :param wins: games won by engine 1.
    :param losses: games won by engine 2.
    :param draws: drawn games (sennichite, jishogi, move limit).
    """

    def __init__(self, wins, losses, draws=0):
        if min(wins, losses, draws) < 0:
            raise ValueError('game counts must not be negative')
        self.wins = int(wins)
        self.losses = int(losses)
        self.draws = int(draws)

    @property
    def games(self):
        """Total games played."""
        return self.wins + self.losses + self.draws

    @property
    def score(self):
        """Expected score in ``[0, 1]``, draws counted as a half."""
        if self.games == 0:
            return 0.5
        return (self.wins + 0.5 * self.draws) / self.games

    @property
    def draw_ratio(self):
        """Fraction of games that were drawn."""
        return self.draws / self.games if self.games else 0.0

    @property
    def outcome_counts(self):
        """Outcome buckets as ``(score, count)`` pairs.

        The trinomial win / draw / loss split.  :class:`PairedMatchStats`
        overrides this with the five pair outcomes, which is all the two need
        to differ in for the variance arithmetic to work for both.
        """
        return ((1.0, self.wins), (0.5, self.draws), (0.0, self.losses))

    @property
    def raw_variance(self):
        """Per-observation variance of the score, straight from the counts.

        Reported for transparency; :attr:`variance` is what the error bars and
        the sequential test actually use.
        """
        counts = self.outcome_counts
        total = sum(count for _, count in counts)
        if total == 0:
            return 0.0
        mu = sum(value * count for value, count in counts) / total
        return sum(count * (value - mu) ** 2 for value, count in counts) / total

    @property
    def variance(self):
        """Per-observation variance of the score, smoothed by a weak prior.

        Draw-heavy matches carry less information per game than decisive ones,
        and this is where that shows up.  See :data:`VARIANCE_PRIOR` for why the
        raw counts are not used directly.
        """
        counts = self.outcome_counts
        total = sum(count for _, count in counts)
        if total == 0:
            return 0.0
        prior = VARIANCE_PRIOR
        smoothed = [(value, count + prior) for value, count in counts]
        weight = sum(count for _, count in smoothed)
        mu = sum(value * count for value, count in smoothed) / weight
        return sum(count * (value - mu) ** 2 for value, count in smoothed) / weight

    @property
    def observations(self):
        """Number of independent observations behind :attr:`variance`.

        For an unpaired match that is simply the number of games.  The paired
        variant overrides it; :func:`sprt` and :attr:`stdev` go through this so
        both kinds of statistics share the same arithmetic.
        """
        return self.games

    @property
    def stdev(self):
        """Standard error of the mean score."""
        if self.observations == 0:
            return 0.0
        return math.sqrt(self.variance / self.observations)

    @property
    def elo(self):
        """Elo difference implied by the observed score."""
        return elo_from_score(self.score)

    def score_interval(self, confidence=0.95):
        """Return the ``(low, high)`` confidence interval of the score.

        :param confidence: two-sided confidence level, e.g. ``0.95``.
        """
        if self.observations == 0:
            return 0.0, 1.0
        z = phi_inv(0.5 + confidence / 2.0)
        return (self.score - z * self.stdev, self.score + z * self.stdev)

    def elo_interval(self, confidence=0.95):
        """Return the ``(low, high)`` confidence interval of the Elo difference."""
        low, high = self.score_interval(confidence)
        return elo_from_score(low), elo_from_score(high)

    @property
    def error_margin(self):
        """Half-width of the 95 % Elo confidence interval.

        This is the ``±`` figure conventionally quoted next to an Elo estimate.
        """
        low, high = self.elo_interval(0.95)
        return (high - low) / 2.0

    @property
    def los(self):
        """Likelihood of superiority as a percentage.

        The probability that engine 1 is genuinely stronger, given the decisive
        games only. Draws carry no information about which side is better and
        are excluded, which is the convention used by chess testing frameworks.
        """
        decisive = self.wins + self.losses
        if decisive == 0:
            return 50.0
        return 100.0 * phi((self.wins - self.losses) / math.sqrt(2.0 * decisive))

    def summary(self):
        """Return the whole thing as a plain dict, ready for JSON."""
        low, high = self.elo_interval(0.95)
        return {
            'wins': self.wins,
            'losses': self.losses,
            'draws': self.draws,
            'games': self.games,
            'score': self.score,
            'draw_ratio': self.draw_ratio,
            'elo': self.elo,
            'elo_low': low,
            'elo_high': high,
            'error_margin': self.error_margin,
            'los': self.los,
        }

    def __repr__(self):
        return ('MatchStats(wins={}, losses={}, draws={}) '
                '<score={:.3f} elo={:+.1f}±{:.1f} los={:.1f}%>'.format(
                    self.wins, self.losses, self.draws,
                    self.score, self.elo, self.error_margin, self.los))


#: Result labels accepted by :class:`PairedMatchStats`, from engine 1's view.
RESULT_WIN, RESULT_LOSS, RESULT_DRAW = 'win', 'loss', 'draw'

#: Score contributed by each result label.
_RESULT_SCORE = {RESULT_WIN: 1.0, RESULT_LOSS: 0.0, RESULT_DRAW: 0.5}


class PairedMatchStats(MatchStats):
    """Statistics for a match whose games come in colour-swapped pairs.

    When both engines are deterministic and every opening is played twice —
    once from each side — the two games of a pair are *not* independent
    samples.  Two equally strong engines score exactly 1-1 on every pair, so
    the raw game-by-game variance overstates the noise enormously: the smoke
    test of this arena scores exactly 50 % over any number of games, while the
    trinomial formula still reports a wide interval.

    Scoring the **pair** instead of the game removes that phantom variance.
    Each pair contributes one observation worth 0, ½, 1, 1½ or 2 points, which
    is the "pentanomial" model used by chess testing frameworks; it typically
    reaches a verdict in roughly half the games a game-by-game test needs.
    That is the difference between a conclusive experiment overnight and one
    that does not fit in the GPU time available.

    An odd trailing game (a match stopped mid-pair) is excluded from the
    statistics and reported by :attr:`unpaired`.

    :param results: game results from engine 1's point of view, **in play
        order**, each one of ``'win'``, ``'loss'`` or ``'draw'``.
    """

    def __init__(self, results):
        results = [result for result in results if result in _RESULT_SCORE]
        self.results = results
        pairs = len(results) // 2
        self.pair_scores = [
            _RESULT_SCORE[results[2 * i]] + _RESULT_SCORE[results[2 * i + 1]]
            for i in range(pairs)]

        counted = results[:2 * pairs]
        super().__init__(
            wins=sum(1 for r in counted if r == RESULT_WIN),
            losses=sum(1 for r in counted if r == RESULT_LOSS),
            draws=sum(1 for r in counted if r == RESULT_DRAW))
        self.unpaired = len(results) - 2 * pairs

    @property
    def pairs(self):
        """Number of complete colour-swapped pairs."""
        return len(self.pair_scores)

    @property
    def observations(self):
        """One observation per pair — this is what makes the statistics paired."""
        return self.pairs

    @property
    def pentanomial(self):
        """Counts of pair outcomes ``[0, ½, 1, 1½, 2]`` points."""
        counts = [0, 0, 0, 0, 0]
        for pair_score in self.pair_scores:
            counts[int(round(pair_score * 2))] += 1
        return counts

    @property
    def score(self):
        """Mean score per game, computed over complete pairs."""
        if not self.pair_scores:
            return 0.5
        return sum(self.pair_scores) / (2.0 * self.pairs)

    @property
    def outcome_counts(self):
        """The five pair outcomes as ``(mean score of the pair, count)``."""
        return tuple((index / 4.0, count)
                     for index, count in enumerate(self.pentanomial))

    def summary(self):
        """Return the whole thing as a plain dict, ready for JSON."""
        summary = super().summary()
        summary.update({
            'paired': True,
            'pairs': self.pairs,
            'unpaired': self.unpaired,
            'pentanomial': self.pentanomial,
        })
        return summary

    def __repr__(self):
        return ('PairedMatchStats(pairs={}) '
                '<score={:.3f} elo={:+.1f}\u00b1{:.1f} los={:.1f}%>'.format(
                    self.pairs, self.score, self.elo, self.error_margin, self.los))


class SprtResult:
    """Outcome of a sequential probability ratio test.

    :param llr: the log-likelihood ratio accumulated so far.
    :param lower: the bound below which ``H0`` is accepted.
    :param upper: the bound above which ``H1`` is accepted.
    :param decision: ``'accept'``, ``'reject'`` or ``'continue'``.
    """

    def __init__(self, llr, lower, upper, decision, elo0, elo1):
        self.llr = llr
        self.lower = lower
        self.upper = upper
        self.decision = decision
        self.elo0 = elo0
        self.elo1 = elo1

    @property
    def finished(self):
        """``True`` once the test has reached one of its bounds."""
        return self.decision != 'continue'

    def summary(self):
        """Return the result as a plain dict, ready for JSON."""
        return {
            'llr': self.llr,
            'lower': self.lower,
            'upper': self.upper,
            'decision': self.decision,
            'elo0': self.elo0,
            'elo1': self.elo1,
        }

    def __repr__(self):
        return 'SprtResult(llr={:+.3f}, bounds=[{:.3f}, {:.3f}], decision={!r})'.format(
            self.llr, self.lower, self.upper, self.decision)


def sprt_bounds(alpha=DEFAULT_ALPHA, beta=DEFAULT_BETA):
    """Return the ``(lower, upper)`` log-likelihood-ratio bounds of an SPRT.

    :param alpha: type-I error rate.
    :param beta: type-II error rate.
    """
    if not 0.0 < alpha < 1.0 or not 0.0 < beta < 1.0:
        raise ValueError('alpha and beta must be in (0, 1)')
    return math.log(beta / (1.0 - alpha)), math.log((1.0 - beta) / alpha)


def sprt(stats, elo0=DEFAULT_ELO0, elo1=DEFAULT_ELO1,
         alpha=DEFAULT_ALPHA, beta=DEFAULT_BETA):
    """Run a sequential probability ratio test on a match in progress.

    Tests ``H0: elo = elo0`` against ``H1: elo = elo1`` using the normal
    approximation to the trinomial score distribution.  Because it is
    *sequential*, a match can be stopped as soon as the answer is clear —
    typically in far fewer games than a fixed-length match of the same power,
    which matters a great deal when one game costs a minute of GPU time.

    :param stats: a :class:`MatchStats` for the games played so far.
    :param elo0: Elo under the null hypothesis (usually ``0``: no improvement).
    :param elo1: Elo under the alternative hypothesis (the gain worth adopting).
    :param alpha: type-I error rate.
    :param beta: type-II error rate.
    :returns: a :class:`SprtResult`.
    """
    lower, upper = sprt_bounds(alpha, beta)

    variance = stats.variance
    if stats.observations < 2 or variance <= 0.0:
        # 分散が0 (全勝・全敗・全引き分け) では正規近似が使えないため保留する
        return SprtResult(0.0, lower, upper, 'continue', elo0, elo1)

    s0 = score_from_elo(elo0)
    s1 = score_from_elo(elo1)
    llr = (stats.observations * (s1 - s0) * (2.0 * stats.score - s0 - s1)
           / (2.0 * variance))

    if llr >= upper:
        decision = 'accept'
    elif llr <= lower:
        decision = 'reject'
    else:
        decision = 'continue'
    return SprtResult(llr, lower, upper, decision, elo0, elo1)


def bradley_terry_ratings(matches, anchor=None, anchor_elo=0.0,
                          prior=2.0, iterations=200, tolerance=1e-9):
    """Fit one Elo rating per engine from a set of pairwise matches.

    Individual matches only ever give a *difference* between two engines.  With
    a history of checkpoints — each tested against its predecessor, some against
    a common baseline — the differences form a graph, and the maximum-likelihood
    ratings that explain the whole graph are more informative than any single
    pairing.  This is the Bradley-Terry model, fitted with the standard
    minorization-maximization iteration.

    A small ``prior`` of virtual drawn games against an average opponent keeps
    the fit finite when an engine has never lost (or never won), and keeps
    engines that are connected by only a handful of games from being placed
    with false confidence.

    :param matches: iterable of mappings with ``player_a``, ``player_b``,
        ``wins`` (of ``player_a``), ``losses`` and ``draws``.
    :param anchor: name of the engine pinned to ``anchor_elo``.  Defaults to the
        engine with the most games, which is usually the common baseline.
    :param anchor_elo: rating assigned to ``anchor``.
    :param prior: strength of the regularising virtual games.
    :param iterations: maximum MM iterations.
    :param tolerance: stop once no rating moves by more than this (in Elo).
    :returns: list of dicts with ``player``, ``elo``, ``games``, ``wins``,
        ``losses``, ``draws`` and ``score``, strongest first.
    """
    # 対戦成績を選手ごと・対戦相手ごとに集計する
    score = {}
    played = {}
    counts = {}
    for match in matches:
        a = match.get('player_a')
        b = match.get('player_b')
        if not a or not b or a == b:
            continue
        wins = float(match.get('wins') or 0)
        losses = float(match.get('losses') or 0)
        draws = float(match.get('draws') or 0)
        total = wins + losses + draws
        if total <= 0:
            continue

        score[a] = score.get(a, 0.0) + wins + 0.5 * draws
        score[b] = score.get(b, 0.0) + losses + 0.5 * draws
        played.setdefault(a, {})[b] = played.setdefault(a, {}).get(b, 0.0) + total
        played.setdefault(b, {})[a] = played.setdefault(b, {}).get(a, 0.0) + total

        for player, w, l in ((a, wins, losses), (b, losses, wins)):
            row = counts.setdefault(player, {'wins': 0.0, 'losses': 0.0, 'draws': 0.0})
            row['wins'] += w
            row['losses'] += l
            row['draws'] += draws

    if not score:
        return []

    players = sorted(score)
    # gamma は 10**(elo/400) 空間での強さ。1.0 (= 0 Elo) から始める
    gamma = {player: 1.0 for player in players}

    for _ in range(iterations):
        largest_change = 0.0
        for player in players:
            denominator = prior / (gamma[player] + 1.0)
            for opponent, games in played[player].items():
                denominator += games / (gamma[player] + gamma[opponent])
            if denominator <= 0.0:
                continue
            updated = (score[player] + prior / 2.0) / denominator
            largest_change = max(largest_change,
                                 abs(400.0 * math.log10(updated / gamma[player])))
            gamma[player] = updated
        if largest_change < tolerance:
            break

    elo = {player: 400.0 * math.log10(value) for player, value in gamma.items()}

    if anchor is None:
        anchor = max(players, key=lambda p: sum(played[p].values()))
    offset = anchor_elo - elo.get(anchor, 0.0)

    rows = []
    for player in players:
        row = counts.get(player, {'wins': 0.0, 'losses': 0.0, 'draws': 0.0})
        games = sum(played[player].values())
        rows.append({
            'player': player,
            'elo': elo[player] + offset,
            'games': int(round(games)),
            'wins': int(round(row['wins'])),
            'losses': int(round(row['losses'])),
            'draws': int(round(row['draws'])),
            'score': (score[player] / games) if games else 0.5,
            'is_anchor': player == anchor,
        })
    return sorted(rows, key=lambda r: r['elo'], reverse=True)
