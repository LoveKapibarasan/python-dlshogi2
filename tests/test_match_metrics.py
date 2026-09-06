"""Tests for the match/rating/backlog views in ``dashboard/metrics_store.py``.

Standard library only, like the rest of ``tests/`` — the store is deliberately
free of pandas and Streamlit so it stays testable without them.

Run with::

    python -m unittest discover -s tests
"""
import json
import os
import shutil
import sys
import tempfile
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, 'dashboard'))

import metrics_store  # noqa: E402


def match_records(run_id='m1', experiment='EXP-001', summary=True,
                  wins=57, losses=38, draws=5, games=None):
    """Build the records one match would write."""
    records = [{
        'type': 'run', 'run_id': run_id, 'kind': 'match', 'timestamp': 100.0,
        'started_at': '2026-09-05 12:00:00', 'git_commit': 'abcdef1234',
        'experiment': experiment, 'issue': 7,
        'player_a': 'branch', 'player_b': 'main',
        'args': {'byoyomi': 1000, 'games': 100},
    }]
    total = wins + losses + draws
    records.append({
        'type': 'metric', 'run_id': run_id, 'scope': 'game',
        'game': total, 'result': 'win', 'wins': wins, 'losses': losses,
        'draws': draws, 'score': 0.5, 'elo': 12.0, 'error_margin': 60.0,
        'los': 70.0, 'llr': 0.5, 'sprt_decision': 'continue',
        'player_a': 'branch', 'player_b': 'main',
    })
    if summary:
        records.append({
            'type': 'metric', 'run_id': run_id, 'scope': 'summary',
            'player_a': 'branch', 'player_b': 'main',
            'wins': wins, 'losses': losses, 'draws': draws,
            'games': games if games is not None else total,
            'score': 0.595, 'elo': 66.8, 'error_margin': 68.4, 'los': 91.6,
            'sprt_decision': 'accept',
        })
        records.append({
            'type': 'event', 'run_id': run_id, 'event': 'run_end',
            'status': 'completed', 'timestamp': 200.0,
        })
    return records


class MatchSummariesTest(unittest.TestCase):
    def test_ignores_non_match_runs(self):
        records = [{'type': 'run', 'run_id': 't1', 'kind': 'train'}]
        self.assertEqual(metrics_store.match_summaries(records), [])

    def test_summary_record_wins(self):
        rows = metrics_store.match_summaries(match_records())
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row['experiment'], 'EXP-001')
        self.assertEqual(row['issue'], 7)
        self.assertEqual((row['wins'], row['losses'], row['draws']), (57, 38, 5))
        self.assertEqual(row['games'], 100)
        self.assertEqual(row['status'], 'completed')
        self.assertAlmostEqual(row['elo'], 66.8)
        self.assertEqual(row['git_commit'], 'abcdef12')

    def test_unfinished_match_falls_back_to_the_last_game(self):
        # 60局で落ちた100局マッチも証拠として読めなければならない
        rows = metrics_store.match_summaries(
            match_records(summary=False, wins=35, losses=20, draws=5))
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row['status'], 'running')
        self.assertEqual(row['games'], 60)
        self.assertEqual(row['wins'], 35)

    def test_newest_first(self):
        records = match_records(run_id='old') + match_records(run_id='new')
        for record in records:
            if record.get('run_id') == 'new' and record.get('type') == 'run':
                record['timestamp'] = 999.0
        rows = metrics_store.match_summaries(records)
        self.assertEqual([r['run_id'] for r in rows], ['new', 'old'])


class RatingTableTest(unittest.TestCase):
    def test_builds_a_scale_from_matches(self):
        rows = metrics_store.match_summaries(match_records())
        table = metrics_store.rating_table(rows, anchor='main')
        by_player = {r['player']: r for r in table}
        self.assertAlmostEqual(by_player['main']['elo'], 0.0)
        self.assertGreater(by_player['branch']['elo'], 0.0)

    def test_matches_without_games_are_skipped(self):
        rows = [{'player_a': 'a', 'player_b': 'b', 'games': 0,
                 'wins': 0, 'losses': 0, 'draws': 0}]
        self.assertEqual(metrics_store.rating_table(rows), [])


class BacklogTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.mkdtemp()
        self.path = os.path.join(self.directory, 'Improvement-Backlog.md')

    def tearDown(self):
        shutil.rmtree(self.directory, ignore_errors=True)

    def write(self, text):
        with open(self.path, 'w', encoding='utf-8') as f:
            f.write(text)

    def test_parses_the_table(self):
        self.write('# Backlog\n\n'
                   'prose that is not a table\n\n'
                   '| ID | 改善案 | 状態 |\n'
                   '|----|--------|------|\n'
                   '| EXP-001 | make it faster | 計測中 |\n'
                   '| EXP-002 | make it smarter | 未着手 |\n\n'
                   'more prose\n')
        rows = metrics_store.load_backlog(self.path)
        self.assertEqual([r['ID'] for r in rows], ['EXP-001', 'EXP-002'])
        self.assertEqual(rows[0]['改善案'], 'make it faster')
        self.assertEqual(rows[1]['状態'], '未着手')

    def test_skips_tables_without_an_id_column(self):
        self.write('| 内訳 | 秒 |\n|------|----|\n| a | 1 |\n\n'
                   '| ID | 改善案 |\n|----|--------|\n| EXP-001 | x |\n')
        rows = metrics_store.load_backlog(self.path)
        self.assertEqual([r['ID'] for r in rows], ['EXP-001'])

    def test_missing_file_is_not_an_error(self):
        self.assertEqual(metrics_store.load_backlog(self.path + '.nope'), [])

    def test_repository_backlog_parses(self):
        # 実際のページが壊れたら気付けるようにしておく
        rows = metrics_store.load_backlog()
        self.assertTrue(rows, 'wiki/Improvement-Backlog.md has no parsable table')
        for row in rows:
            self.assertRegex(row['ID'], r'^EXP-\d{3}$')
            self.assertTrue(row.get('改善案'))
            self.assertTrue(row.get('状態'))

    def test_backlog_ids_are_unique(self):
        ids = [row['ID'] for row in metrics_store.load_backlog()]
        self.assertEqual(len(ids), len(set(ids)))

    def test_join_attaches_matches(self):
        self.write('| ID | 改善案 |\n|----|--------|\n| EXP-001 | faster |\n')
        rows = metrics_store.backlog_with_results(match_records(), self.path)
        by_id = {row['ID']: row for row in rows}
        self.assertEqual(by_id['EXP-001']['match_count'], 1)
        self.assertAlmostEqual(by_id['EXP-001']['measured_elo'], 66.8)
        self.assertEqual(by_id['EXP-001']['sprt_decision'], 'accept')

    def test_join_keeps_experiments_missing_from_the_backlog(self):
        self.write('| ID | 改善案 |\n|----|--------|\n| EXP-001 | faster |\n')
        rows = metrics_store.backlog_with_results(
            match_records(experiment='EXP-999'), self.path)
        by_id = {row['ID']: row for row in rows}
        self.assertIn('EXP-999', by_id)
        self.assertEqual(by_id['EXP-001']['match_count'], 0)


class EndToEndFileTest(unittest.TestCase):
    def test_reads_a_jsonl_file_from_disk(self):
        directory = tempfile.mkdtemp()
        try:
            path = os.path.join(directory, 'match.jsonl')
            with open(path, 'w', encoding='utf-8') as f:
                for record in match_records():
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
                f.write('this line is not json\n')
            rows = metrics_store.match_summaries(metrics_store.load(directory))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['games'], 100)
        finally:
            shutil.rmtree(directory, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
