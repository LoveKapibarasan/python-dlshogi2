"""Streamlit dashboard for the training / self-play development history.

Reads the JSON Lines files written by ``pydlshogi2.train``,
``pydlshogi2.selfplay`` and ``rl_loop.sh`` (see :mod:`pydlshogi2.metrics`) and
renders them as a browsable experiment history: which runs happened, on which
commit and hyper-parameters, how the losses moved, and what the RL loop's
self-play produced along the way.

Run it with::

    pip install -r dashboard/requirements.txt
    streamlit run dashboard/app.py

The metrics directory can be preset with the ``DLSHOGI_METRICS_DIR``
environment variable, and the checkpoint directory with
``DLSHOGI_CHECKPOINT_DIR``; both are editable in the sidebar.
"""
import datetime
import os
import sys

import altair as alt
import pandas as pd
import streamlit as st

# `streamlit run dashboard/app.py` puts dashboard/ on sys.path but not the repo
# root, so make the sibling import work either way.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import metrics_store  # noqa: E402

DEFAULT_METRICS_DIR = os.environ.get('DLSHOGI_METRICS_DIR', 'metrics')
DEFAULT_CHECKPOINT_DIR = os.environ.get('DLSHOGI_CHECKPOINT_DIR', 'checkpoints')

#: Curves offered in the learning-curve tab, in display order.
LOSS_COLUMNS = [
    'train_loss_total', 'train_loss_policy', 'train_loss_value',
    'test_loss_total', 'test_loss_policy', 'test_loss_value',
]
ACCURACY_COLUMNS = ['test_accuracy_policy', 'test_accuracy_value']


def format_timestamp(value):
    """Render a POSIX timestamp as ``YYYY-MM-DD HH:MM:SS`` (empty when unset)."""
    if not value:
        return ''
    return datetime.datetime.fromtimestamp(value).strftime('%Y-%m-%d %H:%M:%S')


@st.cache_data(show_spinner=False)
def load_records(directory, cache_key):
    """Load every metrics record under ``directory``.

    :param cache_key: value that changes whenever the files change, so Streamlit
        invalidates the cache; see :func:`directory_fingerprint`.
    """
    del cache_key  # only used to key the cache
    return metrics_store.load(directory)


def directory_fingerprint(directory):
    """Return a value that changes when any ``.jsonl`` under ``directory`` does.

    Files are appended to while runs are live, so both the file list and their
    sizes/mtimes matter.
    """
    fingerprint = []
    for path in metrics_store.find_metric_files(directory):
        try:
            stat = os.stat(path)
        except OSError:
            continue
        fingerprint.append((path, stat.st_size, stat.st_mtime))
    return tuple(fingerprint)


def long_form(points, columns, key='step'):
    """Reshape metric samples into a tidy frame for Altair.

    :param points: metric records from :func:`metrics_store.curve_points`.
    :param columns: metric field names to melt into ``metric``/``value`` pairs.
    :returns: a DataFrame with ``run_id``, ``step``, ``metric``, ``value``.
    """
    rows = []
    for point in points:
        for column in columns:
            value = point.get(column)
            if value is None:
                continue
            rows.append({
                'run_id': point.get('run_id'),
                key: point.get(key),
                'metric': column,
                'value': value,
            })
    return pd.DataFrame(rows)


def line_chart(frame, x, y, color, title):
    """Render one Altair line chart, or an info box when there is no data."""
    if frame.empty:
        st.info('該当するデータがありません。')
        return
    chart = (
        alt.Chart(frame)
        .mark_line()
        .encode(
            x=alt.X(f'{x}:Q', title=x),
            y=alt.Y(f'{y}:Q', title=y, scale=alt.Scale(zero=False)),
            color=alt.Color(f'{color}:N', title=color),
            tooltip=list(frame.columns),
        )
        .properties(height=380, title=title)
        .interactive()
    )
    st.altair_chart(chart, width='stretch')


def render_runs_tab(summaries):
    """Table of every run: when, on what commit, with which hyper-parameters."""
    st.subheader('Runs')
    if not summaries:
        st.info('メトリクスが見つかりません。`--metrics` を付けて学習を実行してください。')
        return

    rows = []
    for summary in summaries:
        rows.append({
            'run_id': summary.get('run_id'),
            'kind': summary.get('kind'),
            'status': summary.get('status'),
            'started': summary.get('started_at') or format_timestamp(summary.get('timestamp')),
            'commit': summary.get('git_commit'),
            'dirty': summary.get('git_dirty'),
            'host': summary.get('hostname'),
            'gpu': summary.get('gpu_name'),
            'lr': summary.get('lr'),
            'batchsize': summary.get('batchsize'),
            'val_lambda': summary.get('val_lambda'),
            'blocks': summary.get('blocks'),
            'channels': summary.get('channels'),
            'epoch': summary.get('last_epoch'),
            'step': summary.get('last_step'),
            'test_acc_policy': summary.get('test_accuracy_policy'),
            'test_acc_value': summary.get('test_accuracy_value'),
            'samples': summary.get('samples'),
            'last_checkpoint': summary.get('last_checkpoint'),
        })
    st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

    labels = {s.get('run_id'): '{} ({})'.format(s.get('run_id'), s.get('kind'))
              for s in summaries}
    selected = st.selectbox('詳細を見る run', list(labels), format_func=labels.get)
    detail = next((s for s in summaries if s.get('run_id') == selected), None)
    if detail:
        left, right = st.columns(2)
        with left:
            st.markdown('**引数**')
            st.json(detail.get('args') or {})
        with right:
            st.markdown('**実行環境 / 結果**')
            st.json({
                'network': detail.get('network'),
                'train_positions': detail.get('train_positions'),
                'test_positions': detail.get('test_positions'),
                'duration_sec': detail.get('duration_sec'),
                'source_file': detail.get('source_file'),
                'parent_run_id': detail.get('parent_run_id'),
                'resume': detail.get('resume'),
            })


def render_curves_tab(records, summaries):
    """Overlaid learning curves for the runs the user selects."""
    st.subheader('学習曲線')
    train_runs = [s.get('run_id') for s in summaries if s.get('kind') == 'train']
    if not train_runs:
        st.info('学習 run のメトリクスがありません。')
        return

    selected = st.multiselect('比較する run', train_runs,
                              default=train_runs[:3])
    scope = st.radio('粒度', ['interval', 'epoch'], horizontal=True,
                     help='interval = --eval_interval ごと、epoch = エポック終端の全テスト評価')
    if not selected:
        st.info('run を選択してください。')
        return

    points = metrics_store.curve_points(records, run_ids=set(selected), scope=scope)
    if not points:
        st.info('選択した run にこの粒度のデータがありません。')
        return

    loss_metrics = st.multiselect('損失', LOSS_COLUMNS,
                                  default=['train_loss_total', 'test_loss_total'])
    line_chart(long_form(points, loss_metrics), 'step', 'value', 'metric', 'Loss')

    accuracy_frame = long_form(points, ACCURACY_COLUMNS)
    if not accuracy_frame.empty:
        # run と metric の組み合わせで色分けする (複数 run 比較のため)
        accuracy_frame['series'] = accuracy_frame['run_id'] + ' / ' + accuracy_frame['metric']
        line_chart(accuracy_frame, 'step', 'value', 'series', 'Test accuracy')

    with st.expander('生データ'):
        st.dataframe(pd.DataFrame(points), width='stretch', hide_index=True)


def render_rl_tab(records):
    """Per-iteration view of the reinforcement-learning loop."""
    st.subheader('RL ループ')

    iterations = metrics_store.rl_iterations(records)
    selfplay = metrics_store.selfplay_by_iteration(records)
    if not iterations and not selfplay:
        st.info('RL ループのメトリクスがありません (`rl_loop.sh` を実行すると記録されます)。')
        return

    if selfplay:
        st.markdown('**自己対局の統計 (イテレーションごと、全ワーカー合算)**')
        frame = pd.DataFrame(selfplay)
        st.dataframe(frame, width='stretch', hide_index=True)

        rate_frame = frame.melt(
            id_vars='iteration',
            value_vars=[c for c in ('black_win_rate', 'white_win_rate', 'draw_rate')
                        if c in frame.columns],
            var_name='metric', value_name='value')
        line_chart(rate_frame, 'iteration', 'value', 'metric', '勝敗の分布')
        line_chart(
            frame.melt(id_vars='iteration', value_vars=['mean_moves'],
                       var_name='metric', value_name='value'),
            'iteration', 'value', 'metric', '平均手数')

    if iterations:
        st.markdown('**イテレーションの成果物**')
        frame = pd.DataFrame(iterations)
        keep = [c for c in ('iteration', 'model', 'data', 'checkpoint',
                            'data_bytes', 'seconds') if c in frame.columns]
        st.dataframe(frame[keep], width='stretch', hide_index=True)

    # 各イテレーションの学習結果 (train run 側) をイテレーション番号で並べる
    train_points = [p for p in metrics_store.curve_points(records, scope='epoch')
                    if p.get('run_id') and '-train-' in str(p.get('run_id'))]
    if train_points:
        st.markdown('**イテレーションごとの学習後の指標**')
        frame = long_form(train_points, ['test_loss_total', 'test_accuracy_policy'])
        line_chart(frame, 'step', 'value', 'metric', 'RL 学習の推移 (step 軸)')


def render_checkpoints_tab(checkpoint_dir):
    """Files under the checkpoint directory, with optional architecture lookup."""
    st.subheader('チェックポイント')
    rows = metrics_store.list_checkpoints(checkpoint_dir)
    if not rows:
        st.info('`{}` にモデルファイルが見つかりません。'.format(checkpoint_dir))
        return

    frame = pd.DataFrame(rows)
    frame['modified'] = frame['modified'].map(format_timestamp)
    st.dataframe(frame, width='stretch', hide_index=True)

    pth_files = [r['path'] for r in rows if r['path'].endswith('.pth')]
    if not pth_files:
        return
    selected = st.selectbox('ネットワーク構成を読む (.pth)', pth_files)
    # torch のロードは重いのでボタンを押したときだけ実行する
    if st.button('構成を読み込む'):
        network = metrics_store.read_checkpoint_network(selected)
        if network is None:
            st.warning('構成情報を読めませんでした '
                       '(構成が埋め込まれる前の古い checkpoint か、torch が未インストール)。')
        else:
            st.json(network)


def main():
    """Build the whole page."""
    st.set_page_config(page_title='python-dlshogi2 dashboard',
                       page_icon='♘', layout='wide')
    st.title('python-dlshogi2 開発履歴ダッシュボード')

    with st.sidebar:
        st.header('データソース')
        metrics_dir = st.text_input('メトリクスディレクトリ', DEFAULT_METRICS_DIR)
        checkpoint_dir = st.text_input('チェックポイントディレクトリ',
                                       DEFAULT_CHECKPOINT_DIR)
        if st.button('再読み込み'):
            st.cache_data.clear()
        files = metrics_store.find_metric_files(metrics_dir)
        st.caption('{} 個の .jsonl を検出'.format(len(files)))
        with st.expander('ファイル一覧'):
            st.write(files or 'なし')

    records = load_records(metrics_dir, directory_fingerprint(metrics_dir))
    summaries = metrics_store.summarize_runs(records)

    runs_tab, curves_tab, rl_tab, checkpoints_tab = st.tabs(
        ['Runs', '学習曲線', 'RL ループ', 'チェックポイント'])
    with runs_tab:
        render_runs_tab(summaries)
    with curves_tab:
        render_curves_tab(records, summaries)
    with rl_tab:
        render_rl_tab(records)
    with checkpoints_tab:
        render_checkpoints_tab(checkpoint_dir)


main()
