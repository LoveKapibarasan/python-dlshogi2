"""Sphinx configuration for the python-dlshogi2 documentation.

Builds API reference pages with ``autodoc`` from the in-source reStructuredText
docstrings, plus the hand-written guides under ``docs/``.

Build locally with::

    pip install -r docs/requirements.txt
    sphinx-build -b html docs docs/_build/html

Note: ``autodoc`` imports the package, so the project and its runtime
dependencies (``torch``, ``cshogi``, ``numpy``, ``onnxruntime``) must be
installed in the environment that builds the docs.
"""
import os
import sys

# プロジェクトルートをimportパスに追加
sys.path.insert(0, os.path.abspath('..'))
# dashboard/ はパッケージではなくスクリプト置き場なので個別に追加する
sys.path.insert(0, os.path.abspath('../dashboard'))

# -- Project information -----------------------------------------------------
project = 'python-dlshogi2'
author = 'LoveKapibarasan'
copyright = '2026, LoveKapibarasan'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',       # docstringからAPIリファレンスを生成
    'sphinx.ext.autosummary',   # モジュール一覧の自動生成
    'sphinx.ext.napoleon',      # Google/NumPy形式のdocstringも許容
    'sphinx.ext.viewcode',      # ソースコードへのリンク
    'sphinx.ext.intersphinx',   # 外部ドキュメントへの相互リンク
]

autosummary_generate = True
autodoc_typehints = 'description'
autodoc_member_order = 'bysource'
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
}

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'torch': ('https://pytorch.org/docs/stable', None),
    'numpy': ('https://numpy.org/doc/stable', None),
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- HTML output -------------------------------------------------------------
# alabasterはSphinx同梱なので追加依存なしでビルドできる
html_theme = 'alabaster'
html_static_path = ['_static']
