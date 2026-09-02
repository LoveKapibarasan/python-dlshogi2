"""Checks that the wiki sources in ``wiki/`` stay internally consistent.

GitHub renders a wiki page at ``/wiki/<filename without .md>``, so an internal
link is just the filename. Nothing validates those links at publish time — a
renamed page silently produces dead links — hence this test.

Run with::

    python -m unittest discover -s tests
"""
import os
import re
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WIKI_DIR = os.path.join(REPO_ROOT, 'wiki')

#: ``[text](target)`` links. External links and anchors are filtered out below.
LINK_RE = re.compile(r'\[[^\]]*\]\(([^)]+)\)')

#: ``wiki/README.md`` documents the publishing workflow and is not a wiki page.
NOT_A_PAGE = {'README.md'}


def wiki_pages():
    """Return the Markdown files that get published as wiki pages."""
    return sorted(name for name in os.listdir(WIKI_DIR)
                  if name.endswith('.md') and name not in NOT_A_PAGE)


def internal_links(text):
    """Yield the internal wiki targets referenced by ``text``.

    Skips absolute URLs, anchors within the same page, and image/badge links,
    keeping only the ``Page`` and ``Page#section`` forms.
    """
    for target in LINK_RE.findall(text):
        target = target.strip()
        if target.startswith(('http://', 'https://', 'mailto:', '#', '/')):
            continue
        yield target


class WikiSourceTest(unittest.TestCase):
    def setUp(self):
        self.pages = wiki_pages()
        self.page_names = {os.path.splitext(name)[0] for name in self.pages}

    def test_expected_pages_exist(self):
        for required in ('Home.md', '_Sidebar.md'):
            self.assertIn(required, self.pages)

    def test_internal_links_point_at_real_pages(self):
        broken = []
        for name in self.pages:
            with open(os.path.join(WIKI_DIR, name), encoding='utf-8') as f:
                text = f.read()
            for target in internal_links(text):
                page = target.split('#', 1)[0]
                if page and page not in self.page_names:
                    broken.append('{} -> {}'.format(name, target))
        self.assertEqual(broken, [], 'broken wiki links: {}'.format(broken))

    def test_every_page_is_reachable_from_home_or_sidebar(self):
        reachable = set()
        for entry in ('Home.md', '_Sidebar.md'):
            with open(os.path.join(WIKI_DIR, entry), encoding='utf-8') as f:
                text = f.read()
            reachable.update(t.split('#', 1)[0] for t in internal_links(text))

        orphans = sorted(self.page_names - reachable - {'Home', '_Sidebar'})
        self.assertEqual(orphans, [],
                         'pages not linked from Home or the sidebar: {}'.format(orphans))

    def test_publish_script_is_executable(self):
        script = os.path.join(WIKI_DIR, 'publish.sh')
        self.assertTrue(os.path.exists(script))
        self.assertTrue(os.access(script, os.X_OK))


if __name__ == '__main__':
    unittest.main()
