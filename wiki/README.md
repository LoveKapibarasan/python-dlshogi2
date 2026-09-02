# Wiki source

The pages of the [project wiki](https://github.com/LoveKapibarasan/python-dlshogi2/wiki)
are kept **here, in the main repository**, and pushed to the wiki by
`publish.sh`.

## Why not edit the wiki directly?

A GitHub wiki is its own git repository
(`git@github.com:LoveKapibarasan/python-dlshogi2.wiki.git`), separate from the
code. Edited in the browser, it drifts: a page describing `--val_lambda` gets no
review when `--val_lambda` changes, and nothing links the two.

Keeping the source here means wiki changes arrive in the same pull request as
the code they describe, get reviewed, and are reachable from `git log`.

The trade-off is that **edits made in the browser or by cloning the wiki repo
will be overwritten** by the next `publish.sh`. This directory is the source of
truth.

## Publishing

```bash
./wiki/publish.sh                       # push to the wiki
./wiki/publish.sh --dry-run             # show what would change
```

The script clones the wiki repository into a temporary directory, mirrors the
`.md` files from here, and pushes if anything changed.

**The wiki must have at least one page before its git repository exists.** On a
brand-new repository, `publish.sh` will fail to clone; create any page once at
`https://github.com/LoveKapibarasan/python-dlshogi2/wiki` (the "Create the first
page" button — content does not matter, it gets overwritten) and run the script
again.

## Page naming

GitHub derives a page's title from its filename: `Training-Pipeline.md` becomes
"Training Pipeline" at `/wiki/Training-Pipeline`. Link between pages with the
filename, not the title:

```markdown
See [Training Pipeline](Training-Pipeline).
```

`_Sidebar.md` is special — it renders as the sidebar on every page. `Home.md` is
the landing page.

## Adding a page

1. Add `Your-Page.md` here.
2. Link it from `Home.md` and `_Sidebar.md`.
3. Run `./wiki/publish.sh`.

Before writing, check it belongs here rather than in the README (installation
and the shortest path to running) or a docstring (what a function takes and
returns). This wiki is for design background, operating procedures and records.
