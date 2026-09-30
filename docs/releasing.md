# Releasing, rehearsing, and rolling back

Maintainer-only process documentation — not part of the user-facing docs
index (`docs/index.md`), since it has nothing to do with using the library.

## Cutting a real release

`scripts/release.sh VERSION` is the only sanctioned path — see that script's
own header comment for the full step-by-step. In short: it runs every check
`release-validation.yml` runs (plus version/changelog consistency), builds
and smoke-tests the exact artifact that would be published, requires typed
confirmation, then pushes the `vX.Y.Z` tag via `gh release create`. Pushing
that tag triggers `.github/workflows/publish.yml`, which re-runs the full
validation suite against the tagged commit, then pauses on the `pypi`
GitHub Environment's required-reviewer gate until you explicitly approve
that specific run in GitHub's UI (Actions → the run → Review deployments).
Nothing reaches PyPI without that manual click, no matter what triggered
the workflow.

## Rehearsing the publish path before trusting it with a real release (P2-5)

`.github/workflows/publish-testpypi.yml` runs the *exact same* shape of
pipeline as `publish.yml` — the reusable `release-validation.yml` suite,
a build-once/download-artifact hand-off (never a second build right before
upload), a version-agreement check, the same pinned
`pypa/gh-action-pypi-publish` action — against **TestPyPI** instead of
production PyPI. It's `workflow_dispatch`-only (manual trigger, never a tag
push), so it can never fire by accident.

One-time setup (see the workflow file's own header comment for the exact
commands): a TestPyPI account and API token, a `TEST_PYPI_TOKEN` repository
secret, and a `testpypi` GitHub Environment. None of this touches the real
`PYPI_TOKEN` secret or `pypi` environment `publish.yml` uses.

Confirmed already in place for the **production** path (2026-09-29, via
read-only `gh api`/`gh secret list` checks — see
`prompts/action-list-20260923.md` P2-5 for the full note): the `pypi`
GitHub Environment exists with its required-reviewers rule intact, and
`PYPI_TOKEN` is present as a repository secret. What a TestPyPI rehearsal
run additionally proves, that a read-only config check cannot: the reusable
validation suite genuinely produces an installable artifact, the
download-artifact hand-off actually carries that exact artifact through to
the publish step untouched, the environment-approval gate genuinely pauses
the run where expected, and the pinned `pypa/gh-action-pypi-publish` action
genuinely succeeds at an upload with these exact inputs. This still hasn't
been run for real as of this writing — it needs a TestPyPI account, which
only the project owner can create.

After a rehearsal run, verify by hand: the project appears at
`https://test.pypi.org/project/sorethumb-ml/` with the expected version,
and `pip install -i https://test.pypi.org/simple/ sorethumb-ml` succeeds in
a clean venv (`sorethumb --version` and `import sorethumb_ml` both work).

## Rolling back a bad release

PyPI (and TestPyPI) files are immutable once uploaded: the same exact
version number can never be re-uploaded, even after being yanked, and a
file that's already been downloaded/mirrored/cached elsewhere can't be
un-published in any real sense. The correct mechanism for "this release
should not be used" is **yanking**, not deletion:

- A yanked release is skipped by `pip install sorethumb-ml` (no explicit
  version) — pip picks the next available, non-yanked release instead —
  but `pip install sorethumb-ml==X.Y.Z` (an explicit, already-pinned
  version) still works. This means yanking doesn't retroactively break
  someone's existing lockfile/pinned install; it only stops *new,
  unpinned* installs from picking up the bad version.
- How: log into pypi.org (or test.pypi.org for a rehearsal artifact) →
  the project's "Manage" page → the specific release's options menu →
  "Yank release". There is no public API for this as of writing — it's a
  web-UI-only action, deliberately requiring a human in the loop, not
  something `scripts/release.sh` or any CI workflow can do on your behalf.
- Always give a real reason in the yank's text field (e.g., "0.1.0 shipped
  with a scoring regression, see CHANGELOG `[0.1.1]`") — it's shown to
  anyone who looks at the release afterward.

After yanking:

1. Fix the actual underlying bug.
2. Cut a new **patch** version — never attempt to reuse the yanked version
   number, since PyPI will refuse the upload outright.
3. Add a `CHANGELOG.md` entry under the new version explaining what was
   wrong with the yanked one, not just what the new one changes.
4. Run `scripts/release.sh` for the new version as normal.

**Never delete or force-move a git tag** that already triggered a real
publish attempt (`scripts/release.sh` calls this out too: "the tag itself,
once pushed, is meant to be permanent"). If a tag was pushed by mistake
*before* the actual PyPI upload happened (e.g. a typo noticed while the run
is still paused on the environment-approval gate): reject or simply never
approve that deployment in GitHub's Environments UI — no upload happens,
so there's nothing to yank. If the upload already went through, follow the
yank procedure above instead of touching the tag.
