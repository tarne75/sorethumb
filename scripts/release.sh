#!/usr/bin/env bash
# Cut a sorethumb release: run every local check this project has (the same
# ones release-validation.yml runs, plus the version/changelog consistency
# work from P0-10), then create and push the vX.Y.Z tag via `gh release
# create` -- which is what actually triggers .github/workflows/publish.yml.
#
# Deliberately the ONLY way this repo creates a release tag. It never runs
# automatically and never guesses a version -- you must name the exact
# version you mean to release, every check must pass, and you must
# re-type that version to confirm before anything is pushed. Pushing the
# PyPI publish itself still requires a separate manual approval click in
# GitHub's UI (the `pypi` environment's required-reviewers rule, added in
# release-launch-plan.md Item 2) -- this script only gets you to a pushed
# tag, never all the way to PyPI on its own.
#
# Usage:
#   scripts/release.sh 0.2.0            # run every check, then tag+push
#   scripts/release.sh 0.2.0 --dry-run  # run every check, stop before tagging
#
# What this does NOT do, on purpose (each is a separate, deliberate step
# you take yourself first):
#   - Does not bump pyproject.toml's version -- it must already say the
#     version you're releasing.
#   - Does not rewrite CHANGELOG.md's [Unreleased] section into a dated
#     release section -- that section must already exist with real content.
#   - Does not commit or push anything to `main` -- your working tree must
#     already be clean and already match origin/main exactly.
set -euo pipefail

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 VERSION [--dry-run]" >&2
  echo "  e.g. $0 0.2.0" >&2
  exit 2
fi

VERSION="$1"
DRY_RUN=0
if [[ "${2:-}" == "--dry-run" ]]; then
  DRY_RUN=1
elif [[ -n "${2:-}" ]]; then
  echo "Unknown second argument: $2 (only --dry-run is accepted)" >&2
  exit 2
fi

if [[ ! "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "Version must look like X.Y.Z (matching publish.yml's tag pattern), got: $VERSION" >&2
  exit 2
fi

TAG="v$VERSION"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_step() { printf '\n\033[1;34m==>\033[0m %s\n' "$1"; }
_ok() { printf '\033[1;32m    ok:\033[0m %s\n' "$1"; }
_fail() {
  printf '\033[1;31mFAILED:\033[0m %s\n' "$1" >&2
  exit 1
}

# ---------------------------------------------------------------------------
# 1. Repo/git state -- cheap checks first, before spending any time running
#    tests or builds against a state that couldn't be tagged anyway.
# ---------------------------------------------------------------------------

_step "Checking git state"

BRANCH="$(git rev-parse --abbrev-ref HEAD)"
[[ "$BRANCH" == "main" ]] || _fail "on branch '$BRANCH', not 'main'. Switch to main first."
_ok "on main"

[[ -z "$(git status --porcelain)" ]] || _fail "working tree is not clean. Commit or stash first."
_ok "working tree clean"

git fetch origin main --quiet
LOCAL_SHA="$(git rev-parse HEAD)"
REMOTE_SHA="$(git rev-parse origin/main)"
if [[ "$LOCAL_SHA" != "$REMOTE_SHA" ]]; then
  _fail "local main ($LOCAL_SHA) != origin/main ($REMOTE_SHA). Push or pull first -- the tag must point at a commit already on origin/main (publish.yml's verify-tag-on-main job checks this too, so a mismatch here would just fail remotely, later, after using CI minutes)."
fi
_ok "local main matches origin/main ($LOCAL_SHA)"

if git rev-parse "$TAG" >/dev/null 2>&1; then
  _fail "tag $TAG already exists locally. Never re-push an existing tag; pick a new version or investigate why this exists."
fi
if [[ -n "$(git ls-remote --tags origin "refs/tags/$TAG")" ]]; then
  _fail "tag $TAG already exists on origin. Never re-push an existing tag; pick a new version or investigate why this exists."
fi
_ok "$TAG does not already exist, locally or on origin"

# ---------------------------------------------------------------------------
# 2. Confirm the exact commit being tagged already went green on GitHub's
#    own CI, not just "looked fine locally a moment ago".
# ---------------------------------------------------------------------------

_step "Checking GitHub Actions status for $LOCAL_SHA"

CI_CONCLUSION="$(gh run list --branch main --limit 10 --json headSha,conclusion,status \
  --jq "map(select(.headSha == \"$LOCAL_SHA\")) | .[0].conclusion // \"\"")"
if [[ "$CI_CONCLUSION" != "success" ]]; then
  _fail "most recent CI run for $LOCAL_SHA on main is '${CI_CONCLUSION:-not found}', not 'success'. Check https://github.com/tarne75/sorethumb/actions before releasing."
fi
_ok "CI run for this commit succeeded on GitHub"

# ---------------------------------------------------------------------------
# 3. Version and changelog consistency -- the P0-10 concerns, checked
#    locally before anything is tagged rather than only in publish.yml.
# ---------------------------------------------------------------------------

_step "Checking pyproject.toml's version matches $VERSION"

PYPROJECT_VERSION="$(python3 -c "import tomllib; print(tomllib.load(open('pyproject.toml','rb'))['project']['version'])")"
if [[ "$PYPROJECT_VERSION" != "$VERSION" ]]; then
  _fail "pyproject.toml says version = \"$PYPROJECT_VERSION\", not \"$VERSION\". Bump it, commit, push, wait for CI, then re-run this script -- this script never edits pyproject.toml itself."
fi
_ok "pyproject.toml version matches"

_step "Checking CHANGELOG.md has a real [$VERSION] section"

CHANGELOG_BODY="$(awk -v ver="[$VERSION]" '
  /^## \[/ { if (found) exit; if (index($0, ver) == 1 + length("## ")) { found=1; next } }
  found { print }
' CHANGELOG.md)"
if [[ -z "$(echo "$CHANGELOG_BODY" | tr -d '[:space:]')" ]]; then
  _fail "CHANGELOG.md has no non-empty '## [$VERSION]' section. Rename [Unreleased] to '## [$VERSION] - $(date +%Y-%m-%d)' (with real content under it), commit, push, wait for CI, then re-run this script -- this script never edits CHANGELOG.md itself."
fi
_ok "CHANGELOG.md has a [$VERSION] section with content"

# ---------------------------------------------------------------------------
# 4. The same checks release-validation.yml runs, run locally for fast
#    feedback (publish.yml will run them again, for real, against the
#    tagged commit -- this is a pre-flight, not a replacement for that).
# ---------------------------------------------------------------------------

_step "uv lock --check"
uv lock --check
_ok "lockfile matches pyproject.toml"

_step "uv sync --all-extras --frozen"
uv sync --all-extras --frozen
_ok "synced"

_step "ruff check"
uv run ruff check src/ tests/
_step "ruff format --check"
uv run ruff format --check src/ tests/
_step "mypy"
uv run mypy src/
_step "docs/generate_config_docs.py --check"
uv run python docs/generate_config_docs.py --check

_step "pytest: unit, contract, property, repo_check"
uv run pytest -m "unit or contract or property or repo_check" -q

_step "pytest: integration"
uv run pytest -m integration -q

_step "pytest: benchmark (full suite -- accuracy floors, not just the PR smoke subset)"
uv run pytest -m benchmark -q

_step "Build sdist + wheel, twine check --strict"
rm -rf dist
uv build
uvx twine check --strict dist/*

_step "Package-data assertion"
# `grep -q` exits the instant it finds a match, which can SIGPIPE unzip if
# it's still writing -- with pipefail (set above), that signal becomes the
# pipeline's reported exit status even though grep DID find the match,
# spuriously reporting "missing" for content that's actually present. Not
# using -q (grep drains all of unzip's output before exiting) avoids it.
unzip -l dist/*.whl | grep 'sorethumb/py.typed' >/dev/null || _fail "py.typed missing from wheel"
unzip -l dist/*.whl | grep 'sorethumb/store/migrations/001_initial.sql' >/dev/null || _fail "migrations missing from wheel"
_ok "py.typed and migrations present in wheel"

_step "Clean-venv install smoke test (core, then each extra)"
WHL="$(ls dist/*.whl)"
TMP_VENVS="$(mktemp -d)"
trap 'rm -rf "$TMP_VENVS"' EXIT

python3 -m venv "$TMP_VENVS/core"
"$TMP_VENVS/core/bin/pip" install --upgrade pip -q
"$TMP_VENVS/core/bin/pip" install "$WHL" -q
"$TMP_VENVS/core/bin/sorethumb" --version
"$TMP_VENVS/core/bin/python" -c "
import sorethumb
from sorethumb import Config, SourceConfig, run_detection
import sorethumb.explain.shap_tree
import sorethumb.explain.gradient
import sorethumb.report.charts
print('core import OK', sorethumb.__version__)
"

for extra in explain report benchmark dev; do
  case "$extra" in
    explain)   smoke_import="import shap, numba" ;;
    report)    smoke_import="import matplotlib" ;;
    benchmark) smoke_import="import datasets, pandas" ;;
    dev)       smoke_import="import pytest, hypothesis, ruff, mypy" ;;
  esac
  venv="$TMP_VENVS/$extra"
  python3 -m venv "$venv"
  "$venv/bin/pip" install --upgrade pip -q
  "$venv/bin/pip" install "${WHL}[${extra}]" -q
  "$venv/bin/python" -c "
import sorethumb
${smoke_import}
print('$extra import OK', sorethumb.__version__)
"
done
_ok "core + every extra installs cleanly and imports"

WHEEL_VERSION="$(unzip -p "$WHL" '*.dist-info/METADATA' | grep -m1 '^Version:' | cut -d' ' -f2 | tr -d '\r')"
[[ "$WHEEL_VERSION" == "$VERSION" ]] || _fail "wheel metadata version ($WHEEL_VERSION) != requested version ($VERSION)"
_ok "wheel metadata version matches"

rm -rf dist
trap - EXIT
rm -rf "$TMP_VENVS"

# ---------------------------------------------------------------------------
# 5. Everything passed. Stop here for --dry-run; otherwise require an
#    explicit, typed confirmation before doing anything that pushes.
# ---------------------------------------------------------------------------

echo
echo "=========================================================================="
echo "All checks passed for $TAG at commit $LOCAL_SHA."
echo "=========================================================================="

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "--dry-run: stopping here. Nothing was tagged or pushed."
  exit 0
fi

echo
echo "About to run:"
echo "  gh release create $TAG --target $LOCAL_SHA --title \"$TAG\" --notes-file <CHANGELOG excerpt>"
echo
echo "This pushes a real tag, which triggers .github/workflows/publish.yml"
echo "against this exact commit. Actually reaching PyPI still needs your"
echo "separate manual approval in GitHub's UI (the pypi environment's"
echo "required-reviewers gate) -- but the tag itself, once pushed, is meant"
echo "to be permanent; don't push it if you're not ready to commit to that."
echo
read -r -p "Type the version ($VERSION) to confirm, or anything else to abort: " CONFIRM
if [[ "$CONFIRM" != "$VERSION" ]]; then
  echo "Aborted -- no tag created, nothing pushed."
  exit 1
fi

# ---------------------------------------------------------------------------
# 6. Tag + release, via gh.
# ---------------------------------------------------------------------------

_step "Creating and pushing $TAG via gh release create"

NOTES_FILE="$(mktemp)"
trap 'rm -f "$NOTES_FILE"' EXIT
echo "$CHANGELOG_BODY" >"$NOTES_FILE"

gh release create "$TAG" \
  --target "$LOCAL_SHA" \
  --title "$TAG" \
  --notes-file "$NOTES_FILE"

rm -f "$NOTES_FILE"
trap - EXIT

echo
echo "Pushed $TAG. Watch the publish workflow:"
echo "  gh run list --branch $TAG"
echo "  gh run watch <run-id>"
echo
echo "Remember: PyPI publication is paused on a required-reviewer approval"
echo "in the 'pypi' environment -- go approve it in GitHub's UI (Actions ->"
echo "the publish run -> Review deployments) when you're ready for it to"
echo "actually reach PyPI, not before."
