# Releasing, rehearsing, and rolling back

Maintainer-only. Run from macOS, Linux or WSL (`scripts/release.sh` is bash).

## Rehearse on TestPyPI first

**One-time setup:**

1. Create an account on [TestPyPI](https://test.pypi.org/account/register/) (separate from PyPI).
2. Create an account-scoped API token there (the project does not exist yet, so a
   project-scoped one cannot be made), and store it:
   `gh secret set TEST_PYPI_TOKEN --repo tarne75/sorethumb`
3. Create the environment, with you as its required reviewer so the run pauses for
   approval like the real release:

   ```bash
   echo "{\"reviewers\":[{\"type\":\"User\",\"id\":$(gh api user --jq .id)}]}" |
     gh api --method PUT repos/tarne75/sorethumb/environments/testpypi --input -
   ```

**Each rehearsal** publishes a pre-release of `pyproject.toml`'s version. Files on
TestPyPI are immutable, so use a new one every attempt (`0.1.0rc1`, `0.1.0rc2`, ...):

```bash
gh workflow run publish-testpypi.yml -f version=0.1.0rc1
```

The run pauses before the upload: approve the `testpypi` deployment (Actions → the
run → Review deployments). Then verify in a clean venv.
TestPyPI does not mirror the dependencies, so fetch only sorethumb from it and let pip
resolve the rest from PyPI:

```bash
python3 -m venv /tmp/rehearsal && . /tmp/rehearsal/bin/activate
pip download --no-deps -i https://test.pypi.org/simple/ sorethumb-ml==0.1.0rc1 -d /tmp/rehearsal-dist
pip install /tmp/rehearsal-dist/sorethumb_ml-0.1.0rc1-py3-none-any.whl
sorethumb --version
sorethumb --help
python -c "import sorethumb_ml"
deactivate && rm -rf /tmp/rehearsal /tmp/rehearsal-dist   # clean up afterwards
```

`sorethumb --version` must print `sorethumb 0.1.0rc1`, and
<https://test.pypi.org/project/sorethumb-ml/> must show the version. If anything
fails, fix it and rehearse again with the next `rcN`.

## Cut the release

**1. The release-state commit**, on the tag day (UTC):

- `CHANGELOG.md`: rename `## [Unreleased]` to `## [X.Y.Z] - YYYY-MM-DD`, add an empty
  `## [Unreleased]` above it, and set the footer links to
  `[Unreleased]: https://github.com/tarne75/sorethumb/compare/vX.Y.Z...HEAD` and
  `[X.Y.Z]: https://github.com/tarne75/sorethumb/releases/tag/vX.Y.Z`.
- `pyproject.toml`'s version is `X.Y.Z`.
- `SECURITY.md` and `README.md`: remove the pre-release wording (status banner, "Not yet
  on PyPI"), add the PyPI install and badge, and switch `blob/main` links to
  `blob/vX.Y.Z`. `scripts/release.sh`'s header lists each change.

`python scripts/check_release_state.py --version X.Y.Z` lists anything still missing.
Push the commit and let CI pass. If the UTC day changes before tagging, re-date it in a
new commit.

**2. Tag:** `scripts/release.sh X.Y.Z`. It reruns the release checks, builds and
smoke-tests the artifact, asks for confirmation, and pushes the `vX.Y.Z` tag.

**3. Publish:** `publish.yml` revalidates the tagged commit and pauses on the `pypi`
environment. Approve it (Actions → the run → Review deployments). Nothing reaches PyPI
without that approval.

## Roll back

PyPI files are immutable: a version can never be uploaded again, even after yanking.

- **Before the upload** (the run is waiting for approval): reject the deployment.
- **After it**: yank the release on pypi.org (the project's Manage page → the release →
  Yank), with a reason. Pinned installs still work; unpinned ones skip it. Then fix the
  bug, release a new patch version, and say in its CHANGELOG entry what was wrong.

Never delete or move a tag that has started a publish.
