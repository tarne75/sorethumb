# Tests

How the suite is organised and how to run it. Counts, timings and coverage
figures are deliberately not recorded here: they go stale on the next commit,
and CI prints the current ones on every run.

## Lanes

Every test module declares its lane with a module-level `pytestmark`
(`--strict-markers` rejects undeclared names). A plain `pytest` runs only the
fast default lane; the other lanes are opt-in by marker.

| Marker | What it holds | In the default run? |
|---|---|---|
| `unit` | Fast, deterministic, network-free; no real SQLite, subprocess or model serialisation | yes |
| `contract` | Pins a public or persisted contract: public API, CLI exit codes and JSON, database schema, config schema, model manifests | yes |
| `integration` | Real workspace, SQLite store, CLI process, full pipeline run, model (de)serialisation, report rendering | no |
| `property` | Hypothesis property tests | no |
| `repo_check` | Repository consistency: generated docs, README structure, packaging and distribution artifacts, private-reference and review-document guards | no |
| `benchmark` | Accuracy measurement against synthetic and real datasets | no |
| `slow` | Otherwise in-lane but too slow for every run | no |
| `network` | Needs network access | no |

The default `-m` expression lives in `pyproject.toml` (`addopts`); passing your
own `-m` replaces it.

## Running them

```bash
uv run pytest                                  # default lane: unit + contract
uv run pytest -m integration                   # workflows against real SQLite and the CLI
uv run pytest -m "property or repo_check"
uv run pytest -m benchmark                     # accuracy floors; real-dataset tests skip offline
SORETHUMB_REQUIRE_NETWORK=1 uv run pytest -m benchmark   # ...or fail instead of skipping
```

CI (`.github/workflows/`) runs the same lanes: `ci.yml` for pull requests,
`release-validation.yml` (reusable) for the full pre-release set, and
`scripts/release.sh` locally before a tag. `ci.yml` also gates the changed lines
with diff coverage rather than a global coverage threshold, because a global
threshold rewarded padding tests over meaningful ones.

## Conventions

- Tests assert behaviour, not implementation detail: no assertion-free
  `importlib.reload` tests and no checking that a function was merely called.
- Assertions on versions, schema versions and migration lists read the single
  source of truth (package metadata, the bundled migrations) rather than a
  literal that changes at the next release.
- Shared builders live in `tests/factories/` (frames, workspaces, golden-file
  helper); use them instead of re-declaring fixtures per module.
- Golden files in `tests/golden/` are regenerated deliberately with
  `UPDATE_GOLDEN=1 uv run pytest <test>` and the diff reviewed like code.
- A test added for a bug fix must fail against the code without the fix.
