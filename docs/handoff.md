# Session handoff — 2026-09-28

Internal working note, not user-facing documentation (not linked from
`docs/index.md`). Records where a long phase-by-phase hardening session
left off, so a fresh session (human or Claude) can resume without
re-deriving context. Safe to delete once the remaining action-list items
are done and this stops being useful.

## What this project is

`sorethumb` (PyPI distribution name `sorethumb-ml`) is a Python anomaly-
detection library + CLI: config-driven feature encoding, an ensemble of
detectors (isolation_forest, kmeans_distance, one_class_svm, ecod, hbos,
lof), calibration, explanations, a SQLite-backed workspace/store, and a
report renderer. Not yet published to PyPI or tagged — `pyproject.toml` is
at `0.1.0`, unreleased.

## Where things stand right now

Working tree is clean; `main` is pushed and green through commit
`5b76b56` (`git log --oneline -20` for the recent list). Every commit in
this session's run of work follows the same pattern: implement, verify
locally (ruff/mypy/docs-check/full test suite), commit, push, poll CI
(`gh run watch`) until green, mark the item `(DONE)` with a detailed note.

**Two release-prep plans are fully closed** (nothing outstanding in
either):

- `prompts/pre-release-plan.md`
- `prompts/release-launch-plan.md` — distribution renamed to
  `sorethumb-ml`, `publish.yml` uses a `PYPI_TOKEN` (not OIDC) with a
  required-reviewer-gated `pypi` environment, `scripts/release.sh` is the
  only sanctioned way to cut a tag (dry-run first, always). See that
  file's own "Resuming this plan cold" section for the exact audit
  commands if verifying this claim.

**Current work is against `prompts/action-list-20260923.md`** — a
21-item, three-tier (P0/P1/P2) list from an automated code-review pass.
**This file is gitignored (local-only)** — it exists only on this
checkout's disk, not in git history. If it's ever missing, the fallback
record is `CHANGELOG.md`'s `### Fixed` entries (one per completed item,
written in detail) plus `git log` (one commit per item, same detail in
the commit body) — every completed item's *reasoning*, not just its
existence, is preserved in both places redundantly.

### Done: all 8 P0 items, all of P1-1..P1-4

| Item | One-line summary | Commit |
|---|---|---|
| P0-1 | Renamed PyPI distribution to `sorethumb-ml` (import name/CLI/repo unchanged) | `19cc320` + earlier |
| P0-2 | `workspace reset` can no longer delete unrelated data (path-safety guards) | `03ddeba` |
| P0-3 | Redirect credentials no longer leak cross-origin / downgrade to HTTP | `7248d98` |
| P0-4 | Replaced hand-built TOML string interpolation with a real serialiser | `c3caba4` |
| P0-5 | Fixed `_redact_config` clobbering live credentials via `os.environ` | `d235caa` |
| P0-6 | `score --from-run` rejects configs that lie about fit-time settings | `e6cdfe5` |
| P0-7 | One canonical `config_hash` everywhere (was silently two different values) | `858e70e` |
| P0-8 | Default ensemble ranking no longer worse-than-random from one bad detector | `06f85da` |
| P1-1 | Removed dead whole-dataset matrix kept alive per group; fixed memory estimate | `d1da049` |
| P1-2 | Zero-weight (dropped) detectors excluded from explanations | `1410693` |
| P1-3 | `columns.id_column` identity (existence/null/uniqueness) now validated | `c10e654` |
| P1-4 | Group selectors matching nothing now fail loudly, not silently complete | `5b76b56` |

Each has a full "Done 2026-09-2X: ..." paragraph in
`prompts/action-list-20260923.md` (root cause, exact fix, why that design
over alternatives, test list, verification) and a matching `CHANGELOG.md`
entry and commit message. If resuming and any of this seems off, `git log
--oneline -20` and `CHANGELOG.md`'s `### Fixed` section are the sources of
truth over this table.

### Not started: P1-5, P1-6, P1-7, all of P2 (6 items)

Full prompt text for each is in `prompts/action-list-20260923.md`;
one-line summaries:

- **P1-5** — HBOS clips out-of-range values into the edge bin instead of
  treating them as unseen (an arbitrarily distant value can score as
  normal if the edge bin is dense). Fix score application + native
  feature contributions; test dense edge bins, just-inside, exact-edge,
  and increasingly-distant out-of-range values, both direct scoring and
  score-forward.
- **P1-6** — CSV/NDJSON readers hard-code `infer_schema_length` then also
  expand user `read_options`, so the documented override
  (`read_options={"infer_schema_length": ...}`) passes the keyword twice
  and raises `TypeError`. Fix via `setdefault`; test the
  trigger-then-successfully-retry flow, compressed inputs, delimiter
  overrides, invalid option types.
- **P1-7** — HTTP source caching fully downloads+hashes before checking
  the cache, so a "cache hit" saves no network/latency/disk I/O. Add
  ETag/Last-Modified-aware conditional requests (304 reuse); handle
  changed validators, redirects, `cache=false`, no-validator servers;
  test 200→304 reuse, changed content, missing validators, redirect
  identity, interrupted downloads.
- **P2-1** — Tighten the benchmark harness's own honesty: assert the full
  expected (scenario, ablation) matrix, fix/rename `swamping` to match
  what it actually measures, report `anomaly_flag`-based metrics
  separately from ranking metrics (note: P0-8 already added
  `evaluate_flags`/`FlagMetrics` for exactly this — P2-1 is about using it
  more thoroughly + the real-dataset table + README claims), populate or
  remove the empty real-dataset table.
- **P2-2** — CI installs and smoke-tests the wheel but never the sdist.
  Extend `release-validation.yml` to build a clean env from the sdist too,
  same smoke checks, plus file-list inspection for both artifacts
  (py.typed, migrations, LICENSE/README present; no stray review/cache/
  benchmark files) and a wheel-vs-sdist version/metadata-agreement test.
- **P2-3** — README advertises macOS + Linux but integration tests only
  run on Linux. Either add a macOS integration CI job or narrow the
  advertised platform support; keep Windows unadvertised until it has its
  own dedicated job.
- **P2-4** — README still says "not on PyPI" and mixes old install
  instructions. One release-documentation commit updating every install
  command/badge/URL to the real `sorethumb-ml` identity, preferring
  tag-stable links.
- **P2-5** — The actual tag→PyPI publish path (trusted-publisher tuple,
  environment approval, OIDC, artifact hand-off) has never been rehearsed
  end-to-end. Do a disposable/TestPyPI dry run of the real workflow before
  the real `v0.1.0` tag. (Explicitly deferred earlier in this session —
  see `release-launch-plan.md` Item 1's follow-up note.)
- **P2-6** — `CHANGELOG.md`'s `[0.1.0]` section is currently a detailed
  implementation diary (deliberately, so this session's reasoning isn't
  lost — see above). Before release, rewrite it as concise user-facing
  release notes; the diary detail already lives in git history by then.

**P2-6 depends on P2-1..P2-5 substantively landing first** (it's meant to
summarize the finished state, not a moving target) — do it last. The
others have no ordering dependency on each other.

## How to resume

The established pattern for this session, one item at a time:

1. User says "let's do P1-5" (or whichever). Don't start P2-anything
   unprompted — P1-5/6/7 are the natural next three, but always follow
   what's actually asked.
2. Read the item's exact prompt text in `prompts/action-list-20260923.md`
   (grep for the item ID) — the one-liners above are summaries, not the
   full ask.
3. Investigate the actual code before trusting the audit's framing — this
   session found the audit's stated root cause was sometimes subtly wrong
   or already-fixed (e.g. P0-5's literal described bug didn't exist;
   P0-1's item had already been done before the list was reviewed). Grep,
   read, and where useful write a small throwaway repro script before
   deciding on a fix.
4. Implement, then verify locally before ever committing:
   ```bash
   uv run ruff check src/ tests/
   uv run ruff format --check src/ tests/
   uv run mypy src/
   uv run python docs/generate_config_docs.py --check
   uv run pytest tests/unit tests/contract tests/property tests/repo_check tests/integration -q -m "not benchmark"
   uv run pytest tests/benchmark -m benchmark -q   # slower (~2 min); run when the change touches scoring/detectors/pipeline
   ```
5. For a genuinely surprising/load-bearing fix, sanity-check the test
   actually catches the regression: temporarily revert the fix in place,
   confirm the new test fails, then restore it. Done repeatedly this
   session (P1-1, P1-2, P1-4) and worth the extra few minutes.
6. Update `CHANGELOG.md` (`### Fixed`, under the still-unreleased
   `[0.1.0]` section — newest entries go at the *top* of that section),
   mark the item `(DONE)` in `prompts/action-list-20260923.md` with a
   detailed note matching the style of the existing ones, commit (see
   below), push, then poll CI:
   ```bash
   gh run list --limit 3 --branch main
   gh run watch <run-id> --exit-status
   ```
   Only report an item done once CI is actually green.

## Standing rules (established this session, still binding)

- **Commit trailers**: every commit ends with
  `Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>` and a
  `Claude-Session:` line. This *replaces* an older "never add these"
  convention — the harness's own system reminder in-session mandates them
  now. Verify presence with `grep -i "co-authored-by\|claude-session"` on
  the drafted message before committing.
- Never push a git tag or attempt an actual PyPI publish without the user
  explicitly asking for that specific step, ever — `scripts/release.sh`
  exists precisely so this never happens by accident.
- `prompts/` is gitignored — local-only planning/tracking. Never shows up
  in `git status`, never gets committed. This is deliberate.
- Always create a new commit; never amend/force-push.
- One phase per user request — don't proactively start the next item.
- Prefer investigating and fixing the *real* underlying issue over the
  audit list's literal wording when they diverge (with a brief note in
  the completion writeup explaining the divergence) — happened for P0-5
  and P1-4 (both found a real, related bug in a different spot than
  described) and P1-4/P2-1's swamping note (an already-planned rename).

## Key files

- `prompts/action-list-20260923.md` — the authoritative task list + done-notes (gitignored).
- `prompts/pre-release-plan.md`, `prompts/release-launch-plan.md` — closed, reference only.
- `CHANGELOG.md` — `[Unreleased]` section is empty (correct, nothing shipped past `0.1.0` yet); all this session's work is entries under `## [0.1.0]`.
- `scripts/release.sh` — the only sanctioned release path; `--dry-run` first, always.
- `tests/benchmark/` — marked `benchmark`, deselected by default (`pytest -m benchmark` to run); several new files this session (`test_memory_budget.py`, plus extensions to `test_pipeline_accuracy_floors.py`).
