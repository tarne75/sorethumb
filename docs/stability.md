# API and stability policy (pre-1.0)

sorethumb is `0.x`. This page says which parts of it are stable enough to build on, and
what a release is allowed to change.

## What is the supported surface

- **The top-level `sorethumb_ml` namespace.** The names in `sorethumb_ml.__all__`:
  `run_detection`, `score_forward`, `render_report_for_run`, `load_dataset`,
  `build_feature_plan`, `apply_feature_plan`, `list_detectors`, `evaluate_scores`, the
  types `Config`, `SourceConfig`, `RunResult`, `GroupSummary`, `FeaturePlan`,
  `FeatureSpace`, `Metrics`, `Workspace`, `SorethumbError`, the `Detector` protocol for
  third-party detectors, and `__version__`. A contract test pins this list exactly, so a
  name cannot appear or disappear without the change being deliberate.
- **The command line.** Command names, options, the five exit codes and the JSON output
  shapes documented in the [CLI reference](cli_reference.md).
- **The configuration schema** documented in the [configuration reference](configuration.md).
- **The workspace on disk.** The database carries numbered, checksummed migrations, so an
  older workspace is upgraded when a newer sorethumb opens it, and an older sorethumb
  refuses to open a workspace that a newer one has already migrated.

## What is not

Everything else is internal, even where it can be imported: underscore-prefixed modules
(`sorethumb_ml._pipeline`, for instance) and the contents of the sub-packages
(`sorethumb_ml.detectors`, `sorethumb_ml.scoring`, `sorethumb_ml.store`,
`sorethumb_ml.explain`, ...). The sub-packages declare an `__all__` and tests pin it, but
that is to make changes deliberate, not a promise to keep it. Import from the top level, or
tell us what you need that is missing from it.

## What a release may change before 1.0

Pre-1.0, a **minor** version (`0.1` to `0.2`) may change or remove anything above, including
the supported surface. A **patch** version (`0.1.0` to `0.1.1`) is for fixes and does not
break the supported surface. There is no deprecation period before 1.0: a change that
affects the supported surface is listed under *Changed* or *Removed* in the
[CHANGELOG](https://github.com/tarne75/sorethumb/blob/main/CHANGELOG.md), which is the
place to read before upgrading.

After 1.0 this page will be replaced by a semantic-versioning policy with a deprecation
window for the supported surface.

## Changing the surface (maintainers)

Edit `_EXPECTED_TOP_LEVEL_EXPORTS` in `tests/contract/test_public_api.py` and the
package docstring in the same commit as the change to `__all__`, and add a CHANGELOG entry.
