"""Generate docs/configuration.md from the pydantic Config model.

Run from the repository root:
    python docs/generate_config_docs.py

CI checks that the committed file matches a fresh generation AND that the prose
docs do not drift from the schema (unknown `section.field` references, or a
stated default that no longer matches the model):
    python docs/generate_config_docs.py --check
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, get_args, get_origin

# Allow running without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sorethumb.config import (
    ColumnsConfig,
    DetectorConfig,
    ExplainConfig,
    FeaturesConfig,
    HistoryConfig,
    ProfilingConfig,
    ReportConfig,
    RunConfig,
    ScoringConfig,
    SourceConfig,
)

_SECTIONS: list[tuple[str, type, str]] = [
    ("source", SourceConfig, "Where the raw data lives and how to fetch it."),
    ("columns", ColumnsConfig, "Logical roles for specific columns."),
    ("profiling", ProfilingConfig, "Thresholds that control column classification."),
    ("features", FeaturesConfig, "Feature engineering options."),
    ("detectors", DetectorConfig, "Per-detector block (repeatable `[[detectors]]`)."),
    ("scoring", ScoringConfig, "How per-detector scores are combined."),
    ("explain", ExplainConfig, "SHAP-based anomaly explanation controls."),
    ("run", RunConfig, "Execution-level settings."),
    ("history", HistoryConfig, "Period-over-period baseline comparison."),
    ("report", ReportConfig, "Output report settings (cosmetic; excluded from config hash)."),
]


def _type_str(annotation: Any) -> str:
    """Render a Python type annotation as a readable string."""
    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is None:
        if hasattr(annotation, "__name__"):
            return annotation.__name__
        return str(annotation)

    if origin is type(None) or annotation is type(None):
        return "null"

    # Union / Optional
    import types  # noqa: PLC0415

    if origin is types.UnionType or str(origin) in ("typing.Union", "typing.Optional"):
        parts = [_type_str(a) for a in args if a is not type(None)]
        suffix = " | null" if type(None) in args else ""
        return " | ".join(parts) + suffix

    # Literal
    if str(origin) == "typing.Literal":
        return " | ".join(f'"{a}"' for a in args)

    # list, dict
    if origin is list:
        inner = _type_str(args[0]) if args else "any"
        return f"list[{inner}]"
    if origin is dict:
        k = _type_str(args[0]) if args else "str"
        v = _type_str(args[1]) if len(args) > 1 else "any"
        return f"dict[{k}, {v}]"

    return str(annotation)


def _default_str(field_info: Any) -> str:
    """Render a pydantic FieldInfo default as a readable string."""
    from pydantic_core import PydanticUndefined  # noqa: PLC0415

    if field_info.default is not PydanticUndefined:
        d = field_info.default
        if d is None:
            return "null"
        if isinstance(d, bool):
            return str(d).lower()
        if isinstance(d, str):
            return f'"{d}"'
        return str(d)
    if field_info.default_factory is not None:
        try:
            v = field_info.default_factory()
            if isinstance(v, list) and not v:
                return "[]"
            if isinstance(v, dict) and not v:
                return "{}"
            return str(v)
        except Exception:  # noqa: BLE001
            return "(computed)"
    return "**required**"


def _section_table(model: type, section: str) -> str:
    """Render one section's fields as a Markdown table."""
    rows = ["| Field | Type | Default | Description |", "| --- | --- | --- | --- |"]
    for name, field_info in model.model_fields.items():
        type_s = _type_str(field_info.annotation).replace("|", "\\|")
        default_s = _default_str(field_info)
        desc = (field_info.description or "").replace("|", "\\|").replace("\n", " ")
        rows.append(f"| `{section}.{name}` | {type_s} | {default_s} | {desc} |")
    return "\n".join(rows)


def _extra_params_section() -> list[str]:
    """Document the detector ``extra_params`` escape hatch.

    Enumerates the accepted keys per detector from the installed scikit-learn.
    """
    from sorethumb.detectors import registry  # noqa: PLC0415

    lines = [
        "## Detector `extra_params`",
        "",
        "A pass-through to the underlying scikit-learn estimator.",
        "",
        "`isolation_forest`, `lof`, `kmeans_distance` and `one_class_svm` each wrap a",
        "scikit-learn estimator. Any constructor argument the wrapper does not expose",
        "directly can be handed to it through a nested `extra_params` table:",
        "",
        "```toml",
        "[[detectors]]",
        'name = "isolation_forest"',
        "params = { n_estimators = 300, extra_params = { n_jobs = 4, max_features = 0.8 } }",
        "```",
        "",
        "Keys are validated when the detector is constructed, before any training runs:",
        "",
        "- Keys sorethumb manages itself — `random_state`, `contamination`, `novelty`,",
        "  `n_clusters` — are rejected. Use `run.seed`, `scoring.contamination`, or the",
        "  detector's own `k` instead.",
        "- Keys already exposed as a wrapper argument (`n_estimators`, `nu`, `n_neighbors`,",
        "  `n_init`, …) are rejected, so there is one unambiguous source.",
        "- Any other key the estimator does not accept is rejected up front, with the",
        "  estimator's full parameter list in the error message.",
        "",
        "`ecod` and `hbos` have no underlying estimator and reject any non-empty",
        "`extra_params`.",
        "",
        "### Accepted keys",
        "",
        "Generated from the installed scikit-learn; the exact set may shift between",
        "scikit-learn releases.",
        "",
    ]
    for name in ("isolation_forest", "lof", "kmeans_distance", "one_class_svm"):
        cls = registry.get(name)
        getter = getattr(cls, "available_extra_params", None)
        if not callable(getter):
            continue
        lines += [f"#### `{name}`", "", "| Key | scikit-learn default |", "| --- | --- |"]
        lines += [f"| `{k}` | `{v!r}` |" for k, v in getter().items()]
        lines.append("")
    return lines


def generate() -> str:
    """Return the full configuration.md content."""
    lines = [
        "# Configuration reference",
        "",
        "> **Auto-generated** from `src/sorethumb/config.py` by `docs/generate_config_docs.py`.",
        "> Do not edit manually — run `python docs/generate_config_docs.py` to regenerate.",
        "",
        "sorethumb is configured through a single TOML file (default: `sorethumb.toml`).",
        "A config file is optional — passing a data file directly to `sorethumb run`",
        'uses all defaults with `workdir = "."` and prompts to save a config on first run:',
        "",
        "```bash",
        "sorethumb run /path/to/data.parquet",
        "```",
        "",
        "Run `sorethumb init` to create a fully commented starter file with every option documented.",
        "Run `sorethumb config schema` to emit the JSON schema.",
        "For scenario-based TOML snippets see [configuration-examples.md](configuration-examples.md).",
        "",
        "## Resolution order",
        "",
        "1. `sorethumb.toml` (or `--config PATH`)",
        "2. Environment variables with the `SORETHUMB_` prefix",
        "3. Command-line flags (highest priority)",
        "",
        "## Config hash",
        "",
        "`Config.config_hash()` is a 32-character hex digest that covers all",
        "result-affecting fields. Cosmetic or execution-only fields (`run.workdir`,",
        "`run.log_level`, `run.slow_stage_seconds`, `run.reuse_models`,",
        "`source.dataset_id`, and the entire `[report]` section) are excluded",
        "so trivial changes do not invalidate cached artefacts.",
        "",
    ]

    for section, model, description in _SECTIONS:
        lines += [
            f"## `[{section}]` — {description}",
            "",
            _section_table(model, section),
            "",
        ]

    lines += _extra_params_section()

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prose-doc drift check (extends the check beyond configuration.md)
# ---------------------------------------------------------------------------

# Docs (relative to the repo root) whose prose is scanned for schema drift.
_PROSE_DOCS = [
    "README.md",
    "docs/adapting-to-your-data.md",
    "docs/configuration-examples.md",
    "docs/example-runs.md",
    "docs/models.md",
    "docs/index.md",
]

_SECTION_NAMES = {name for name, _model, _desc in _SECTIONS}

# `section.field` inside backticks, e.g. `profiling.null_ratio_flag`.
_FIELD_REF_RE = re.compile(r"`([a-z_]+)\.([a-z_][a-z0-9_]*)`")
# `section.field` … default[s to] <value>   (value: number / quoted / bare word)
_DEFAULT_CLAIM_RE = re.compile(
    r"`([a-z_]+)\.([a-z_][a-z0-9_]*)`[^.\n]{0,90}?"
    r"default(?:s)?(?:\s+(?:to|is))?[\s:=]+\s*[`\"']?([A-Za-z0-9_.+\-]+)[`\"']?",
    re.IGNORECASE,
)
_FENCE_RE = re.compile(r"^```.*?^```", re.DOTALL | re.MULTILINE)


def _model_defaults() -> dict[str, dict[str, Any]]:
    """{section: {field: default_value}} for every non-required field."""
    from pydantic_core import PydanticUndefined  # noqa: PLC0415

    out: dict[str, dict[str, Any]] = {}
    for section, model, _desc in _SECTIONS:
        fields: dict[str, Any] = {}
        for name, info in model.model_fields.items():
            if info.default is not PydanticUndefined:
                fields[name] = info.default
            elif info.default_factory is not None:
                try:
                    fields[name] = info.default_factory()  # type: ignore[call-arg]
                except Exception:  # noqa: BLE001, S112
                    continue
        out[section] = fields
    return out


def _canon(value: Any) -> Any:
    """Normalise a default (schema or doc-claimed) for comparison."""
    s = str(value).strip().strip("`\"'").lower()
    if s in {"none", "null"}:
        return "none"
    if s in {"true", "false"}:
        return s
    try:
        return float(s)
    except ValueError:
        return s


def check_doc_drift(root: Path) -> list[str]:
    """Return a list of drift problems found in the prose docs (empty = clean)."""
    defaults = _model_defaults()
    problems: list[str] = []

    for rel in _PROSE_DOCS:
        path = root / rel
        if not path.exists():
            problems.append(f"{rel}: listed for drift-checking but the file is missing")
            continue
        prose = _FENCE_RE.sub("", path.read_text(encoding="utf-8"))

        for section, field in _FIELD_REF_RE.findall(prose):
            if section in _SECTION_NAMES and field not in defaults.get(section, {}):
                # Skip `run.workdir` etc. — required fields are not in `defaults`.
                model = next(m for n, m, _ in _SECTIONS if n == section)
                if field not in model.model_fields:
                    problems.append(f"{rel}: `{section}.{field}` is not a field of [{section}]")

        for section, field, claimed in _DEFAULT_CLAIM_RE.findall(prose):
            if section not in defaults or field not in defaults[section]:
                continue
            actual = defaults[section][field]
            if isinstance(actual, (list, dict)):
                continue  # doc claims about collection defaults are not scanned
            if _canon(claimed) != _canon(actual):
                problems.append(
                    f"{rel}: `{section}.{field}` documented default {claimed!r} "
                    f"but the schema default is {actual!r}"
                )

    return problems


_LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


def check_broken_links(root: Path) -> list[str]:
    """Return relative Markdown links (in any tracked .md) whose target is missing."""
    problems: list[str] = []
    md_files = sorted(root.glob("*.md")) + sorted((root / "docs").glob("*.md"))
    for md in md_files:
        for target in _LINK_RE.findall(md.read_text(encoding="utf-8")):
            dest = target.split("#", 1)[0].strip()
            if not dest or dest.startswith(("http://", "https://", "mailto:")):
                continue
            if not (md.parent / dest).resolve().exists():
                problems.append(f"{md.relative_to(root)}: link -> {dest} (target does not exist)")
    return problems


def main() -> None:
    """Entry point: write or check docs/configuration.md (+ prose drift on --check)."""
    root = Path(__file__).parent.parent
    out_path = Path(__file__).parent / "configuration.md"
    content = generate()

    if "--check" in sys.argv:
        failed = False
        if not out_path.exists():
            print("ERROR: docs/configuration.md does not exist. Run: python docs/generate_config_docs.py")
            failed = True
        elif out_path.read_text(encoding="utf-8") != content:
            print(
                "ERROR: docs/configuration.md is out of date.\n"
                "Run: python docs/generate_config_docs.py\n"
                "Then commit the updated file."
            )
            failed = True
        else:
            print("OK: docs/configuration.md is up to date.")

        drift = check_doc_drift(root)
        if drift:
            print("\nERROR: prose docs have drifted from src/sorethumb/config.py:")
            for p in drift:
                print(f"  - {p}")
            failed = True
        else:
            print(f"OK: no schema drift in {len(_PROSE_DOCS)} prose docs.")

        links = check_broken_links(root)
        if links:
            print("\nERROR: broken relative Markdown links:")
            for p in links:
                print(f"  - {p}")
            failed = True
        else:
            print("OK: no broken relative Markdown links.")

        sys.exit(1 if failed else 0)
    else:
        out_path.write_text(content, encoding="utf-8")
        print(f"Written: {out_path}")


if __name__ == "__main__":
    main()
