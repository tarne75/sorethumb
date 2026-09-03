"""Generate docs/configuration.md from the pydantic Config model.

Run from the repository root:
    python docs/generate_config_docs.py

CI checks that the committed file matches a fresh generation:
    python docs/generate_config_docs.py --check
"""

from __future__ import annotations

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


def generate() -> str:
    """Return the full configuration.md content."""
    lines = [
        "# Configuration reference",
        "",
        "> **Auto-generated** from `src/sorethumb/config.py` by `docs/generate_config_docs.py`.",
        "> Do not edit manually — run `python docs/generate_config_docs.py` to regenerate.",
        "",
        "sorethumb is configured through a single TOML file (default: `sorethumb.toml`).",
        "Run `sorethumb init` to create a fully commented starter file.",
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
        "`Config.config_hash()` is a 16-character hex digest that covers all",
        "result-affecting fields. Cosmetic fields (`run.log_level`,",
        "`run.slow_stage_seconds`, and the entire `[report]` section) are excluded",
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

    return "\n".join(lines)


def main() -> None:
    """Entry point: write or check docs/configuration.md."""
    out_path = Path(__file__).parent / "configuration.md"
    content = generate()

    if "--check" in sys.argv:
        if not out_path.exists():
            print("ERROR: docs/configuration.md does not exist. Run: python docs/generate_config_docs.py")
            sys.exit(1)
        committed = out_path.read_text(encoding="utf-8")
        if committed != content:
            print(
                "ERROR: docs/configuration.md is out of date.\n"
                "Run: python docs/generate_config_docs.py\n"
                "Then commit the updated file."
            )
            sys.exit(1)
        print("OK: docs/configuration.md is up to date.")
    else:
        out_path.write_text(content, encoding="utf-8")
        print(f"Written: {out_path}")


if __name__ == "__main__":
    main()
