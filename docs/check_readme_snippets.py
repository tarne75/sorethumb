"""Execute / validate every fenced code block in README.md.

CI runs this so a snippet that stops being copy-paste runnable fails the build:

    python docs/check_readme_snippets.py

- ```python``` blocks are run in a subprocess from a fresh temp dir (nonzero
  exit fails). The 60-second quickstart generates its own small synthetic
  dataset in-process -- no network access, no dependency beyond this
  package's own core requirements.
- ```toml``` blocks must parse; a block whose top-level tables are config
  sections is merged onto a minimal base and fed to ``Config`` to catch schema
  drift.
- ```bash``` blocks: every ``sorethumb <subcommand>`` must be a real command
  (checked via ``--help``) and every ``--long-flag`` on that line must appear in
  its help text. ``pip install ".[extra]"`` must name a real optional-dependency
  group.

A second entry point extracts the lone ```python``` block verbatim, for
release-validation.yml's build job to run against a clean, extras-free wheel
install (the exact published snippet is executed as a required release
test) -- this dev-environment check alone can't stand in for that, since it
runs against the editable source checkout, not an installed distribution:

    python docs/check_readme_snippets.py --extract-python OUTPUT_PATH
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

_FENCE_RE = re.compile(r"^```(\w*)[ \t]*\n(.*?)^```[ \t]*$", re.DOTALL | re.MULTILINE)
_CONFIG_SECTIONS = {
    "source",
    "columns",
    "profiling",
    "features",
    "detectors",
    "scoring",
    "explain",
    "run",
    "history",
    "report",
}


def _blocks(readme: str) -> list[tuple[int, str, str]]:
    out: list[tuple[int, str, str]] = []
    for m in _FENCE_RE.finditer(readme):
        line_no = readme[: m.start()].count("\n") + 1
        out.append((line_no, m.group(1).lower(), m.group(2)))
    return out


def _check_python(body: str) -> str | None:
    with tempfile.TemporaryDirectory() as td:
        script = Path(td) / "snippet.py"
        script.write_text(body, encoding="utf-8")
        proc = subprocess.run(  # noqa: S603
            [sys.executable, str(script)],
            cwd=td,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=900,
            check=False,
        )
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-15:]
        return "python snippet exited " + str(proc.returncode) + ":\n    " + "\n    ".join(tail)
    return None


def _check_toml(body: str) -> str | None:
    try:
        data = tomllib.loads(body)
    except tomllib.TOMLDecodeError as exc:
        return f"toml snippet does not parse: {exc}"

    top = set(data)
    if not (top & _CONFIG_SECTIONS) or "project" in top or top - _CONFIG_SECTIONS - {"project"}:
        return None  # not a sorethumb config fragment (e.g. a pyproject entry-point block)

    from sorethumb_ml.config import Config  # noqa: PLC0415

    merged: dict = {"source": {"uri": "data.csv"}, "run": {"workdir": "."}}
    for k, v in data.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = {**merged[k], **v}
        else:
            merged[k] = v
    try:
        Config.model_validate(merged)
    except Exception as exc:  # noqa: BLE001
        return f"toml config fragment is not valid against the schema: {exc}"
    return None


def _command_options() -> dict[tuple[str, ...], set[str]]:
    """{(subcommand[, subsubcommand]): {long option strings}} from the live Typer app.

    Introspects the click command tree directly — parsing rendered ``--help`` is
    unreliable because rich truncates option names at narrow terminal widths.
    """
    import typer  # noqa: PLC0415

    from sorethumb_ml.cli import app  # noqa: PLC0415

    def _opts(cmd: object) -> set[str]:
        out: set[str] = set()
        for p in getattr(cmd, "params", []):
            out.update(o for o in (*p.opts, *p.secondary_opts) if o.startswith("--"))
        return out

    root = typer.main.get_command(app)
    table: dict[tuple[str, ...], set[str]] = {}
    for name, cmd in getattr(root, "commands", {}).items():
        sub = getattr(cmd, "commands", None)
        if sub:  # a group like `config` / `workspace`
            for subname, subcmd in sub.items():
                table[name, subname] = _opts(subcmd) | _opts(cmd)
        else:
            table[(name,)] = _opts(cmd)
    return table


def _check_bash(body: str) -> list[str]:
    problems: list[str] = []
    commands = _command_options()
    groups = {path[0] for path in commands if len(path) == 2}

    for raw in body.splitlines():
        line = raw.strip()
        if line.startswith(("#", "cd ", "git ", "jq ")) or not line:
            continue
        cmd = line.split("|", 1)[0].strip()  # drop pipes (| jq …)
        tokens = cmd.split()

        if tokens[:2] == ["pip", "install"]:
            for extra in re.findall(r"\.\[([a-z0-9_,\-]+)\]", cmd):
                for name in extra.split(","):
                    if not _extra_exists(name.strip()):
                        problems.append(f"pip extra [{name}] is not in pyproject optional-dependencies")
            continue

        if not tokens or tokens[0] != "sorethumb":
            continue

        args = [t for t in tokens[1:] if not t.startswith("-")]
        sub = args[0] if args else ""
        if sub in {"", "--version"}:
            continue
        # `config schema`, `workspace vacuum`, … are two-token subcommands.
        path = tuple(args[:2]) if sub in groups and len(args) > 1 else (sub,)
        if path not in commands:
            problems.append(f"`sorethumb {' '.join(path)}` is not a valid command")
            continue
        for flag in re.findall(r"(?<!\S)(--[a-z][a-z0-9-]+)", cmd):
            if flag not in commands[path]:
                problems.append(f"`sorethumb {' '.join(path)}` has no {flag} option")
    return problems


def _extra_exists(name: str) -> bool:
    pp = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return name in pp.get("project", {}).get("optional-dependencies", {})


def _extract_python(out_path: Path) -> None:
    """Write the lone ```python``` README block verbatim to *out_path*.

    Errors (rather than silently picking one) if README.md doesn't have
    exactly one -- a second block would mean whichever one a caller runs
    against a clean install is no longer obviously "the" published snippet.
    """
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    blocks = [body for _, lang, body in _blocks(readme) if lang == "python"]
    if len(blocks) != 1:
        print(f"ERROR: expected exactly one python block in README.md, found {len(blocks)}.")
        sys.exit(1)
    out_path.write_text(blocks[0], encoding="utf-8")


def main() -> None:
    """Check every README fenced block; exit non-zero on the first batch of problems."""
    if len(sys.argv) == 3 and sys.argv[1] == "--extract-python":
        _extract_python(Path(sys.argv[2]))
        return

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    problems: list[str] = []
    counts = {"python": 0, "toml": 0, "bash": 0}

    for line_no, lang, body in _blocks(readme):
        if lang == "python":
            counts["python"] += 1
            if (p := _check_python(body)) is not None:
                problems.append(f"README.md:{line_no}: {p}")
        elif lang == "toml":
            counts["toml"] += 1
            if (p := _check_toml(body)) is not None:
                problems.append(f"README.md:{line_no}: {p}")
        elif lang == "bash":
            counts["bash"] += 1
            problems += [f"README.md:{line_no}: {p}" for p in _check_bash(body)]

    if problems:
        print("ERROR: README.md snippets have problems:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)

    print(
        f"OK: {counts['python']} python, {counts['toml']} toml, {counts['bash']} bash "
        "README snippets checked."
    )


if __name__ == "__main__":
    main()
