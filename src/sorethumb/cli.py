"""CLI entry point.

Exposes the ``sorethumb`` command. At M0 only ``--version`` is wired; subsequent
milestones add commands to this app.
"""

import typer

import sorethumb

app = typer.Typer(
    name="sorethumb",
    no_args_is_help=True,
    rich_markup_mode="markdown",
    help="**sorethumb** — unsupervised anomaly detection for tabular data.",
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"sorethumb {sorethumb.__version__}")
        raise typer.Exit


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        callback=_version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    """Unsupervised anomaly detection for tabular data."""
