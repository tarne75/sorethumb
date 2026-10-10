"""Source provenance: query values, userinfo and fragments never reach disk, logs, reports or errors.

An https source is served by an ``httpx.MockTransport`` so the *full* URI really goes
over the (fake) wire -- the guarantee is that it goes nowhere else. Every secret in
``tests/factories/source_uris.py`` is hunted for in the whole workspace (every SQLite
value and every file), the captured log, the CLI output and any raised error, while the
cache and identity behaviour stays deterministic.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import traceback
from pathlib import Path
from urllib.parse import urlsplit

import httpx
import pytest
from typer.testing import CliRunner

from sorethumb_ml import Config, score_forward
from sorethumb_ml._pipeline import run_detection
from sorethumb_ml.cli import _run_result_to_dict, app
from sorethumb_ml.config import SourceConfig
from sorethumb_ml.errors import ConfigError, SourceError
from sorethumb_ml.io import source
from sorethumb_ml.io.fingerprint import logical_dataset_id
from sorethumb_ml.io.uri import display_source_uri, source_digest
from sorethumb_ml.store.db import Store
from sorethumb_ml.store.workspace import Workspace
from tests.factories.configs import make_config
from tests.factories.frames import write_planted_csv
from tests.factories.source_uris import SIGNED_URI_CASES

pytestmark = pytest.mark.integration

runner = CliRunner()
CASE_IDS = list(SIGNED_URI_CASES)
BASE = "https://data.example.com/export/t.csv"


class FakeServer:
    """Serves one CSV for any https URL, recording the raw URL of every request."""

    def __init__(self, csv_bytes: bytes) -> None:
        self.csv_bytes = csv_bytes
        self.urls: list[str] = []
        self.mode = "ok"

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.urls.append(str(request.url))
        if self.mode == "forbidden":
            return httpx.Response(403, text="nope")
        if self.mode == "server_error":
            return httpx.Response(500, text="boom")
        if self.mode == "connect_error":
            raise httpx.ConnectError(f"cannot connect to {request.url}", request=request)
        return httpx.Response(200, content=self.csv_bytes, headers={"etag": '"v1"'})


@pytest.fixture
def server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeServer:
    csv_path = tmp_path / "served.csv"
    write_planted_csv(csv_path, n_normal=120, n_anomaly=4, seed=0)
    srv = FakeServer(csv_path.read_bytes())
    monkeypatch.setattr(source.httpx, "HTTPTransport", lambda **_kw: httpx.MockTransport(srv.handler))
    monkeypatch.setattr(source, "_assert_host_is_safe", lambda _url: None)
    monkeypatch.setattr(source.time, "sleep", lambda _s: None)
    return srv


def _config(tmp_path: Path, uri: str, *, name: str = "ws", dataset_id: str | None = None) -> Config:
    cfg = make_config(tmp_path / "unused.csv", tmp_path / name)
    return cfg.model_copy(
        update={"source": SourceConfig(uri=uri, format="csv", cache=True, dataset_id=dataset_id)}
    )


def _database_text(workdir: Path) -> str:
    """Every value of every column of every table in the workspace database."""
    conn = sqlite3.connect(str(workdir / "sorethumb.db"))
    try:
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        chunks = []
        for table in tables:
            for row in conn.execute(f'SELECT * FROM "{table}"'):
                chunks.extend(str(v) for v in row)
        return "\n".join(chunks)
    finally:
        conn.close()


def _file_blobs(workdir: Path) -> dict[str, bytes]:
    return {str(p.relative_to(workdir)): p.read_bytes() for p in workdir.rglob("*") if p.is_file()}


def assert_clean(secrets: list[str], *, workdir: Path, texts: dict[str, str]) -> None:
    for secret in secrets:
        assert secret not in _database_text(workdir), f"{secret!r} in a database value"
        for rel, blob in _file_blobs(workdir).items():
            assert secret.encode() not in blob, f"{secret!r} in workspace file {rel}"
        for label, text in texts.items():
            assert secret not in text, f"{secret!r} in {label}"


# ── persisted state, logs, reports, results ───────────────────────────────────


@pytest.mark.parametrize("name", CASE_IDS)
def test_secrets_absent_from_everything_a_run_writes_or_returns(
    name: str, tmp_path: Path, server: FakeServer, caplog: pytest.LogCaptureFixture
) -> None:
    uri, secrets = SIGNED_URI_CASES[name]
    caplog.set_level(logging.DEBUG)
    cfg = _config(tmp_path, uri)

    result = run_detection(cfg)

    assert result.n_succeeded == 1
    assert result.report_status == "success"
    assert server.urls, "the fake server was never asked for the data"
    if "?" in uri and "#" not in uri and "@" not in urlsplit(uri).netloc:
        assert urlsplit(uri).query == urlsplit(server.urls[0]).query, "the full query must reach the request"

    shown = display_source_uri(uri)
    with Workspace.open(tmp_path / "ws") as ws:
        run_row = ws.store.get_run(result.run_id)
        dataset_row = ws.store.get_dataset(result.dataset_fp)
    assert run_row is not None
    assert dataset_row is not None
    assert dataset_row["source_uri"] == shown
    assert json.loads(run_row["config_json"])["source"]["uri"] == shown
    assert dataset_row["source_digest"] == run_row["source_digest"] == source_digest(uri)
    assert result.dataset_uri == shown
    assert result.source_digest == source_digest(uri)

    report_text = Path(result.report_path).read_text(encoding="utf-8") if result.report_path else ""
    assert report_text, "no report was generated"
    assert shown.replace("&", "&amp;") in report_text or shown in report_text

    assert_clean(
        secrets,
        workdir=tmp_path / "ws",
        texts={
            "log": caplog.text,
            "RunResult repr": repr(result),
            "run --json payload": json.dumps(_run_result_to_dict(result)),
            "report": report_text,
        },
    )


@pytest.mark.usefixtures("server")
@pytest.mark.parametrize("name", ["client_secret", "aws_sigv4", "azure_sas"])
def test_secrets_absent_after_score_forward_with_a_refreshed_url(
    name: str, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    uri, secrets = SIGNED_URI_CASES[name]
    refreshed = uri.replace(secrets[0], "REFRESHEDVALUE0000")
    assert refreshed != uri
    caplog.set_level(logging.DEBUG)

    first = run_detection(_config(tmp_path, uri), no_report=True)
    scored = score_forward(_config(tmp_path, refreshed), first.run_id, no_report=True)

    assert scored.n_succeeded == 1
    assert scored.dataset_fp == first.dataset_fp  # a refresh keeps the dataset's history
    assert scored.source_digest == source_digest(refreshed) != first.source_digest
    assert_clean(
        [*secrets, "REFRESHEDVALUE0000"],
        workdir=tmp_path / "ws",
        texts={"log": caplog.text, "payload": json.dumps(_run_result_to_dict(scored))},
    )


@pytest.mark.usefixtures("server")
def test_dry_run_result_carries_only_the_display_uri(tmp_path: Path) -> None:
    uri, secrets = SIGNED_URI_CASES["jwt"]
    result = run_detection(_config(tmp_path, uri), dry_run=True)
    assert result.dataset_uri == display_source_uri(uri)
    assert_clean(secrets, workdir=tmp_path / "ws", texts={"repr": repr(result)})


# ── CLI surfaces ──────────────────────────────────────────────────────────────


def _write_toml(tmp_path: Path, uri: str) -> Path:
    toml = tmp_path / "sorethumb.toml"
    toml.write_text(
        f"""\
[source]
uri = {json.dumps(uri)}
format = "csv"

[run]
workdir = {json.dumps(str(tmp_path / "ws"))}
seed = 0

[columns]
id_column = "id"

[[detectors]]
name = "isolation_forest"
""",
        encoding="utf-8",
    )
    return toml


@pytest.mark.usefixtures("server")
@pytest.mark.parametrize("name", ["client_secret", "google_v4", "userinfo"])
def test_cli_run_show_and_report_output_never_contain_the_secrets(name: str, tmp_path: Path) -> None:
    uri, secrets = SIGNED_URI_CASES[name]
    toml = _write_toml(tmp_path, uri)

    ran = runner.invoke(app, ["run", "--config", str(toml), "--json", "--log-level", "DEBUG"])
    assert ran.exit_code == 0, ran.output
    payload = json.loads(ran.output[ran.output.index("{") :])
    assert payload["dataset_uri"] == display_source_uri(uri)
    assert payload["source_digest"] == source_digest(uri)

    shown = runner.invoke(app, ["show", "--config", str(toml), "--run-id", payload["run_id"]])
    reported = runner.invoke(app, ["report", "--config", str(toml), "--run-id", payload["run_id"]])
    assert_clean(
        secrets,
        workdir=tmp_path / "ws",
        texts={"run output": ran.output, "show output": shown.output, "report output": reported.output},
    )


# ── raised errors ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("mode", ["forbidden", "server_error", "connect_error"])
@pytest.mark.parametrize("name", CASE_IDS)
def test_secrets_absent_from_download_errors_and_their_tracebacks(
    name: str, mode: str, tmp_path: Path, server: FakeServer, caplog: pytest.LogCaptureFixture
) -> None:
    uri, secrets = SIGNED_URI_CASES[name]
    server.mode = mode
    caplog.set_level(logging.DEBUG)

    with pytest.raises(SourceError) as excinfo:
        run_detection(_config(tmp_path, uri), no_report=True)

    rendered = "".join(traceback.format_exception(excinfo.value))
    assert_clean(secrets, workdir=tmp_path / "ws", texts={"exception": rendered, "log": caplog.text})
    assert display_source_uri(uri) in str(excinfo.value)


def test_unsupported_scheme_error_is_redacted(tmp_path: Path) -> None:
    secret = "ftp-secret-value-1"
    with pytest.raises((SourceError, ConfigError)) as excinfo:
        source.resolve_source(
            SourceConfig.model_construct(uri=f"ftp://h/p.csv?token={secret}", format="csv"), tmp_path
        )
    assert secret not in "".join(traceback.format_exception(excinfo.value))


def test_a_redirect_to_a_presigned_url_leaks_nothing(
    tmp_path: Path, server: FakeServer, caplog: pytest.LogCaptureFixture
) -> None:
    presigned = "https://bucket.s3.example.com/t.csv?X-Amz-Signature=REDIRSIG42&X-Amz-Credential=REDIRCRED42"
    original = server.handler

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "api.example.com":
            server.urls.append(str(request.url))
            return httpx.Response(302, headers={"location": presigned})
        return original(request)

    server.handler = handler  # type: ignore[method-assign]
    caplog.set_level(logging.DEBUG)

    result = run_detection(_config(tmp_path, "https://api.example.com/export.csv?api_token=APITOKEN42"))

    assert result.n_succeeded == 1
    assert presigned in server.urls, "the redirect was not followed to the presigned URL"
    assert_clean(
        ["REDIRSIG42", "REDIRCRED42", "APITOKEN42"],
        workdir=tmp_path / "ws",
        texts={"log": caplog.text, "payload": json.dumps(_run_result_to_dict(result))},
    )


@pytest.mark.parametrize("mode", ["forbidden", "connect_error"])
def test_an_error_after_a_redirect_to_a_presigned_url_leaks_nothing(
    tmp_path: Path, server: FakeServer, mode: str, caplog: pytest.LogCaptureFixture
) -> None:
    presigned = "https://bucket.s3.example.com/t.csv?X-Amz-Signature=REDIRSIG43"
    original = server.handler

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "api.example.com":
            return httpx.Response(302, headers={"location": presigned})
        return original(request)

    server.handler = handler  # type: ignore[method-assign]
    server.mode = mode
    caplog.set_level(logging.DEBUG)

    with pytest.raises(SourceError) as excinfo:
        run_detection(_config(tmp_path, "https://api.example.com/export.csv"), no_report=True)

    rendered = "".join(traceback.format_exception(excinfo.value))
    assert "REDIRSIG43" not in rendered
    assert "REDIRSIG43" not in caplog.text


@pytest.mark.usefixtures("server")
def test_httpx_request_log_line_is_scrubbed(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    uri, secrets = SIGNED_URI_CASES["aws_sigv4"]
    source.resolve_source(SourceConfig(uri=uri, format="csv", cache=False), tmp_path)
    request_lines = [r.getMessage() for r in caplog.records if r.name == "httpx"]
    assert request_lines, "httpx did not log the request (the scrub filter would be untested)"
    assert not any(secret in line for line in request_lines for secret in secrets)
    assert any("X-Amz-Signature=REDACTED" in line for line in request_lines)


# ── cache and identity stay deterministic ─────────────────────────────────────


@pytest.mark.usefixtures("server")
def test_cache_index_is_keyed_by_a_hash_of_the_full_url_and_holds_no_secret(tmp_path: Path) -> None:
    uri, secrets = SIGNED_URI_CASES["azure_sas"]
    cache_dir = tmp_path / "cache"
    first = source.resolve_source(SourceConfig(uri=uri, format="csv", cache=True), cache_dir)
    second = source.resolve_source(SourceConfig(uri=uri, format="csv", cache=True), cache_dir)

    assert first == second
    metas = sorted((cache_dir / ".http_meta").glob("*.json"))
    assert [m.name for m in metas] == [f"{hashlib.sha256(uri.encode()).hexdigest()}.json"]
    assert source._cache_meta_path(cache_dir, uri) == metas[0]
    for secret in secrets:
        assert not any(secret.encode() in p.read_bytes() for p in cache_dir.rglob("*") if p.is_file())


@pytest.mark.usefixtures("server")
def test_same_uri_gives_same_identity_and_a_resumed_run(tmp_path: Path) -> None:
    uri = SIGNED_URI_CASES["google_v4"][0]
    a = run_detection(_config(tmp_path, uri), no_report=True)
    b = run_detection(_config(tmp_path, uri), no_report=True)
    assert (a.dataset_fp, a.run_id, a.config_hash, a.source_digest) == (
        b.dataset_fp,
        b.run_id,
        b.config_hash,
        b.source_digest,
    )


@pytest.mark.usefixtures("server")
def test_a_signature_refresh_stays_one_dataset_with_a_distinct_digest(tmp_path: Path) -> None:
    old = f"{BASE}?X-Amz-Date=20260101&X-Amz-Signature=aaaa1111"
    new = f"{BASE}?X-Amz-Date=20260202&X-Amz-Signature=bbbb2222"
    a = run_detection(_config(tmp_path, old), no_report=True)
    b = run_detection(_config(tmp_path, new), no_report=True)

    assert a.dataset_fp == b.dataset_fp
    assert a.source_digest != b.source_digest
    with Workspace.open(tmp_path / "ws") as ws:
        n_datasets = ws.store._conn.execute("SELECT COUNT(*) FROM dataset").fetchone()[0]
        digest = ws.store.get_dataset(a.dataset_fp)["source_digest"]  # type: ignore[index]
    assert n_datasets == 1
    assert digest == b.source_digest  # the dataset row tracks the latest source


@pytest.mark.usefixtures("server")
def test_changed_query_key_set_is_refused_without_a_dataset_id(tmp_path: Path) -> None:
    run_detection(_config(tmp_path, f"{BASE}?table=sales"), no_report=True)

    with pytest.raises(ConfigError, match=r"source\.dataset_id") as excinfo:
        run_detection(_config(tmp_path, f"{BASE}?report=costs-value-77"), no_report=True)

    assert "costs-value-77" not in str(excinfo.value)
    assert "sales" not in str(excinfo.value).replace("table", "")  # key names only, never values


@pytest.mark.usefixtures("server")
def test_the_suggested_dataset_id_keeps_the_existing_history(tmp_path: Path) -> None:
    first = run_detection(_config(tmp_path, f"{BASE}?table=sales"), no_report=True)

    with pytest.raises(ConfigError) as excinfo:
        run_detection(_config(tmp_path, f"{BASE}?table=sales&format=csv"), no_report=True)
    assert f'source.dataset_id = "{first.dataset_fp}"' in str(excinfo.value)

    again = run_detection(
        _config(tmp_path, f"{BASE}?table=sales&format=csv", dataset_id=first.dataset_fp), no_report=True
    )
    assert again.dataset_fp == first.dataset_fp
    with Workspace.open(tmp_path / "ws") as ws:
        n_runs = ws.store._conn.execute(
            "SELECT COUNT(*) FROM run WHERE dataset_fp = ?", (first.dataset_fp,)
        ).fetchone()[0]
    assert n_runs == 2


@pytest.mark.usefixtures("server")
@pytest.mark.parametrize(
    "stored_by_old_version",
    [
        # Old redaction re-encoded keys through urlencode: these no longer match the
        # key names read from the unchanged config, but the source has not changed.
        f"{BASE}?filter%5B0%5D=a&sig=REDACTED",
        f"{BASE}?download=&sig=REDACTED",
    ],
)
def test_a_dataset_row_from_before_migration_009_is_not_checked(
    tmp_path: Path, stored_by_old_version: str
) -> None:
    current = f"{BASE}?filter[0]=a&sig=b" if "filter" in stored_by_old_version else f"{BASE}?download&sig=b"
    cfg = _config(tmp_path, current)
    dataset_fp = logical_dataset_id(None, current)
    with Workspace.init(tmp_path / "ws") as ws:
        ws.store.upsert_dataset(
            dataset_fp=dataset_fp,
            source_uri=stored_by_old_version,
            schema_fingerprint="s",
            content_fingerprint="c",
            n_rows=1,
            n_cols=1,
        )  # no source_digest: what migration 009 leaves on an upgraded row

    result = run_detection(cfg, no_report=True)

    assert result.dataset_fp == dataset_fp
    with Workspace.open(tmp_path / "ws") as ws:
        row = ws.store.get_dataset(dataset_fp)
    assert row is not None
    assert row["source_digest"] == source_digest(current)  # checked from the next run on


@pytest.mark.usefixtures("server")
def test_dataset_id_keeps_differently_parameterised_sources_apart(tmp_path: Path) -> None:
    a = run_detection(_config(tmp_path, f"{BASE}?table=sales", dataset_id="sales"), no_report=True)
    b = run_detection(_config(tmp_path, f"{BASE}?report=costs", dataset_id="costs"), no_report=True)
    assert a.dataset_fp == "sales"
    assert b.dataset_fp == "costs"
    assert a.source_digest != b.source_digest


@pytest.mark.usefixtures("server")
def test_values_only_difference_is_distinguished_by_digest_and_run_id_when_dataset_id_is_set(
    tmp_path: Path,
) -> None:
    a = run_detection(_config(tmp_path, f"{BASE}?table=sales", dataset_id="d"), no_report=True)
    b = run_detection(_config(tmp_path, f"{BASE}?table=costs", dataset_id="d"), no_report=True)
    assert a.dataset_fp == b.dataset_fp == "d"
    assert a.source_digest != b.source_digest
    assert a.config_hash != b.config_hash
    assert a.run_id != b.run_id


# ── pre-existing workspaces are scrubbed on upgrade ───────────────────────────


def test_migration_009_scrubs_values_stored_by_an_earlier_version(tmp_path: Path) -> None:
    db = tmp_path / "old.db"
    Store(db).close()

    leaky_uri = "https://alice:pw-9@h.example.com/t.csv?client_secret=OLDSECRET1&jwt=OLDSECRET2&fmt=csv"
    leaky_cfg = json.dumps({"source": {"uri": leaky_uri, "format": "csv"}, "run": {"seed": 1}})
    conn = sqlite3.connect(str(db))
    conn.execute(
        "INSERT INTO dataset (dataset_fp, source_uri, schema_fingerprint, content_fingerprint,"
        " n_rows, n_cols, first_seen, last_seen) VALUES ('d1', ?, 's', 'c', 1, 1, 't', 't')",
        (leaky_uri,),
    )
    conn.execute(
        "INSERT INTO run (run_id, dataset_fp, config_hash, config_json, seed, library_version,"
        " python_version, started_at) VALUES ('r1', 'd1', 'h', ?, 1, 'v', 'p', 't')",
        (leaky_cfg,),
    )
    conn.execute(
        "INSERT INTO run (run_id, dataset_fp, config_hash, config_json, seed, library_version,"
        " python_version, started_at) VALUES ('r2', 'd1', 'h', 'not json', 1, 'v', 'p', 't')"
    )
    conn.execute("DELETE FROM schema_migration WHERE version = 9")
    conn.commit()
    conn.close()

    with Store(db) as store:
        dataset = store.get_dataset("d1")
        run_row = store.get_run("r1")
        odd_row = store.get_run("r2")
    assert dataset is not None
    assert run_row is not None
    assert odd_row is not None

    expected = "https://***@h.example.com/t.csv?client_secret=REDACTED&jwt=REDACTED&fmt=REDACTED"
    assert dataset["source_uri"] == expected
    assert json.loads(run_row["config_json"]) == {
        "source": {"uri": expected, "format": "csv"},
        "run": {"seed": 1},
    }
    assert odd_row["config_json"] == "not json"
    assert dataset["source_digest"] is None
    for f in tmp_path.iterdir():  # the db file itself and any WAL/SHM left beside it
        blob = f.read_bytes()
        for leaked in (b"OLDSECRET1", b"OLDSECRET2", b"pw-9"):
            assert leaked not in blob, f"{leaked!r} survives in {f.name}"


def _seed_pre_009_workspace(db: Path, rows: list[tuple[str, str]]) -> None:
    """A workspace as an earlier version left it: migrated to 008, URIs stored as given."""
    Store(db).close()
    conn = sqlite3.connect(str(db))
    for dataset_fp, uri in rows:
        conn.execute(
            "INSERT INTO dataset (dataset_fp, source_uri, schema_fingerprint, content_fingerprint,"
            " n_rows, n_cols, first_seen, last_seen) VALUES (?, ?, 's', 'c', 1, 1, 't', 't')",
            (dataset_fp, uri),
        )
    conn.execute("DELETE FROM schema_migration WHERE version = 9")
    conn.commit()
    conn.close()


def test_migration_009_leaves_no_overwritten_value_in_free_pages(tmp_path: Path) -> None:
    db = tmp_path / "big.db"
    # Long values spill into overflow pages, which the UPDATE frees rather than reuses.
    secret = "FREEPAGESECRET"
    _seed_pre_009_workspace(
        db,
        [
            (f"d{i}", f"https://h.example.com/t.csv?client_secret={secret}{i:04d}{'x' * 3000}")
            for i in range(50)
        ],
    )
    assert secret.encode() in db.read_bytes()

    Store(db).close()

    for f in tmp_path.iterdir():
        assert secret.encode() not in f.read_bytes(), f"overwritten value survives in {f.name}"


def test_failed_compaction_after_the_scrub_is_a_warning_not_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    db = tmp_path / "w.db"
    _seed_pre_009_workspace(db, [("d1", "https://h.example.com/t.csv?client_secret=S")])
    real_execute = sqlite3.Connection.execute

    class _Conn(sqlite3.Connection):
        def execute(self, sql: str, *args: object) -> sqlite3.Cursor:  # type: ignore[override]
            if sql == "VACUUM":
                raise sqlite3.OperationalError("database is locked")
            return real_execute(self, sql, *args)

    real_connect = sqlite3.connect
    monkeypatch.setattr(sqlite3, "connect", lambda *a, **kw: real_connect(*a, **{**kw, "factory": _Conn}))
    caplog.set_level(logging.WARNING)

    with Store(db) as store:
        dataset = store.get_dataset("d1")

    assert dataset is not None
    assert dataset["source_uri"] == "https://h.example.com/t.csv?client_secret=REDACTED"
    assert "Could not compact" in caplog.text


def test_a_fresh_workspace_is_not_compacted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[None] = []
    monkeypatch.setattr(Store, "_compact", lambda _self: calls.append(None))
    Store(tmp_path / "new.db").close()
    assert calls == []
