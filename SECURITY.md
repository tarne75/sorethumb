# Security Policy

## Trust boundary: workspaces are executable, not just data

A `sorethumb` workspace (the directory named by `--workdir` or `run.workdir`;
by default `./sorethumb-workspace/` in the current directory) stores fitted
detector estimators and score calibrators, under `models/`, as
`joblib`/pickle files. Unpickling is code execution: loading a pickle file can
run arbitrary Python chosen by whoever produced that file, not just the
`sorethumb` code you installed.

`sorethumb score --from-run RUN_ID` loads these files directly
(`joblib.load`) with no sandboxing, and `run.reuse_models` does the same to
reload a persisted model within the same run. Treat any workspace you did
not create yourself as untrusted input, in the same category as running a
downloaded script. Do not run these commands against a workspace copied from
another machine, a shared drive, a CI artifact, or any other party unless you
fully trust its provenance.

Workspaces do carry SHA-256 file digests for the estimator and calibrator
files, recorded in each model's manifest and checked on load. **This is an
integrity check, not a security boundary.** It detects accidental corruption
or a misplaced/swapped file — a mismatch fails closed with an error. It does
**not** detect a maliciously crafted pickle: an attacker who controls the
file also controls the digest recorded alongside it, so a hostile workspace
can carry a digest that "matches" perfectly. Do not treat a passing digest
check as evidence that a workspace is safe to load.

## Source downloads: SSRF-adjacent hardening, not a sandbox

`source.uri` pointed at `http(s)://` is fetched with `httpx`. Redirects are
followed manually and capped; any redirect target whose host differs from
the one you configured must resolve to a public, non-reserved address --
refusing an obvious pivot to a cloud metadata endpoint (169.254.169.254) or
another internal/link-local target a compromised or malicious remote server
redirects to. The host you configure yourself is never blocked, even if it
is internal (e.g. `http://localhost:8080/export.csv`) -- that is a
deliberate choice you made, not something a third party redirected you
into. This is **not** a defence against DNS rebinding (the resolved address
is not pinned for the actual connection) and does not sandbox the remote
server in any other way; only fetch datasets from sources you trust.

Credentials are only ever sent over HTTPS. A `source.uri` that starts with
`http://` (in any letter case) combined with `source.auth` other than
`none`, or with `user:password@` in the URI, is rejected when the config is
loaded; the same check runs again before every request and redirect hop, so
nothing credentialed leaves the process in cleartext (including via an
HTTP-to-HTTPS redirect, whose first request would otherwise already have
leaked it). There is no loopback exception and no override: put the endpoint
behind TLS, or fetch it without credentials. A plain `http://` source with
no credentials still works.

If `source.auth`/`source.auth_env_var` is configured, the resulting
`Authorization` header is sent only to the exact origin (scheme, host, and
effective port) `source.uri` names. A redirect to any other origin — a
different host, a different port, or even a same-host scheme change — gets
the request without it, so a compromised or malicious remote server cannot
use a redirect to have your credential sent elsewhere. A redirect that
would downgrade the connection from HTTPS to HTTP is refused outright, not
just stripped of the credential, since that also removes transport
security for the response body itself.

`source.max_download_bytes` bounds both the declared `Content-Length` and
the actual streamed size, so an unbounded or misconfigured response cannot
exhaust disk space.

### What is recorded about a source URI

`source.uri` may carry secrets (signed-URL signatures, API keys, tokens), so
the full URI exists **only in memory**, for the HTTP request it describes.
Everywhere it is stored, logged or reported -- `dataset.source_uri`,
`run.config_json`, the log file, console output, `--json` output, HTML/CSV
reports and error messages (including tracebacks) -- sorethumb uses a
redacted display form instead:

    https://host.example.com/export/data.csv?X-Amz-Signature=REDACTED&X-Amz-Date=REDACTED

The scheme, host, non-default port, path and the query **key names** are kept;
**every** query value is replaced by `REDACTED`, blank ones included (a blank
value is indistinguishable from a redacted one, which is the point). Userinfo
becomes `***@`, and the fragment is dropped. A query segment with no `=` is
replaced whole, since it could itself be the secret. Redaction does not consult
a list of "sensitive" parameter names, so `client_secret`, `jwt`, `accessKey`
or a name nobody has thought of are protected exactly like `token`. The
`httpx` request log line, which would otherwise print the full URL, is
scrubbed the same way. There is no setting that turns redaction off or restores
the raw values.

To tell two sources that differ only in a query value apart, sorethumb also
records `source_digest` (on `dataset` and `run`, and in `run --json`): a
SHA-256 of the canonical full URI. It cannot be reversed, but it is an
*unsalted* hash, so a secret with little entropy (a short numeric PIN, say)
could be confirmed by someone who has the database and can guess it. Treat the
workspace as you would any file derived from your data.

Known limits:

- **Secrets in the URI path** (`https://host/<token>/data.csv`) are not
  recognised and are stored as written. Put credentials in the query string,
  `source.auth` or `source.auth_env_var`.
- The `source.auth_env_var` token is read from the environment at call time and
  never written anywhere.
- Workspaces created by an earlier version held query values in clear text.
  Opening one runs migration 009, which irreversibly rewrites the stored
  `dataset.source_uri` and `run.config_json` to the redacted form. Copies of the
  old database (backups, snapshots) and SQLite free pages are outside its reach;
  rotate any credential that was ever stored.

Dataset identity is unaffected by redaction. The logical dataset id derived
from a URI ignores the query, so refreshing a signed URL keeps one dataset's
history together. Because that also means two URIs differing only in a query
value look like one dataset, a run is **refused** when `source.dataset_id` is
unset and the set of query key names differs from the dataset's previous run;
set `source.dataset_id` to say explicitly what is, or is not, the same dataset.
`config_hash` (and so the run id) still covers the full URI, so a refreshed
signature starts a new run rather than silently reusing the old one's models.

## Supported Platforms

Tested via CI on **Linux, macOS and Windows** (x64) — unit, contract and
full integration (real workspace, SQLite, CLI subprocess, report rendering)
suites on all three, and on Windows also an install of the built wheel run
from PowerShell. Keep a workspace on a local drive: SQLite and atomic file
replacement are not reliable on network shares or in synced folders, on any
OS. See the README's
[Supported platforms](https://github.com/tarne75/sorethumb#supported-platforms)
section.

## Supported Versions

`sorethumb` is pre-1.0 (currently `0.x`, alpha) and has not yet had its first
tagged release. Only the latest released version on PyPI is supported;
security fixes are not backported to older `0.x` releases. Once the project
reaches 1.0, this section will be updated with an explicit support window.

| Version | Supported |
| ------- | --------- |
| Latest `0.x` release | :white_check_mark: |
| Older `0.x` releases | :x: |

If a released version needs to be pulled for a security issue, it is yanked
on PyPI (never deleted — PyPI doesn't allow that), which stops new,
unpinned installs from picking it up while leaving existing pinned installs
unaffected. See
[docs/releasing.md](https://github.com/tarne75/sorethumb/blob/main/docs/releasing.md#rolling-back-a-bad-release)
for the exact procedure.

## Reporting a Vulnerability

Please do not report security vulnerabilities through public GitHub issues.

Email the maintainer, Tarne Westcott, directly at hello@t4-digital.uk (the
project's security contact) with the subject line "sorethumb security
vulnerability". This is a small project maintained on a best-effort basis: I
will try to acknowledge a report promptly and to fix confirmed issues as soon
as I reasonably can, but I cannot promise a response time.

Please include: a description of the vulnerability, steps to reproduce,
potential impact, and any suggested fix.
