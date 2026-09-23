# Security Policy

## Trust boundary: workspaces are executable, not just data

A `sorethumb` workspace (the directory named by `run.workdir`, default
`.sorethumb/`) stores fitted detector estimators and score calibrators as
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

A source URI's userinfo (`user:pass@host`) and known signed-URL/token query
parameters are stripped before the URI is logged or persisted
(`dataset.source_uri`, `run.config_json`) — but the *auth token itself*
(`source.auth_env_var`) is read from the environment at call time and never
written anywhere. If you embed a credential directly in `source.uri` in a
form this redaction doesn't recognise, it will still reach the HTTP request
line/headers as normal, and any *unrecognised* query parameter is not
redacted — prefer `source.auth`/`source.auth_env_var` over embedding
credentials in the URI itself.

## Supported Versions

`sorethumb` is pre-1.0 (currently `0.x`, alpha) and has not yet had its first
tagged release. Only the latest released version on PyPI is supported;
security fixes are not backported to older `0.x` releases. Once the project
reaches 1.0, this section will be updated with an explicit support window.

| Version | Supported |
| ------- | --------- |
| Latest `0.x` release | :white_check_mark: |
| Older `0.x` releases | :x: |

## Reporting a Vulnerability

Please do not report security vulnerabilities through public GitHub issues.

Email the maintainer directly at hello@t4-digital.uk with the subject
line "sorethumb security vulnerability". You will receive a response within
72 hours acknowledging receipt.

Please include: a description of the vulnerability, steps to reproduce,
potential impact, and any suggested fix.
