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

## Reporting a Vulnerability

Please do not report security vulnerabilities through public GitHub issues.

Email the maintainer directly at hello@t4-digital.uk with the subject
line "sorethumb security vulnerability". You will receive a response within
72 hours acknowledging receipt.

Please include: a description of the vulnerability, steps to reproduce,
potential impact, and any suggested fix.
