"""Source URIs shaped like the real thing, each with the secret values to hunt for.

Every case is ``(uri, secrets)``: *secrets* are the exact substrings that must never
appear at rest, in a log, in a report or in an error once the URI has been through
redaction. Percent-encoded values are listed in both their encoded and decoded form.
"""

from __future__ import annotations

from urllib.parse import unquote

_AWS_SIGNATURE = "fe5f80f77d5fa3beca038a248ff027d0445342fe2855ddc963176630326f1024"
_AWS_CREDENTIAL = "AKIAIOSFODNN7EXAMPLE%2F20260101%2Fus-east-1%2Fs3%2Faws4_request"
_GOOG_SIGNATURE = "3a1b6c9d0e2f4a5b7c8d9e0f1a2b3c4d5e6f708192a3b4c5d6e7f8091a2b3c4d"
_GOOG_CREDENTIAL = (
    "svc-account%40my-project.iam.gserviceaccount.com%2F20260101%2Fauto%2Fstorage%2Fgoog4_request"
)
_AZURE_SIG = "Zm9vYmFyU0lHTkFUVVJF%2Bxyz%2F%3D"
_AZURE_SKOID = "11111111-2222-3333-4444-555555555555"

_RAW_CASES: dict[str, tuple[str, list[str]]] = {
    "client_secret": (
        "https://data.example.com/t.csv?client_id=app1&client_secret=CS-9f3a7c1e&format=csv",
        ["CS-9f3a7c1e"],
    ),
    "credential": ("https://data.example.com/t.csv?credential=CRED-77aa11bb", ["CRED-77aa11bb"]),
    "jwt": (
        "https://data.example.com/t.csv?jwt=eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ4In0.SIGPART_zz9",
        ["eyJhbGciOiJIUzI1NiJ9", "eyJzdWIiOiJ4In0", "SIGPART_zz9"],
    ),
    "accessKey": ("https://data.example.com/t.csv?accessKey=AKIAEXAMPLEKEY0001", ["AKIAEXAMPLEKEY0001"]),
    "mixed_case_arbitrary_name": (
        "https://data.example.com/t.csv?XyZzY_Tok-N=MiXeDcAsE-value-42&Plain=plain-val-7",
        ["MiXeDcAsE-value-42", "plain-val-7"],
    ),
    "duplicate_keys": (
        "https://data.example.com/t.csv?k=first-secret-1&k=second-secret-2&K=third-secret-3",
        ["first-secret-1", "second-secret-2", "third-secret-3"],
    ),
    "blank_values": (
        "https://data.example.com/t.csv?token=&other=&real=after-blank-val",
        ["after-blank-val"],
    ),
    "bare_segment": ("https://data.example.com/t.csv?bare-secret-segment-9&a=b", ["bare-secret-segment-9"]),
    "userinfo": ("https://alice:s3cret-pw-55@data.example.com/t.csv?x=1", ["alice", "s3cret-pw-55"]),
    "fragment": ("https://data.example.com/t.csv?x=1#frag-secret-31", ["frag-secret-31"]),
    "aws_sigv4": (
        "https://bucket.s3.us-east-1.amazonaws.com/t.csv?X-Amz-Algorithm=AWS4-HMAC-SHA256"
        f"&X-Amz-Credential={_AWS_CREDENTIAL}&X-Amz-Date=20260101T000000Z&X-Amz-Expires=3600"
        f"&X-Amz-SignedHeaders=host&X-Amz-Signature={_AWS_SIGNATURE}",
        [_AWS_SIGNATURE, _AWS_CREDENTIAL, "AKIAIOSFODNN7EXAMPLE", "20260101T000000Z"],
    ),
    "google_v4": (
        "https://storage.googleapis.com/bkt/t.csv?X-Goog-Algorithm=GOOG4-RSA-SHA256"
        f"&X-Goog-Credential={_GOOG_CREDENTIAL}&X-Goog-Date=20260101T000000Z&X-Goog-Expires=900"
        f"&X-Goog-SignedHeaders=host&X-Goog-Signature={_GOOG_SIGNATURE}",
        [_GOOG_SIGNATURE, _GOOG_CREDENTIAL, "my-project.iam.gserviceaccount.com"],
    ),
    "azure_sas": (
        "https://acct.blob.core.windows.net/cont/t.csv?sv=2022-11-02&ss=b&srt=co&sp=r"
        f"&se=2026-12-31T00%3A00%3A00Z&st=2026-01-01T00%3A00%3A00Z&spr=https&sig={_AZURE_SIG}"
        f"&skoid={_AZURE_SKOID}",
        [_AZURE_SIG, _AZURE_SKOID, "2026-12-31"],
    ),
}


def _with_decoded(secrets: list[str]) -> list[str]:
    out: list[str] = []
    for secret in secrets:
        out.append(secret)
        if unquote(secret) != secret:
            out.append(unquote(secret))
    return out


SIGNED_URI_CASES: dict[str, tuple[str, list[str]]] = {
    name: (uri, _with_decoded(secrets)) for name, (uri, secrets) in _RAW_CASES.items()
}
