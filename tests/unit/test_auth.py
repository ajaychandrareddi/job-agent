from __future__ import annotations

import time

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from job_agent.config import Settings
from job_agent.security.auth import (
    JWTAuthenticator,
    RateLimiter,
    StaticTokenAuthenticator,
    build_authenticator,
)

SECRET = "s" * 32


def token(claims: dict[str, object], key: str = SECRET, alg: str = "HS256") -> str:
    return jwt.encode(claims, key, algorithm=alg)


def claims(**kw: object) -> dict[str, object]:
    base: dict[str, object] = {
        "sub": "user-1",
        "exp": int(time.time()) + 300,
        "iss": "https://idp.example",
        "aud": "job-agent",
    }
    base.update(kw)
    return base


class TestStaticTokens:
    def test_matches_only_exact_tokens(self) -> None:
        auth = StaticTokenAuthenticator({"alice": "a" * 20, "bob": "b" * 20})
        assert auth.authenticate("a" * 20) == "alice"
        assert auth.authenticate("b" * 20) == "bob"
        assert auth.authenticate("a" * 19) is None and auth.authenticate("") is None


class TestJWT:
    auth = JWTAuthenticator(
        key=SECRET, algorithms=["HS256"], issuer="https://idp.example", audience="job-agent"
    )

    def test_valid(self) -> None:
        assert self.auth.authenticate(token(claims())) == "user-1"

    @pytest.mark.parametrize(
        "bad",
        [
            claims(exp=int(time.time()) - 3600),  # expired
            claims(aud="other"),
            claims(iss="https://evil.example"),
            {k: v for k, v in claims().items() if k != "sub"},
            {k: v for k, v in claims().items() if k != "exp"},
            claims(sub=""),
        ],
    )
    def test_rejected_claims(self, bad: dict[str, object]) -> None:
        assert self.auth.authenticate(token(bad)) is None

    def test_wrong_key_and_garbage(self) -> None:
        assert self.auth.authenticate(token(claims(), key="x" * 32)) is None
        assert self.auth.authenticate("not-a-jwt") is None

    def test_unsigned_none_algorithm_token_rejected(self) -> None:
        unsigned = jwt.encode(claims(), key="", algorithm="none")
        assert self.auth.authenticate(unsigned) is None
        with pytest.raises(ValueError):
            JWTAuthenticator(key=SECRET, algorithms=["none"])

    def test_rs256_with_public_key(self) -> None:
        private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        pem_private = private.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        ).decode()
        pem_public = (
            private.public_key()
            .public_bytes(
                serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
            )
            .decode()
        )
        auth = JWTAuthenticator(key=pem_public, algorithms=["RS256"])
        assert (
            auth.authenticate(
                token({"sub": "rs", "exp": int(time.time()) + 60}, key=pem_private, alg="RS256")
            )
            == "rs"
        )
        # Algorithm confusion: an HS256 token signed with the public key must fail.
        forged = jwt.api_jws.PyJWS().encode(
            b'{"sub":"evil","exp":9999999999}', key=b"x" * 32, algorithm="HS256"
        )
        assert auth.authenticate(forged) is None

    def test_build_from_settings(self) -> None:
        jwt_auth = build_authenticator(
            Settings(  # type: ignore[call-arg]
                _env_file=None,
                api_auth_mode="jwt",
                api_jwt_secret=SECRET,
                api_jwt_algorithms="HS256",
            )
        )
        assert jwt_auth.authenticate(token({"sub": "x", "exp": int(time.time()) + 60})) == "x"
        static = build_authenticator(
            Settings(  # type: ignore[call-arg]
                _env_file=None, api_static_tokens="t" * 16 + ":dev"
            )
        )
        assert static.authenticate("t" * 16) == "dev"


def test_rate_limiter_token_bucket() -> None:
    now = {"t": 0.0}
    limiter = RateLimiter(60, clock=lambda: now["t"])  # one token per second, burst 60
    assert all(limiter.acquire("k") is None for _ in range(60))
    wait = limiter.acquire("k")
    assert wait is not None and 0 < wait <= 1
    assert limiter.acquire("other") is None  # independent keys
    now["t"] = 1.0
    assert limiter.acquire("k") is None


def test_rate_limiter_bounds_memory() -> None:
    limiter = RateLimiter(5, max_keys=3)
    for i in range(10):
        limiter.acquire(f"k{i}")
    assert len(limiter._buckets) <= 3
