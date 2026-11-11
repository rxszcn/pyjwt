import socket

import pytest

import jwt
from jwt.api_jwk import PyJWK
from jwt.api_jws import PyJWS
from jwt.exceptions import (
    ExpiredSignatureError,
    InsecureJWKSUrlError,
    InvalidAlgorithmError,
    InvalidConfigurationError,
    InvalidSignatureError,
    PyJWKClientError,
)
from jwt.hardening import (
    JWKSUrlPolicy,
    audit_verification_config,
    normalize_algorithms,
    resolve_jwks_uri,
    validate_options,
)
from jwt.utils import base64url_encode

SECRET = "a" * 32
JWKS_URL = "https://jwks.example/.well-known/jwks.json"


@pytest.fixture
def public_dns(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make any host name resolve to a public (non-private) address."""

    def fake_getaddrinfo(host: str, port: int, **kwargs: object) -> list[tuple]:
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port))]

    monkeypatch.setattr("jwt.hardening.socket.getaddrinfo", fake_getaddrinfo)


@pytest.fixture
def private_dns(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make any host name resolve to an internal (private) address."""

    def fake_getaddrinfo(host: str, port: int, **kwargs: object) -> list[tuple]:
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.17", port))]

    monkeypatch.setattr("jwt.hardening.socket.getaddrinfo", fake_getaddrinfo)


def make_token(payload: dict[str, object] | None = None, **headers: object) -> str:
    return jwt.encode(payload or {"hello": "world"}, SECRET, "HS256", headers=headers)


class TestAlgorithmAllowlist:
    def test_bare_string_is_a_single_exact_algorithm(self) -> None:
        token = make_token()

        # The documented convenience form keeps working.
        assert jwt.decode(token, SECRET, algorithms="HS256") == {"hello": "world"}

        # It is no longer a substring search: a token whose alg merely occurs
        # inside the configured string used to be accepted.
        with pytest.raises(InvalidAlgorithmError, match="not allowed"):
            jwt.decode(token, SECRET, algorithms="HS384HS256")
        with pytest.raises(InvalidAlgorithmError, match="not allowed"):
            jwt.decode(token, SECRET, algorithms="HS384, HS256")

    @pytest.mark.parametrize(
        "algorithms",
        [
            ["HS256"],
            ("HS256",),
            {"HS256"},
            frozenset({"HS256"}),
            {"HS256": True},
            iter(["HS256"]),
        ],
    )
    def test_exact_match_holds_for_every_container_shape(self, algorithms: object) -> None:
        token = make_token()
        assert jwt.decode(token, SECRET, algorithms=algorithms) == {  # type: ignore[arg-type]
            "hello": "world"
        }

    def test_normalize_algorithms_rejects_non_string_entries(self) -> None:
        with pytest.raises(InvalidConfigurationError):
            normalize_algorithms(["HS256", b"HS384"])
        with pytest.raises(InvalidConfigurationError):
            normalize_algorithms(b"HS256")
        with pytest.raises(InvalidConfigurationError):
            normalize_algorithms([""])

    def test_pyjws_constructor_treats_string_as_one_algorithm(self) -> None:
        jws = PyJWS(algorithms="HS256")

        assert jws.get_algorithms() == ["HS256"]
        token = PyJWS().encode(b"hello world", SECRET, "HS256")
        assert jws.decode(token, SECRET, algorithms=["HS256"]) == b"hello world"

    def test_instance_allowlist_applies_to_pyjwk_path(self) -> None:
        # A PyJWK bound to HS384 must not slip past a PyJWS instance which was
        # constructed with only HS256.
        jwk = PyJWK(
            {
                "kty": "oct",
                "alg": "HS384",
                "k": base64url_encode(b"b" * 48).decode(),
            }
        )
        token = PyJWS().encode(b"hello world", jwk)

        assert PyJWS().decode(token, jwk) == b"hello world"

        restricted = PyJWS(algorithms=["HS256"])
        with pytest.raises(InvalidAlgorithmError, match="not allowed by this PyJWS"):
            restricted.decode(token, jwk)
        # Supplying the key's algorithm explicitly does not override the
        # instance-level allow-list either.
        with pytest.raises(InvalidAlgorithmError, match="not allowed by this PyJWS"):
            restricted.decode(token, jwk, algorithms=["HS384"])


class TestOptionValidation:
    @pytest.mark.parametrize(
        "options",
        [
            {"verify_expiry": True},
            {"verify_expiration": True},
            {"verifyExp": True},
            {"verify_nfb": True},
        ],
    )
    def test_misspelled_options_raise_instead_of_being_ignored(
        self, options: dict[str, object]
    ) -> None:
        expired = jwt.encode({"exp": 1}, SECRET, "HS256")

        # The dangerous variant: signature verification is off, so the
        # silently-ignored option also left the expiration check off.
        with pytest.raises(InvalidConfigurationError, match="Unrecognized options"):
            jwt.decode(
                expired,
                SECRET,
                algorithms=["HS256"],
                options={"verify_signature": False, **options},
            )

        with pytest.raises(InvalidConfigurationError, match="Unrecognized options"):
            jwt.PyJWT(options=options)  # type: ignore[arg-type]

    def test_valid_option_spellings_still_work(self) -> None:
        expired = jwt.encode({"exp": 1}, SECRET, "HS256")

        with pytest.raises(ExpiredSignatureError):
            jwt.decode(
                expired,
                SECRET,
                algorithms=["HS256"],
                options={"verify_exp": True},
            )
        assert jwt.decode(
            expired,
            SECRET,
            algorithms=["HS256"],
            options={"verify_exp": False},
        ) == {"exp": 1}

    @pytest.mark.parametrize("value", [None, "false", "true", 0.0, [False]])
    def test_ambiguous_option_values_raise(self, value: object) -> None:
        expired = jwt.encode({"exp": 1}, SECRET, "HS256")
        with pytest.raises(InvalidConfigurationError, match="must be a bool"):
            jwt.decode(
                expired,
                SECRET,
                algorithms=["HS256"],
                options={"verify_exp": value},
            )

    def test_integer_booleans_keep_their_historical_meaning(self) -> None:
        expired = jwt.encode({"exp": 1}, SECRET, "HS256")

        # 0 previously disabled the check and 1 enabled it; preserve that.
        assert jwt.decode(
            expired,
            SECRET,
            algorithms=["HS256"],
            options={"verify_exp": 0},
        ) == {"exp": 1}
        with pytest.raises(ExpiredSignatureError):
            jwt.decode(
                expired,
                SECRET,
                algorithms=["HS256"],
                options={"verify_exp": 1},
            )

    def test_require_must_be_a_sequence_of_claim_names(self) -> None:
        token = make_token()

        with pytest.raises(InvalidConfigurationError, match="'require'"):
            jwt.decode(
                token,
                SECRET,
                algorithms=["HS256"],
                options={"require": "exp"},
            )
        with pytest.raises(InvalidConfigurationError, match="'require'"):
            jwt.decode(
                token,
                SECRET,
                algorithms=["HS256"],
                options={"require": ["exp", 1]},  # type: ignore[list-item]
            )

    def test_validate_options_returns_a_copy(self) -> None:
        options = {"verify_exp": True}
        validated = validate_options(options)

        assert validated == {"verify_exp": True}
        assert options == {"verify_exp": True}
        assert validated is not options

    def test_pyjws_validates_its_options_too(self) -> None:
        with pytest.raises(InvalidConfigurationError, match="Unrecognized options"):
            PyJWS(options={"verify_expiry": True})  # type: ignore[typeddict-unknown-key]

        # The same escape hatch applies at the JWS layer.
        PyJWS(
            options={"verify_expiry": True},  # type: ignore[typeddict-unknown-key]
            strict_options=False,
        )

    def test_legacy_mode_is_an_explicit_compatibility_escape_hatch(self) -> None:
        expired = jwt.encode({"exp": 1}, SECRET, "HS256")
        legacy = jwt.PyJWT(
            options={"verify_signature": False, "verify_expiry": True},
            strict_options=False,
        )

        # The typo is still ignored here, but only because the caller opted
        # out of strict validation explicitly.
        assert legacy.decode(expired, SECRET, algorithms=["HS256"]) == {"exp": 1}

    def test_instance_level_options_are_actually_effective(self) -> None:
        # An option configured on the instance must describe what runs when
        # decode() is called without per-call options. Previously the
        # instance-level verify_signature=False was silently replaced by the
        # hard-coded default at decode time.
        instance = jwt.PyJWT(options={"verify_signature": False})
        token = jwt.encode({"hello": "world"}, SECRET, "HS256")

        assert instance.decode(token) == {"hello": "world"}

        # Per-call options still win, so an explicitly re-enabled check runs.
        with pytest.raises(InvalidSignatureError):
            instance.decode(
                token,
                b"b" * 32,
                algorithms=["HS256"],
                options={"verify_signature": True},
            )


class TestAudit:
    def test_reports_ineffective_and_unsafe_constraints(self) -> None:
        findings = audit_verification_config(
            {"verify_exp": True, "verify_expiry": True},
            algorithms=["HS256", "none"],
        )
        codes = {finding.code for finding in findings}

        assert "unknown-option" in codes
        assert "none-algorithm-allowed" in codes
        assert "claim-not-required:exp" in codes
        assert "audience-not-configured" in codes

    def test_reports_missing_and_mixed_allowlists(self) -> None:
        findings = audit_verification_config(algorithms=None)
        assert "missing-algorithm-allowlist" in {f.code for f in findings}

        findings = audit_verification_config(algorithms=["HS256", "RS256"])
        assert "mixed-key-families" in {f.code for f in findings}

    def test_reports_disabled_signature_verification(self) -> None:
        findings = audit_verification_config({"verify_signature": False})
        assert "signature-verification-disabled" in {f.code for f in findings}

    def test_clean_config_has_no_findings(self) -> None:
        findings = audit_verification_config(
            {"require": ["exp", "nbf", "iat"], "verify_aud": False, "verify_iss": False},
            algorithms=["HS256"],
            audience="api",
            issuer="issuer",
        )
        assert findings == []

    def test_pyjwk_key_allows_omitting_algorithms(self) -> None:
        findings = audit_verification_config(key_is_jwk=True)
        assert "missing-algorithm-allowlist" not in {f.code for f in findings}

    def test_findings_are_sorted_by_severity(self) -> None:
        findings = audit_verification_config(
            {"verify_signature": False, "verify_expiry": True},
            algorithms=["none"],
        )
        order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        assert [order[f.severity] for f in findings] == sorted(
            order[f.severity] for f in findings
        )


class TestJWKSUrlPolicy:
    def test_allows_public_https_url_inside_origin_allowlist(self, public_dns: None) -> None:
        policy = JWKSUrlPolicy(allowed_origins=["https://jwks.example"])
        assert policy.validate(JWKS_URL) == JWKS_URL

    @pytest.mark.parametrize(
        "uri",
        [
            "http://jwks.example/jwks.json",  # HTTPS is required by default
            "https://other.example/jwks.json",  # origin not allow-listed
            "https://user:pass@jwks.example/jwks.json",  # user info
            "https://jwks.example/jwks.json#fragment",
        ],
    )
    def test_rejects_urls_outside_policy(self, uri: str, public_dns: None) -> None:
        policy = JWKSUrlPolicy(allowed_origins=["https://jwks.example"])
        with pytest.raises(InsecureJWKSUrlError):
            policy.validate(uri)

    @pytest.mark.parametrize(
        "uri",
        [
            "https://127.0.0.1/jwks.json",
            "https://169.254.169.254/latest/meta-data/",  # cloud metadata
            "https://[::1]/jwks.json",
            "https://[::ffff:127.0.0.1]/jwks.json",
        ],
    )
    def test_rejects_literal_non_public_addresses(self, uri: str) -> None:
        with pytest.raises(InsecureJWKSUrlError, match="non-public address"):
            JWKSUrlPolicy().validate(uri)

    def test_rejects_host_names_resolving_to_private_addresses(
        self, private_dns: None
    ) -> None:
        policy = JWKSUrlPolicy(allowed_origins=["https://jwks.example"])
        with pytest.raises(InsecureJWKSUrlError, match="non-public address"):
            policy.validate(JWKS_URL)

    def test_private_addresses_require_an_explicit_opt_in(self) -> None:
        policy = JWKSUrlPolicy(
            allowed_origins=["https://127.0.0.1"],
            allow_private_addresses=True,
        )
        assert policy.validate("https://127.0.0.1/jwks.json")

    def test_http_can_be_explicitly_enabled(self, public_dns: None) -> None:
        policy = JWKSUrlPolicy(
            allowed_origins=["http://jwks.example"], require_https=False
        )
        assert policy.validate("http://jwks.example/jwks.json")

    def test_invalid_origin_configuration_is_rejected(self) -> None:
        with pytest.raises(InvalidConfigurationError):
            JWKSUrlPolicy(allowed_origins=["jwks.example"])  # type: ignore[list-item]
        with pytest.raises(InvalidConfigurationError):
            JWKSUrlPolicy(allowed_origins="https://jwks.example")  # type: ignore[arg-type]

    def test_resolve_jwks_uri_requires_an_allowlist(self) -> None:
        header = {"jku": JWKS_URL}

        with pytest.raises(InvalidConfigurationError, match="allowed_origins"):
            resolve_jwks_uri(header, allowed_origins=None)
        with pytest.raises(InvalidConfigurationError, match="must not be empty"):
            resolve_jwks_uri(header, allowed_origins=[])
        with pytest.raises(InvalidConfigurationError, match="not a single string"):
            resolve_jwks_uri(header, allowed_origins=JWKS_URL)  # type: ignore[arg-type]

    def test_resolve_jwks_uri_rejects_schemes_and_hosts(self) -> None:
        with pytest.raises(InsecureJWKSUrlError):
            resolve_jwks_uri(
                {"jku": "ftp://jwks.example/jwks.json"},
                allowed_origins=["https://jwks.example"],
                resolve_dns=False,
            )
        with pytest.raises(InsecureJWKSUrlError, match="non-public address"):
            resolve_jwks_uri(
                {"jku": "https://127.0.0.1/jwks.json"},
                allowed_origins=["https://127.0.0.1"],
            )

    def test_resolve_jwks_uri_validates_the_header_value(self) -> None:
        with pytest.raises(PyJWKClientError, match="does not contain a 'jku'"):
            resolve_jwks_uri({}, allowed_origins=["https://jwks.example"])
        with pytest.raises(PyJWKClientError, match="non-empty string"):
            resolve_jwks_uri(
                {"jku": 123}, allowed_origins=["https://jwks.example"]
            )


class TestClientIntegration:
    def test_client_validates_url_when_policy_is_supplied(self, public_dns: None) -> None:
        policy = JWKSUrlPolicy(allowed_origins=["https://jwks.example"])
        client = jwt.PyJWKClient(JWKS_URL, url_policy=policy)
        assert client.uri == JWKS_URL

        with pytest.raises(InsecureJWKSUrlError):
            jwt.PyJWKClient(
                "https://internal.example/jwks.json", url_policy=policy
            )

    def test_client_revalidates_before_each_fetch(self, public_dns: None) -> None:
        policy = JWKSUrlPolicy()
        client = jwt.PyJWKClient(JWKS_URL, url_policy=policy)
        # Simulate a subclass or later mutation redirecting the client.
        client.uri = "https://127.0.0.1/jwks.json"

        with pytest.raises(InsecureJWKSUrlError, match="non-public address"):
            client.fetch_data()

    def test_fixed_uri_behaviour_is_unchanged_without_a_policy(self) -> None:
        # Internal endpoints passed as trusted configuration keep working;
        # the policy is opt-in so existing integrations do not break.
        jwt.PyJWKClient("http://localhost/jwks.json")
        jwt.PyJWKClient("http://10.0.0.17/jwks.json")

    def test_from_token_header_builds_a_policy_bound_client(self, public_dns: None) -> None:
        token = make_token(jku=JWKS_URL)

        client = jwt.PyJWKClient.from_token_header(
            token, allowed_origins=["https://jwks.example"]
        )

        assert client.uri == JWKS_URL
        assert isinstance(client.url_policy, JWKSUrlPolicy)

    def test_from_token_header_rejects_attacker_controlled_urls(
        self, private_dns: None
    ) -> None:
        token = make_token(jku="https://evil.example/jwks.json")
        with pytest.raises(InsecureJWKSUrlError, match="not in the allowed origins"):
            jwt.PyJWKClient.from_token_header(
                token, allowed_origins=["https://jwks.example"]
            )

        internal = make_token(jku="https://internal.example/jwks.json")
        with pytest.raises(InsecureJWKSUrlError, match="non-public address"):
            jwt.PyJWKClient.from_token_header(
                internal,
                allowed_origins=["https://internal.example"],
            )

    def test_from_token_header_requires_an_allowlist(self) -> None:
        token = make_token(jku=JWKS_URL)
        with pytest.raises(InvalidConfigurationError, match="allowed_origins"):
            jwt.PyJWKClient.from_token_header(token, allowed_origins=None)
        with pytest.raises(InvalidConfigurationError, match="must not be empty"):
            jwt.PyJWKClient.from_token_header(token, allowed_origins=[])

    def test_from_token_header_accepts_a_header_mapping(self, public_dns: None) -> None:
        client = jwt.PyJWKClient.from_token_header(
            {"jku": JWKS_URL},
            allowed_origins=["https://jwks.example"],
        )
        assert client.uri == JWKS_URL
