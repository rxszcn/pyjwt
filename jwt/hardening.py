"""Strict configuration checks and opt-in hardening helpers.

PyJWT's public API deliberately accepts a few loosely typed arguments (for
example, ``algorithms`` is documented as a sequence but a bare string also
passes a naive ``isinstance`` check).  Historical behaviour has also been to
ignore unknown entries in the ``options`` mapping.  Both are convenient for
callers, but they make it possible for a configuration which *claims* to
enable a verification constraint to silently do something weaker at runtime:

* ``algorithms="HS256"`` used to be a substring test, so
  ``algorithms="HS384HS256"`` also accepted an ``HS256`` token.
* A ``PyJWK`` carries its own algorithm, which used to bypass the allow-list
  configured on the ``PyJWS`` instance.
* ``options={"verify_expiry": True}`` was silently ignored, so a typo could
  leave expiration checks disabled (particularly when combined with
  ``verify_signature=False``).
* A JWKS URL taken from an unverified token header (``jku``) could point at
  an internal address.

This module centralises the normalisation and validation used by the rest of
the library, and exposes helpers which can be called at configuration time to
*detect* problems before a request is ever verified.  The runtime behaviour is
tightened where the old behaviour was ambiguous; callers which used the
documented, correctly typed API are unaffected.
"""

from __future__ import annotations

import ipaddress
import socket
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from difflib import get_close_matches
from typing import Any
from urllib.parse import urlsplit

from .exceptions import (
    InsecureJWKSUrlError,
    InvalidConfigurationError,
    PyJWKClientError,
)

# Options accepted by PyJWT/PyJWS.  Keep this in sync with jwt.types.
_VERIFY_FLAGS = frozenset(
    {
        "verify_signature",
        "verify_exp",
        "verify_nbf",
        "verify_iat",
        "verify_aud",
        "verify_iss",
        "verify_sub",
        "verify_jti",
    }
)
_BOOL_OPTIONS = _VERIFY_FLAGS | {"strict_aud", "enforce_minimum_key_length"}
OPTION_KEYS = frozenset(_BOOL_OPTIONS | {"require"})

_HMAC_ALGORITHMS = frozenset({"HS256", "HS384", "HS512"})
_ASYMMETRIC_ALGORITHMS = frozenset(
    {
        "RS256",
        "RS384",
        "RS512",
        "PS256",
        "PS384",
        "PS512",
        "ES256",
        "ES256K",
        "ES384",
        "ES521",
        "ES512",
        "EdDSA",
    }
)

_SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}


def _suggestion(key: str, candidates: Iterable[str]) -> str:
    matches = get_close_matches(key, list(candidates), n=1, cutoff=0.6)
    if not matches:
        return ""
    return f" Did you mean {matches[0]!r}?"


def _as_bool(key: str, value: Any, source: str) -> bool:
    if isinstance(value, bool):
        return value
    # Historically 0/1 were accepted through plain truthiness.  Preserve
    # exactly those two values, but reject everything else: ``None``,
    # ``"false"`` and friends are ambiguous and have silently disabled checks
    # in the past.
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    raise InvalidConfigurationError(
        f"{source}[{key!r}] must be a bool (True or False), got "
        f"{value!r} of type {type(value).__name__}. Ambiguous values can "
        "silently turn a verification constraint off; pass an explicit bool."
    )


def _as_require(value: Any, source: str) -> list[str]:
    if isinstance(value, (str, bytes, bytearray)):
        raise InvalidConfigurationError(
            f"{source}['require'] must be a sequence of claim names, not a "
            f"single {type(value).__name__}. A bare string would be iterated "
            "character by character."
        )
    if isinstance(value, Mapping) or not isinstance(value, Iterable):
        raise InvalidConfigurationError(
            f"{source}['require'] must be a sequence of claim names, got "
            f"{value!r} of type {type(value).__name__}."
        )
    claims = list(value)
    for claim in claims:
        if not isinstance(claim, str) or not claim:
            raise InvalidConfigurationError(
                f"{source}['require'] entries must be non-empty strings, got "
                f"{claim!r} of type {type(claim).__name__}."
            )
    return claims


def validate_options(
    options: Mapping[str, Any] | None,
    *,
    strict: bool = True,
    allowed_keys: Iterable[str] = OPTION_KEYS,
    source: str = "options",
) -> dict[str, Any]:
    """Validate and normalise a verification ``options`` mapping.

    With ``strict=True`` (the default) unknown keys and ambiguous values
    raise :class:`~jwt.exceptions.InvalidConfigurationError` instead of being
    ignored at runtime.  This is what catches typos such as
    ``verify_expiry`` before they turn into a silently skipped check.

    ``strict=False`` reproduces the historical tolerant behaviour.  It exists
    as an explicit compatibility escape hatch for integrations which cannot
    be fixed immediately; it should not be used for new code.

    A copy is always returned; the caller's mapping is never mutated.
    """
    if options is None:
        return {}
    if not isinstance(options, Mapping):
        # Preserve the historical TypeError for non-mapping options.
        raise TypeError(
            f"{source} must be a mapping, not {type(options).__name__}"
        )

    data = dict(options)
    if not strict:
        return data

    allowed = frozenset(allowed_keys)
    unknown = [key for key in data if key not in allowed]
    if unknown:
        details = "; ".join(
            f"{key!r}{_suggestion(str(key), allowed)}" for key in unknown
        )
        raise InvalidConfigurationError(
            f"Unrecognized {source} key(s): {details}. Unknown keys are "
            "ignored by PyJWT, so they can silently leave a verification "
            "constraint unenforced. Remove them or use the documented "
            "spelling."
        )

    for key, value in data.items():
        if key in _BOOL_OPTIONS:
            data[key] = _as_bool(key, value, source)
        elif key == "require":
            data[key] = _as_require(value, source)
    return data


def normalize_algorithms(
    algorithms: Any,
    *,
    source: str = "algorithms",
) -> tuple[str, ...] | None:
    """Return ``algorithms`` as a tuple of exact algorithm names.

    A bare string is treated as a *single* algorithm name rather than as an
    iterable of characters or a substring to search.  This keeps the
    convenient ``algorithms="HS256"`` activation working while making the
    allow-list an exact-match check.

    Mappings are accepted for backwards compatibility and their keys are used
    (the historical ``"alg" in mapping`` check had the same effect).
    """
    if algorithms is None:
        return None
    if isinstance(algorithms, str):
        return (algorithms,)
    if isinstance(algorithms, (bytes, bytearray, memoryview)):
        raise InvalidConfigurationError(
            f"{source} must be a sequence of algorithm names, not "
            f"{type(algorithms).__name__}. Use e.g. ['HS256']."
        )
    if isinstance(algorithms, Mapping):
        items: list[Any] = list(algorithms.keys())
    else:
        try:
            items = list(algorithms)
        except TypeError as err:
            raise InvalidConfigurationError(
                f"{source} must be a sequence of algorithm names, got "
                f"{algorithms!r} of type {type(algorithms).__name__}."
            ) from err

    for item in items:
        if not isinstance(item, str) or not item:
            raise InvalidConfigurationError(
                f"{source} entries must be non-empty strings, got {item!r} "
                f"of type {type(item).__name__}."
            )
    return tuple(items)


@dataclass(frozen=True)
class AuditFinding:
    """A single result returned by :func:`audit_verification_config`."""

    code: str
    """Machine-readable identifier, e.g. ``"unknown-option"``."""
    severity: str
    """One of ``"critical"``, ``"high"``, ``"medium"`` or ``"low"``."""
    message: str
    """Human-readable explanation and remediation hint."""


def _loose_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    return None


def audit_verification_config(
    options: Mapping[str, Any] | None = None,
    *,
    algorithms: Any = None,
    key_is_jwk: bool = False,
    audience: Any = None,
    issuer: Any = None,
) -> list[AuditFinding]:
    """Inspect a verification configuration and report ineffective controls.

    This is the detection companion to the runtime checks in
    :func:`validate_options` and :func:`normalize_algorithms`.  Call it at
    startup (or in a unit test) to prove that the constraints a caller
    believes are enabled really are.

    The function never raises for insecure (but well-formed) configuration;
    it returns findings sorted by severity.  A non-mapping ``options`` is a
    programming error and still raises :class:`TypeError`.
    """
    if options is not None and not isinstance(options, Mapping):
        raise TypeError(
            f"options must be a mapping, not {type(options).__name__}"
        )

    raw: dict[str, Any] = {} if options is None else dict(options)
    findings: list[AuditFinding] = []

    unknown = [key for key in raw if key not in OPTION_KEYS]
    for key in unknown:
        findings.append(
            AuditFinding(
                code="unknown-option",
                severity="medium",
                message=(
                    f"Unknown option {key!r} is not understood by PyJWT. "
                    "Strict runtime validation rejects it; if it is ignored "
                    "(for example in legacy mode) it enforces nothing."
                    f"{_suggestion(str(key), OPTION_KEYS)}"
                ),
            )
        )

    for key in sorted(_BOOL_OPTIONS & set(raw)):
        value = raw[key]
        if _loose_bool(value) is None:
            findings.append(
                AuditFinding(
                    code="invalid-option-value",
                    severity="medium",
                    message=(
                        f"Option {key!r} has value {value!r}; only bool values "
                        "(or 0/1) are understood, and other truthy/falsy "
                        "values can change behaviour unexpectedly."
                    ),
                )
            )

    require: list[str] = []
    if "require" in raw:
        try:
            require = _as_require(raw["require"], "options")
        except InvalidConfigurationError as err:
            findings.append(
                AuditFinding(
                    code="invalid-option-value",
                    severity="medium",
                    message=str(err),
                )
            )
    require_set = set(require)

    def enabled(flag: str) -> bool:
        default = _loose_bool(raw.get("verify_signature", True))
        if default is None:
            default = True
        value = _loose_bool(raw.get(flag, default))
        if value is None:
            value = default
        return value

    if not enabled("verify_signature"):
        findings.append(
            AuditFinding(
                code="signature-verification-disabled",
                severity="high",
                message=(
                    "verify_signature is disabled. Token contents are not "
                    "authenticated; claim checks alone do not make an "
                    "untrusted token trustworthy."
                ),
            )
        )

    for claim, flag, severity in (
        ("exp", "verify_exp", "medium"),
        ("nbf", "verify_nbf", "low"),
        ("iat", "verify_iat", "low"),
    ):
        if enabled(flag) and claim not in require_set:
            findings.append(
                AuditFinding(
                    code=f"claim-not-required:{claim}",
                    severity=severity,
                    message=(
                        f"{flag} is enabled but {claim!r} is not listed in "
                        f"'require'. The {claim!r} check only runs when the "
                        "claim is present, so a token which omits it is "
                        "accepted without that check. Add "
                        f"{claim!r} to options['require'] to enforce it."
                    ),
                )
            )

    try:
        allowlist = normalize_algorithms(algorithms)
    except InvalidConfigurationError as err:
        findings.append(
            AuditFinding(
                code="invalid-algorithms",
                severity="high",
                message=str(err),
            )
        )
        allowlist = None

    if allowlist is None:
        if not key_is_jwk:
            findings.append(
                AuditFinding(
                    code="missing-algorithm-allowlist",
                    severity="critical",
                    message=(
                        "No algorithm allow-list was supplied. Signature "
                        "verification must be constrained to algorithms "
                        "chosen independently of the token header."
                    ),
                )
            )
    else:
        if "none" in allowlist:
            findings.append(
                AuditFinding(
                    code="none-algorithm-allowed",
                    severity="critical",
                    message=(
                        "The 'none' algorithm is in the allow-list. Tokens "
                        "with an empty signature would be accepted."
                    ),
                )
            )
        if set(allowlist) & _HMAC_ALGORITHMS and set(allowlist) & (
            _ASYMMETRIC_ALGORITHMS
        ):
            findings.append(
                AuditFinding(
                    code="mixed-key-families",
                    severity="high",
                    message=(
                        "The allow-list mixes HMAC and asymmetric algorithms. "
                        "Both interpret key material differently, which is "
                        "the precondition for key-type confusion attacks."
                    ),
                )
            )

    if enabled("verify_aud") and audience is None:
        findings.append(
            AuditFinding(
                code="audience-not-configured",
                severity="medium",
                message=(
                    "verify_aud is enabled but no audience was supplied. "
                    "Tokens without an 'aud' claim pass, while tokens with "
                    "one are rejected."
                ),
            )
        )
    if enabled("verify_iss") and issuer is None:
        findings.append(
            AuditFinding(
                code="issuer-not-configured",
                severity="medium",
                message=(
                    "verify_iss is enabled but no issuer was supplied, so "
                    "the issuer check is skipped."
                ),
            )
        )

    findings.sort(key=lambda finding: _SEVERITY_ORDER.get(finding.severity, 9))
    return findings


_DEFAULT_PORTS = {"http": 80, "https": 443}


def _parse_origin(uri: str, *, what: str) -> tuple[str, str, int]:
    try:
        parts = urlsplit(uri)
        port = parts.port
    except ValueError as err:
        raise InsecureJWKSUrlError(f"Invalid JWKS {what} {uri!r}: {err}") from err

    scheme = parts.scheme.lower()
    host = parts.hostname
    if scheme not in _DEFAULT_PORTS or not host:
        raise InsecureJWKSUrlError(
            f"Invalid JWKS {what} {uri!r}: expected an http(s) URL with a host."
        )
    if port is None:
        port = _DEFAULT_PORTS[scheme]
    return scheme, host.lower(), port


def _is_global_address(
    address: ipaddress.IPv4Address | ipaddress.IPv6Address,
) -> bool:
    if isinstance(address, ipaddress.IPv6Address):
        mapped = address.ipv4_mapped
        if mapped is not None:
            address = mapped
    return address.is_global


def _resolve_addresses(
    host: str,
    port: int,
    *,
    resolve_dns: bool,
) -> list[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        literal = None
    if literal is not None:
        return [literal]
    if not resolve_dns:
        return []

    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except (socket.gaierror, OSError, UnicodeError) as err:
        raise InsecureJWKSUrlError(
            f"Unable to resolve JWKS host {host!r}: {err}"
        ) from err

    addresses: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = []
    for info in infos:
        sockaddr = info[4]
        address_text = sockaddr[0]
        if isinstance(address_text, str):
            # IPv6 scope IDs (``fe80::1%eth0``) are not accepted by
            # ipaddress.ip_address.
            address_text = address_text.split("%", 1)[0]
        try:
            addresses.append(ipaddress.ip_address(address_text))
        except ValueError as err:
            raise InsecureJWKSUrlError(
                f"Unable to interpret JWKS host {host!r} address "
                f"{address_text!r}: {err}"
            ) from err

    if not addresses:
        raise InsecureJWKSUrlError(
            f"Unable to resolve JWKS host {host!r}: no addresses returned."
        )
    return addresses


@dataclass(frozen=True)
class JWKSUrlPolicy:
    """Policy applied to a JWKS URL before it is fetched.

    ``PyJWKClient(uri)`` treats ``uri`` as trusted application configuration
    and keeps its historical behaviour.  Use this policy when the URL can be
    influenced by an attacker, for example a ``jku`` header copied out of an
    unverified token: pass it to :class:`~jwt.PyJWKClient` or use
    :meth:`~jwt.PyJWKClient.from_token_header`.

    :param allowed_origins: Optional iterable of origins (``scheme://host``
        or ``scheme://host:port``) a URL must match exactly.  When omitted
        any public origin is accepted, subject to the other checks.
    :param require_https: Require the ``https`` scheme (default ``True``).
    :param allow_private_addresses: Permit loopback, private, link-local and
        otherwise non-global addresses (default ``False``).  Enable this only
        for intentionally internal endpoints which are also origin-restricted.
    :param resolve_dns: Resolve host names and check every resolved address
        (default ``True``).  Disabling it leaves DNS-rebinding risk in place;
        it exists only for environments where resolution at validation time is
        undesirable.

    .. note::

        Resolution happens immediately before the request, but a hostile DNS
        server can still return a different answer for the second lookup
        (DNS rebinding). Pin the endpoint by IP or enforce network egress
        controls when that threat is in scope.
    """

    allowed_origins: Iterable[str] | None = field(
        default=None, compare=False, hash=False
    )
    require_https: bool = True
    allow_private_addresses: bool = False
    resolve_dns: bool = True
    _normalized_origins: frozenset[tuple[str, str, int]] = field(
        init=False, repr=False
    )

    def __post_init__(self) -> None:
        origins = self.allowed_origins
        if origins is None:
            normalized: frozenset[tuple[str, str, int]] = frozenset()
        else:
            if isinstance(origins, str) or not isinstance(origins, Iterable):
                raise InvalidConfigurationError(
                    "allowed_origins must be a sequence of origins, not "
                    f"{type(origins).__name__}. Use e.g. "
                    "['https://issuer.example']."
                )
            try:
                normalized = frozenset(
                    _parse_origin(origin, what="origin") for origin in origins
                )
            except InsecureJWKSUrlError as err:
                # Origin entries are trusted configuration, so report a bad
                # entry as a configuration error rather than a URL rejection.
                raise InvalidConfigurationError(str(err)) from err
        object.__setattr__(self, "_normalized_origins", normalized)

    def validate(self, uri: str) -> str:
        """Check ``uri`` against the policy and return it unchanged.

        :raises InsecureJWKSUrlError: if the URL is not permitted.
        """
        if not isinstance(uri, str):
            raise InsecureJWKSUrlError(
                f"JWKS URL must be a string, not {type(uri).__name__}"
            )
        scheme, host, port = _parse_origin(uri, what="URL")
        parts = urlsplit(uri)

        if parts.username is not None or parts.password is not None:
            raise InsecureJWKSUrlError(
                f"JWKS URL {uri!r} must not contain user information."
            )
        if parts.fragment:
            raise InsecureJWKSUrlError(
                f"JWKS URL {uri!r} must not contain a fragment."
            )

        if self.require_https and scheme != "https":
            raise InsecureJWKSUrlError(
                f"JWKS URL {uri!r} must use https; got {scheme!r}."
            )

        origin = (scheme, host, port)
        if (
            self._normalized_origins
            and origin not in self._normalized_origins
        ):
            allowed_display = ", ".join(
                f"{s}://{h}:{p}" for s, h, p in self._normalized_origins
            )
            raise InsecureJWKSUrlError(
                f"JWKS URL {uri!r} is not in the allowed origins "
                f"({allowed_display})."
            )

        if not self.allow_private_addresses:
            for address in _resolve_addresses(
                host, port, resolve_dns=self.resolve_dns
            ):
                if not _is_global_address(address):
                    raise InsecureJWKSUrlError(
                        f"JWKS URL {uri!r} resolves to non-public address "
                        f"{address}; refusing to fetch from private, "
                        "loopback, link-local or reserved networks. Set "
                        "allow_private_addresses=True only for an explicitly "
                        "oriented internal endpoint."
                    )
        return uri


def resolve_jwks_uri(
    header: Mapping[str, Any],
    *,
    allowed_origins: Iterable[str] | None,
    allow_private_addresses: bool = False,
    require_https: bool = True,
    resolve_dns: bool = True,
) -> str:
    """Resolve the ``jku`` value from an *unverified* token header safely.

    Unlike :meth:`PyJWKClient.get_signing_key_from_jwt`, this helper never
    trusts a URL merely because it appeared in a token.  An explicit,
    non-empty origin allow-list is required, HTTPS is required by default,
    and non-public addresses are rejected.

    :raises InvalidConfigurationError: if no usable allow-list is configured.
    :raises PyJWKClientError: if the header has no valid ``jku`` value.
    :raises InsecureJWKSUrlError: if the URL fails the policy.
    """
    if not isinstance(header, Mapping):
        raise TypeError(
            f"header must be a mapping, not {type(header).__name__}"
        )
    if allowed_origins is None:
        raise InvalidConfigurationError(
            "allowed_origins is required when a JWKS URL comes from a token "
            "header. Pass an explicit list such as "
            "['https://issuer.example']."
        )
    if isinstance(allowed_origins, str):
        raise InvalidConfigurationError(
            "allowed_origins must be a sequence of origins, not a single "
            "string. Use e.g. ['https://issuer.example']."
        )
    try:
        origins = list(allowed_origins)
    except TypeError as err:
        raise InvalidConfigurationError(
            "allowed_origins must be a sequence of origins, not "
            f"{type(allowed_origins).__name__}."
        ) from err
    if not origins:
        raise InvalidConfigurationError(
            "allowed_origins must not be empty when a JWKS URL comes from a "
            "token header; an empty allow-list would accept any public URL."
        )

    jku = header.get("jku")
    if jku is None:
        raise PyJWKClientError(
            "Token header does not contain a 'jku' (JWK Set URL) parameter."
        )
    if not isinstance(jku, str) or not jku:
        raise PyJWKClientError(
            "Token header 'jku' parameter must be a non-empty string."
        )

    policy = JWKSUrlPolicy(
        allowed_origins=origins,
        require_https=require_https,
        allow_private_addresses=allow_private_addresses,
        resolve_dns=resolve_dns,
    )
    policy.validate(jku)
    return jku


__all__ = [
    "AuditFinding",
    "JWKSUrlPolicy",
    "OPTION_KEYS",
    "audit_verification_config",
    "normalize_algorithms",
    "resolve_jwks_uri",
    "validate_options",
]
