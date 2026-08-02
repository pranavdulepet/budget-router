from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Mapping

REDACTED = "[REDACTED]"
SENSITIVE_KEYS = re.compile(
    r"(api[_-]?key|authorization|password|passwd|secret|access[_-]?token|private[_-]?key)",
    re.IGNORECASE,
)
SECRET_PATTERNS = (
    re.compile(
        r"\b(?:sk[-_]|tk[-_]|ghp_|github_pat_|tml-)[A-Za-z0-9_-]{16,}\b"
    ),
    re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{12,}\b", re.IGNORECASE),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"\bTINKER_API_KEY\s*=\s*\S+", re.IGNORECASE),
)
PRIVATE_REASONING = (
    re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL),
    re.compile(
        r"<(?:reasoning|analysis)>.*?</(?:reasoning|analysis)>",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(r"<(?:think|reasoning|analysis)>.*\Z", re.IGNORECASE | re.DOTALL),
)
OPAQUE_TOKEN = re.compile(r"[A-Za-z0-9_+/=-]{32,}")


def _looks_like_unknown_secret(token: str) -> bool:
    """Conservatively identify opaque tokens while allowing public identifiers."""
    # Filesystem and public URL paths are common in coding-agent transcripts.
    if "/" in token:
        return False
    # Canonical traces intentionally contain Git SHA-1, SHA-256, and Docker IDs.
    if len(token) in {40, 64} and re.fullmatch(r"[0-9a-fA-F]+", token):
        return False
    return _entropy(token) >= 4.2


def _redact_text(value: str) -> str:
    result = value
    for pattern in PRIVATE_REASONING:
        result = pattern.sub("[HIDDEN_REASONING_REMOVED]", result)
    for pattern in SECRET_PATTERNS:
        result = pattern.sub(REDACTED, result)
    # The scanner also treats unknown high-entropy opaque tokens as suspicious.
    # Redact those candidates rather than rejecting an otherwise useful coding
    # trajectory after the provider call has already completed.
    result = OPAQUE_TOKEN.sub(
        lambda match: (
            REDACTED
            if _looks_like_unknown_secret(match.group(0))
            else match.group(0)
        ),
        result,
    )
    return result


def redact(value: Any) -> Any:
    """Recursively remove credentials while retaining visible research state."""
    if isinstance(value, Mapping):
        return {
            str(key): (
                REDACTED
                if SENSITIVE_KEYS.search(str(key))
                else redact(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact(item) for item in value)
    if isinstance(value, str):
        return _redact_text(value)
    return value


def _entropy(value: str) -> float:
    if not value:
        return 0.0
    counts = Counter(value)
    return -sum(
        count / len(value) * math.log2(count / len(value)) for count in counts.values()
    )


def scan_for_secrets(value: Any, *, path: str = "$") -> tuple[str, ...]:
    """Return paths containing known credential shapes or suspicious tokens."""
    findings: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            item_path = f"{path}.{key}"
            if SENSITIVE_KEYS.search(str(key)) and item not in (None, "", REDACTED):
                findings.append(item_path)
            findings.extend(scan_for_secrets(item, path=item_path))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            findings.extend(scan_for_secrets(item, path=f"{path}[{index}]"))
    elif isinstance(value, str):
        if any(pattern.search(value) for pattern in SECRET_PATTERNS):
            findings.append(path)
        for token in OPAQUE_TOKEN.findall(value):
            if _looks_like_unknown_secret(token):
                findings.append(path)
                break
    return tuple(dict.fromkeys(findings))
