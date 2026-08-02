from __future__ import annotations

import pytest

from scripts.tinker_protocol_smoke import _context_tokens


def test_context_tokens_prefers_treatment_value_without_pool_default() -> None:
    assert _context_tokens({"model": "m", "context_tokens": 32_768}, {}) == 32_768


def test_context_tokens_supports_legacy_pool_default() -> None:
    assert _context_tokens({"model": "m"}, {"context_tokens": 65_536}) == 65_536


def test_context_tokens_rejects_missing_value() -> None:
    with pytest.raises(ValueError, match="missing context_tokens for treatment m"):
        _context_tokens({"model": "m"}, {})
