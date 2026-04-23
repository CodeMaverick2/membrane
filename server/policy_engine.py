"""Lightweight policy checks (VERIFY verb, dry-run)."""
from __future__ import annotations

from typing import List, Optional, Tuple

from server.text_policy import forbidden_tokens_hitting


def verify_send_content(
    content: str,
    forbidden_substrings: List[str],
) -> Tuple[bool, str, List[str]]:
    """
    Returns (ok, message, hitting_tokens).
    ok=True iff no forbidden substring would appear in a SEND with this body.
    """
    hits = forbidden_tokens_hitting(content, forbidden_substrings)
    if hits:
        return False, f"Would violate policy: tokens {hits!r} must not be sent.", hits
    return True, "VERIFY: proposed content passes forbidden-token check.", []


def disclosure_tier_hint(
    tier: Optional[str],
    surface: Optional[str],
) -> Tuple[bool, str]:
    """Minimal IFC guard: FORBIDDEN tier cannot pair with user-visible surface."""
    if not tier or not surface:
        return True, ""
    if tier.upper() == "FORBIDDEN" and surface == "USER_REPLY":
        return False, "FORBIDDEN disclosure tier is not allowed on USER_REPLY."
    return True, ""
