"""Small helpers with no dependencies on the rest of the package."""

from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

# Query parameters that carry no identity, only attribution. Dawn and Tribune
# both syndicate the same story under several of these, and without stripping
# them the same article is published two or three times.
_TRACKING_PARAMS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "utm_id", "utm_name", "utm_reader", "utm_brand", "utm_social",
    "fbclid", "gclid", "dclid", "msclkid", "igshid", "mc_cid", "mc_eid",
    "ref", "referrer", "source", "amp", "output", "__twitter_impression",
    "smid", "partner", "cmpid", "ncid", "spm", "share",
}

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_SLUG_STRIP_RE = re.compile(r"[^a-z0-9]+")


def canonical_url(raw: str) -> str:
    """Reduce a URL to a stable identity string.

    Lowercases scheme and host, drops `www.`, removes tracking parameters and
    fragments, and normalises AMP variants so the AMP and canonical copies of a
    story collapse to one entry.
    """
    raw = (raw or "").strip()
    if not raw:
        return ""

    parts = urlsplit(raw)
    scheme = (parts.scheme or "https").lower()
    if scheme not in ("http", "https"):
        return ""

    host = (parts.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    if not host:
        return ""

    # Keep a non-default port; drop 80/443, which are noise.
    if parts.port and parts.port not in (80, 443):
        host = f"{host}:{parts.port}"

    path = parts.path or "/"
    for amp_suffix in ("/amp", "/amp/", ".amp", "/amp.html"):
        if path.endswith(amp_suffix):
            path = path[: -len(amp_suffix)] or "/"
            break
    if len(path) > 1:
        path = path.rstrip("/") or "/"

    kept = [
        (k, v) for k, v in parse_qsl(parts.query, keep_blank_values=False)
        if k.lower() not in _TRACKING_PARAMS
    ]
    query = urlencode(sorted(kept)) if kept else ""

    return urlunsplit((scheme, host, path, query, ""))


def article_id(canonical: str) -> str:
    """Primary key. Derived from the URL, never the title.

    Titles get silently edited after publication all the time; URLs almost never
    do. Keying on the title would make the same story reappear the next day
    under a corrected headline.
    """
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:16]


def stable_key(parts: list[str] | tuple[str, ...]) -> str:
    joined = "|".join(sorted(parts))
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:12]


def strip_html(text: str) -> str:
    """Flatten feed markup to plain text.

    A regex rather than a parser, deliberately: this output is only ever model
    input, never rendered, so a stray malformed tag costs nothing. Pulling in an
    HTML parser would also mean shipping a tool that could be pointed at article
    bodies, which is a capability this project should not have.
    """
    if not text:
        return ""
    text = _TAG_RE.sub(" ", text)
    text = (
        text.replace("&nbsp;", " ").replace("&amp;", "&").replace("&lt;", "<")
        .replace("&gt;", ">").replace("&quot;", '"').replace("&#39;", "'")
        .replace("&rsquo;", "'").replace("&ldquo;", '"').replace("&rdquo;", '"')
        .replace("&mdash;", "—").replace("&ndash;", "–").replace("&hellip;", "…")
    )
    return _WS_RE.sub(" ", text).strip()


def truncate(text: str, limit: int) -> str:
    """Cut to `limit` characters on a word boundary where possible."""
    text = text.strip()
    if len(text) <= limit:
        return text
    cut = text[:limit]
    space = cut.rfind(" ")
    if space > limit * 0.6:
        cut = cut[:space]
    return cut.rstrip(" ,;:.—–-") + "…"


def slugify(text: str) -> str:
    text = unicodedata.normalize("NFKD", text or "")
    text = text.encode("ascii", "ignore").decode("ascii").lower()
    return _SLUG_STRIP_RE.sub("-", text).strip("-") or "item"


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", (text or "").lower())


def shares_long_run(a: str, b: str, n: int = 12) -> bool:
    """True if `a` and `b` share a contiguous run of >= n words.

    This is the plagiarism tripwire. A model instructed to write original prose
    will occasionally paste the publisher's blurb instead; an instruction is not
    enforcement, so every generated summary is checked mechanically.
    """
    wa, wb = words(a), words(b)
    if len(wa) < n or len(wb) < n:
        return False
    seen = {tuple(wb[i:i + n]) for i in range(len(wb) - n + 1)}
    return any(tuple(wa[i:i + n]) in seen for i in range(len(wa) - n + 1))


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def read_json(path: Path, default: Any) -> Any:
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        # A truncated state file must not take the site down; starting from the
        # default just means one day of possible repeats.
        return default


def write_json_atomic(path: Path, payload: Any) -> None:
    """Write via a temp file + rename so a crash can never truncate state.

    Sorted keys and indent=1 keep the git diff line-oriented and reviewable
    instead of one 500 KB line.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1, sort_keys=True, ensure_ascii=False)
        fh.write("\n")
    os.replace(tmp, path)


def env_flag(name: str) -> bool:
    return (os.environ.get(name) or "").strip().lower() in ("1", "true", "yes", "on")


def has_key(name: str) -> bool:
    """Presence check only. The value is never logged or returned."""
    return bool((os.environ.get(name) or "").strip())
