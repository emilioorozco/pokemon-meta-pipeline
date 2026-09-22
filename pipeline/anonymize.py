"""HMAC anonymization of player handles in a parsed-game blob.

Why: the parsed blobs carry in-game handles in structured fields (players,
winner, statsByPlayer keys, segment.player, entry.actor) and in free text
(segment titles, entry text, sub-entry text and details, field values,
unparsedLines, extras). Nothing with a handle may land in the lake, so bronze
rewrites every occurrence before writing. Handles are replaced by a keyed HMAC
token rather than a random id so the same handle maps to the same token across
games and across runs, which keeps per-handle joins possible without holding a
mapping table. The key comes from HANDLE_HMAC_KEY in the caller's environment;
this module only takes bytes and never touches the environment.

Every handle is hashed, the owner's included; there is no keep list.

Word boundary: a handle is replaced only where it stands as a whole token, so
`Ash` in `Ashes` or in `Ash_K` is left alone. "Whole token" means the
character before and after the match is not a word character (letter, digit
or underscore, Unicode-aware). Spaces and punctuation are boundaries, which is
what the log text needs: handles are followed by `'s Turn`, ` drew`, `)` and
so on. Handles that themselves contain spaces or underscores are still matched
whole because all handles go into one alternation, longest first, so `Ash K`
wins over `Ash` wherever both could match.
"""

import copy
import hashlib
import hmac
import re
from collections.abc import Callable, Iterable
from typing import Any

TOKEN_HEX_LENGTH = 16

Rewrite = Callable[[Any], Any]


def token_for(handle: str, key: bytes) -> str:
    """Deterministic 16-hex token for a handle under a key."""
    if not key:
        raise ValueError("anonymization key must not be empty")
    digest = hmac.new(key, handle.encode("utf-8"), hashlib.sha256).hexdigest()
    return digest[:TOKEN_HEX_LENGTH]


def handles_in(blob: dict[str, Any]) -> set[str]:
    """The handles a blob names in its structured fields.

    Works for v1 blobs (no summary): players come from statsByPlayer keys and
    segment players. Empty strings are not handles.
    """
    found: set[str] = set()
    summary = blob.get("summary") or {}
    found.update(_strings(summary.get("players") or []))
    for field in ("winner", "opponentName"):
        value = summary.get(field)
        if isinstance(value, str):
            found.add(value)
    found.update(_strings((blob.get("statsByPlayer") or {}).keys()))
    for segment in blob.get("segments") or []:
        player = segment.get("player")
        if isinstance(player, str):
            found.add(player)
    found.discard("")
    return found


def anonymize(blob: dict[str, Any], key: bytes) -> dict[str, Any]:
    """Return a deep copy of the blob with every handle replaced by its token.

    Structured fields are rewritten by exact lookup; free text by a whole-token
    regex over all handles. The input is not modified.
    """
    if not key:
        raise ValueError("anonymization key must not be empty")
    out = copy.deepcopy(blob)
    handles = handles_in(out)
    mapping = {handle: token_for(handle, key) for handle in handles}
    pattern = _pattern(handles)

    def text(value: Any) -> Any:
        """Rewrite handles inside a string; other values pass through."""
        if not isinstance(value, str) or pattern is None:
            return value
        return pattern.sub(lambda m: mapping[m.group(0)], value)

    def exact(value: Any) -> Any:
        """Rewrite a value that should be exactly a handle; fall back to text."""
        if isinstance(value, str) and value in mapping:
            return mapping[value]
        return text(value)

    def strings(values: Any) -> Any:
        if isinstance(values, list):
            return [text(v) for v in values]
        return values

    summary = out.get("summary")
    if isinstance(summary, dict):
        if isinstance(summary.get("players"), list):
            summary["players"] = [exact(p) for p in summary["players"]]
        for field in ("winner", "opponentName"):
            if field in summary:
                summary[field] = exact(summary[field])

    stats = out.get("statsByPlayer")
    if isinstance(stats, dict):
        out["statsByPlayer"] = {exact(handle): side for handle, side in stats.items()}

    for segment in out.get("segments") or []:
        if "player" in segment:
            segment["player"] = exact(segment["player"])
        if "title" in segment:
            segment["title"] = text(segment["title"])
        for entry in segment.get("entries") or []:
            _rewrite_action(entry, exact, text)
            for sub in entry.get("subs") or []:
                _rewrite_action(sub, exact, text)
                if "details" in sub:
                    sub["details"] = strings(sub["details"])

    if "unparsedLines" in out:
        out["unparsedLines"] = strings(out["unparsedLines"])
    extras = out.get("extras")
    if isinstance(extras, dict):
        out["extras"] = {tag: strings(lines) for tag, lines in extras.items()}
    return out


def assert_no_handles(blob: dict[str, Any], handles: set[str]) -> list[str]:
    """Paths of every string (values and dict keys) still holding a handle.

    Walks the whole structure, decklists included, so the caller sees any place
    the rewrite missed. Empty list means clean. A dict key that matches is
    reported as `<parent>.<key>` and the placeholder is kept for anything
    nested under it, so the returned paths never echo a handle themselves.
    """
    pattern = _pattern(handles)
    if pattern is None:
        return []
    hits: list[str] = []

    def walk(node: Any, path: str) -> None:
        if isinstance(node, str):
            if pattern.search(node):
                hits.append(path or "$")
        elif isinstance(node, dict):
            for k, v in node.items():
                key_leaks = isinstance(k, str) and pattern.search(k) is not None
                label = "<key>" if key_leaks else str(k)
                child = f"{path}.{label}" if path else label
                if key_leaks:
                    hits.append(child)
                walk(v, child)
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")

    walk(blob, "")
    return hits


def _rewrite_action(action: dict[str, Any], exact: Rewrite, text: Rewrite) -> None:
    """Rewrite actor, text and every string field value of an entry or sub-entry in place."""
    if "actor" in action:
        action["actor"] = exact(action["actor"])
    if "text" in action:
        action["text"] = text(action["text"])
    fields = action.get("fields")
    if isinstance(fields, dict):
        action["fields"] = {k: text(v) for k, v in fields.items()}


def _pattern(handles: Iterable[str]) -> re.Pattern[str] | None:
    """One alternation over all handles, longest first, matched as whole tokens."""
    ordered = sorted((h for h in handles if h), key=lambda h: (-len(h), h))
    if not ordered:
        return None
    alternation = "|".join(re.escape(h) for h in ordered)
    return re.compile(rf"(?<!\w)(?:{alternation})(?!\w)")


def _strings(values: Iterable[Any]) -> Iterable[str]:
    return (v for v in values if isinstance(v, str))
