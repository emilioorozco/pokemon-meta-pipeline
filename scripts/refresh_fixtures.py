"""Refresh the committed test fixtures: real stock games, anonymized under a throwaway key.

What this is: one repeatable command that reads the application's parsed blobs
out of S3, picks a small set that covers the shapes this pipeline has to handle,
and writes each pick to `tests/fixtures` as an anonymized JSON file. It never
writes to the source bucket, and no blob is ever stored as it was read: the
bodies live in memory for the length of the run and only the anonymized,
scrubbed copies reach the disk.

Why a throwaway key instead of HANDLE_HMAC_KEY: the fixtures are committed, and
bronze is anonymized with the pipeline's real key. A fixture written under that
same key would carry the very tokens bronze carries, so anyone with the public
repository could join a public fixture against a private bronze row, or confirm
a guessed handle by hashing it and looking for the token. Each run therefore
draws its own 32 random bytes, uses them for every pick so tokens stay
consistent inside the set, and drops them when the process exits: the key is
never printed, never written, never reused. `--seed-key` exists only to
reproduce one run while debugging it.

Selection: a candidate is a stock export without full decklists, not excluded
from stats, with at least one segment and no opponent decklist (a pasted list is
still someone else's list). Candidates are sorted by gameId, then picked
greedily to satisfy, in this order: every distinct opponent archetype, three
games on each seat, one game that ended in a concede, the longest game, one win
and one loss. A target the candidates cannot satisfy is reported and never
fatal, and any slots left under `--count` are filled in gameId order.

Scrub: anonymization only rewrites handles, so the fields that identify the
uploader, the account or the event are deleted afterwards. `summary.userId`
becomes `user-<8 hex>` of the same keyed hash, and uploadTokenId, uploadClient,
deckId, deckName, deckVersion, notes, tournamentId, tournamentName, round, elo
and matchId are removed. `gameId` is kept: it is a content hash of the log, not
an account identifier, and it is what makes a fixture traceable back to a bug
report.

Every pick is re-validated against the contract and checked twice for leaks: the
whole-token check the pipeline itself uses, plus a plain substring scan for the
raw handles and the raw userId. Either failure aborts the run before anything is
written, and only the gameId prefix and the masked path are printed. A handle
made only of hex characters can collide with a token and trip the substring
scan; that is a safe failure, and a re-run draws a new key and clears it.
"""

import argparse
import copy
import json
import os
import secrets
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from pydantic import ValidationError

from pipeline.anonymize import anonymize, assert_no_handles, handles_in, token_for
from pipeline.config import REPO_ROOT
from pipeline.contract import ContractError, ParsedBlobV2, parse_blob
from pipeline.settings import DEFAULT_PREFIX, DEFAULT_REGION
from pipeline.source import SourceBlob, get_blob, list_parsed_keys

if TYPE_CHECKING:  # the boto3 stubs are a dev dependency, not a runtime one
    from mypy_boto3_s3.client import S3Client

SCRUBBED_KEYS: Final = (
    "uploadTokenId",
    "uploadClient",
    "deckId",
    "deckName",
    "deckVersion",
    "notes",
    "tournamentId",
    "tournamentName",
    "round",
    "elo",
    "matchId",
)
CONCEDE_REASONS: Final = frozenset({"concede", "opponent_concede"})
SEAT_QUOTA: Final = 3
SEATS: Final = (0, 1)
KEY_BYTES: Final = 32
USER_TOKEN_LENGTH: Final = 8
GAME_ID_PREFIX: Final = 8
DEFAULT_COUNT: Final = 10
DEFAULT_OUT: Final = REPO_ROOT / "tests" / "fixtures"
FIXTURE_GLOB: Final = "game-*.json"
PATHS_SHOWN: Final = 5


class FixtureError(RuntimeError):
    """A pick could not be made safe. The message carries a gameId prefix and paths only."""


@dataclass(frozen=True)
class Candidate:
    """One game that may become a fixture: the selection keys plus the body as read.

    `raw` is out of the comparison and out of `repr` so a candidate can be
    printed or compared in a test without dragging a whole blob along.
    """

    game_id: str
    my_side: int | None
    result: str
    end_reason: str
    turn_count: int
    opponent_archetype: str | None
    raw: dict[str, Any] = field(repr=False, compare=False)

    @classmethod
    def of(cls, blob: ParsedBlobV2, raw: dict[str, Any]) -> "Candidate":
        summary = blob.summary
        return cls(
            game_id=summary.game_id,
            my_side=summary.my_side,
            result=summary.result,
            end_reason=summary.end_reason,
            turn_count=summary.turn_count,
            opponent_archetype=summary.opponent_archetype,
            raw=raw,
        )

    def line(self, number: int) -> str:
        """The one-line report for this pick. Archetypes are deck names, not handles."""
        seat = "?" if self.my_side is None else self.my_side
        return (
            f"{number:02d} {self.game_id[:GAME_ID_PREFIX]} seat={seat} result={self.result} "
            f"end={self.end_reason} turns={self.turn_count} "
            f"opp_archetype={self.opponent_archetype or 'unknown'}"
        )


@dataclass(frozen=True)
class Selection:
    """What `select_fixtures` chose, in pick order, and how the coverage came out."""

    picked: list[Candidate]
    met: list[str]
    unmet: list[str]


@dataclass(frozen=True)
class Scan:
    """What one pass over the prefix saw. `skipped` counts everything that is not a v2 blob."""

    listed: int
    skipped: int
    candidates: list[Candidate]


@dataclass(frozen=True)
class Fixture:
    """One anonymized, scrubbed, re-validated body and the name it is written under."""

    number: int
    candidate: Candidate
    body: dict[str, Any] = field(repr=False, compare=False)

    @property
    def filename(self) -> str:
        return f"game-{self.number:02d}-{self.candidate.game_id[:GAME_ID_PREFIX]}.json"


Helps = Callable[[Candidate, Sequence[Candidate]], bool]
Met = Callable[[Sequence[Candidate]], bool]
Describe = Callable[[Sequence[Candidate]], str]


@dataclass(frozen=True)
class _Target:
    """One coverage goal: does this candidate advance it, is it reached, how does it read."""

    name: str
    helps: Helps
    met: Met
    describe: Describe


def is_candidate(blob: ParsedBlobV2) -> bool:
    """Whether a game may become a fixture; see the module docstring for why each test."""
    summary = blob.summary
    return (
        summary.export_variant == "stock"
        and not summary.has_full_decklists
        and summary.excluded_from_stats is not True
        and len(blob.segments) > 0
        and blob.opponent_decklist is None
    )


def select_fixtures(candidates: Sequence[Candidate], count: int) -> Selection:
    """Pick at most `count` candidates, covering as much as the candidates allow.

    Deterministic: the pool is sorted by gameId first, and each step takes the
    candidate that advances the highest-priority unmet target, breaking ties by
    the next targets and then by gameId. Once no candidate advances anything,
    the remaining slots are filled in gameId order, so a large `--count` still
    yields that many fixtures. Targets that cannot be reached are reported in
    `unmet` rather than raised.
    """
    pool = sorted(candidates, key=lambda c: c.game_id)
    targets = _targets(pool)
    picked: list[Candidate] = []
    remaining = list(pool)

    while remaining and len(picked) < count:
        unmet = [target for target in targets if not target.met(picked)]
        if not unmet:
            break
        best = min(remaining, key=lambda c: (_miss(c, picked, unmet), c.game_id))
        if all(_miss(best, picked, unmet)):  # it advances nothing; the rest is filler
            break
        picked.append(best)
        remaining.remove(best)

    picked += remaining[: max(0, count - len(picked))]
    met = [target.describe(picked) for target in targets if target.met(picked)]
    unreached = [target.describe(picked) for target in targets if not target.met(picked)]
    return Selection(picked=picked, met=met, unmet=unreached)


def scrub(anon: dict[str, Any], key: bytes, raw_user_id: str) -> dict[str, Any]:
    """Return a copy with the account-identifying summary keys gone and userId tokenized.

    Only `summary` is touched: anonymization has already rewritten every handle
    elsewhere, and the keys listed in `SCRUBBED_KEYS` are the ones that name an
    account, a deck, an upload or an event rather than the game. The input is
    not modified.
    """
    out = copy.deepcopy(anon)
    summary = out.get("summary")
    if isinstance(summary, dict):
        summary["userId"] = user_token(raw_user_id, key)
        for name in SCRUBBED_KEYS:
            summary.pop(name, None)
    return out


def user_token(raw_user_id: str, key: bytes) -> str:
    """The stand-in for a userId: the same keyed hash the handles get, shortened."""
    return "user-" + token_for(raw_user_id, key)[:USER_TOKEN_LENGTH]


def leak_paths(anon: dict[str, Any], raw: dict[str, Any], expected_user_id: str) -> list[str]:
    """Masked paths where a pre-anonymization value survived; empty means the pick is clean.

    Two checks: the pipeline's whole-token check, and a substring scan for the
    same handles plus the raw userId, which also catches a handle that only
    appears inside a longer word. `summary.userId` is compared against the token
    this run computed instead of being scanned, because the raw id may well be a
    prefix of its own replacement.
    """
    handles = handles_in(raw)
    paths = list(assert_no_handles(anon, handles))
    scanned = copy.deepcopy(anon)
    summary = scanned.get("summary")
    if not isinstance(summary, dict) or summary.get("userId") != expected_user_id:
        paths.append("summary.userId")
    if isinstance(summary, dict):
        summary.pop("userId", None)

    needles = {handle for handle in handles if handle}
    raw_user_id = _summary_of(raw).get("userId")
    if isinstance(raw_user_id, str) and raw_user_id:
        needles.add(raw_user_id)
    for path in _substring_paths(scanned, needles):
        if path not in paths:
            paths.append(path)
    return paths


def build_fixture(candidate: Candidate, number: int, key: bytes) -> Fixture:
    """Anonymize, scrub, re-validate and leak-check one pick. Raises `FixtureError` if unsafe."""
    raw = candidate.raw
    short = candidate.game_id[:GAME_ID_PREFIX]
    raw_user_id = str(_summary_of(raw).get("userId", ""))
    expected = user_token(raw_user_id, key)
    body = scrub(anonymize(raw, key), key, raw_user_id)
    try:
        ParsedBlobV2.model_validate(body)
    except ValidationError as exc:
        # Locations only, and not chained: a pydantic message and the traceback
        # it carries can both quote the value that failed validation.
        locations = [".".join(str(part) for part in error["loc"]) for error in exc.errors()]
        raise FixtureError(
            f"{short} no longer fits the contract at {locations[:PATHS_SHOWN]}"
        ) from None
    paths = leak_paths(body, raw, expected)
    if paths:
        raise FixtureError(f"{short} still carries a source value at {paths[:PATHS_SHOWN]}")
    return Fixture(number=number, candidate=candidate, body=body)


def scan_bucket(s3: "S3Client", bucket: str, prefix: str) -> Scan:
    """Read every object under the prefix and keep the candidates in memory."""
    listed = 0
    skipped = 0
    candidates: list[Candidate] = []
    for obj in list_parsed_keys(s3, bucket, prefix):
        listed += 1
        pair = _as_v2(get_blob(s3, bucket, obj.key))
        if pair is None:
            skipped += 1
            continue
        raw, blob = pair
        if is_candidate(blob):
            candidates.append(Candidate.of(blob, raw))
    return Scan(listed=listed, skipped=skipped, candidates=candidates)


def write_fixtures(fixtures: Sequence[Fixture], out_dir: Path) -> list[Path]:
    """Replace the fixture set in `out_dir` with these bodies, one file each."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in sorted(out_dir.glob(FIXTURE_GLOB)):
        stale.unlink()
    written: list[Path] = []
    for fixture in fixtures:
        path = out_dir / fixture.filename
        path.write_text(json.dumps(fixture.body, indent=2, sort_keys=True) + "\n")
        written.append(path)
    return written


def report(scan: Scan, selection: Selection) -> str:
    """The whole output of a run: counts, one line per pick, and the coverage verdict."""
    lines = [
        f"listed: {scan.listed}",
        f"not a v2 blob: {scan.skipped}",
        f"candidates: {len(scan.candidates)}",
        f"picked: {len(selection.picked)}",
    ]
    lines += [candidate.line(number) for number, candidate in enumerate(selection.picked, start=1)]
    lines.append("coverage met: " + ("; ".join(selection.met) or "none"))
    lines.append("coverage unmet: " + ("; ".join(selection.unmet) or "none"))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Refresh the fixture set from the bucket and print the coverage summary."""
    args = _parse_args(argv)
    key = args.seed_key or secrets.token_bytes(KEY_BYTES)
    scan = scan_bucket(_default_client(), args.bucket, DEFAULT_PREFIX)
    selection = select_fixtures(scan.candidates, args.count)
    try:
        fixtures = [
            build_fixture(candidate, number, key)
            for number, candidate in enumerate(selection.picked, start=1)
        ]
    except FixtureError as exc:
        # Nothing has been written yet, so the previous fixture set survives.
        print(f"aborted: {exc}", file=sys.stderr)
        return 1
    write_fixtures(fixtures, args.out)
    print(report(scan, selection))
    return 0


def _targets(pool: Sequence[Candidate]) -> list[_Target]:
    """The coverage goals for this pool, highest priority first."""
    archetypes = _archetypes(pool)
    longest = min(pool, key=lambda c: (-c.turn_count, c.game_id)) if pool else None

    def seats(picked: Sequence[Candidate], side: int) -> int:
        return sum(1 for c in picked if c.my_side == side)

    def results(picked: Sequence[Candidate]) -> set[str]:
        return {c.result for c in picked}

    def conceded(picked: Sequence[Candidate]) -> int:
        return sum(1 for c in picked if c.end_reason in CONCEDE_REASONS)

    return [
        _Target(
            name="archetypes",
            helps=lambda c, picked: (
                bool(c.opponent_archetype) and c.opponent_archetype not in _archetypes(picked)
            ),
            met=lambda picked: _archetypes(picked) >= archetypes,
            describe=lambda picked: (
                f"archetypes: {len(_archetypes(picked))} of {len(archetypes)} distinct"
            ),
        ),
        _Target(
            name="seats",
            helps=lambda c, picked: (
                c.my_side is not None
                and c.my_side in SEATS
                and seats(picked, c.my_side) < SEAT_QUOTA
            ),
            met=lambda picked: all(seats(picked, side) >= SEAT_QUOTA for side in SEATS),
            describe=lambda picked: (
                f"seats: {seats(picked, 0)} on seat 0 and {seats(picked, 1)} on seat 1, "
                f"want {SEAT_QUOTA} each"
            ),
        ),
        _Target(
            name="concede",
            helps=lambda c, picked: c.end_reason in CONCEDE_REASONS and conceded(picked) == 0,
            met=lambda picked: conceded(picked) > 0,
            describe=lambda picked: f"concede: {conceded(picked)} game(s) ended in a concede",
        ),
        _Target(
            name="longest",
            helps=lambda c, picked: (
                longest is not None
                and c.game_id == longest.game_id
                and not any(p.game_id == longest.game_id for p in picked)
            ),
            met=lambda picked: longest is None or any(p.game_id == longest.game_id for p in picked),
            describe=lambda picked: (
                "longest: none to pick"
                if longest is None
                else f"longest: the {longest.turn_count}-turn game"
            ),
        ),
        _Target(
            name="results",
            helps=lambda c, picked: c.result in ("win", "loss") and c.result not in results(picked),
            met=lambda picked: {"win", "loss"} <= results(picked),
            describe=lambda picked: (
                "results: " + (", ".join(sorted(results(picked) & {"win", "loss"})) or "neither")
            ),
        ),
    ]


def _miss(
    candidate: Candidate, picked: Sequence[Candidate], unmet: Sequence[_Target]
) -> tuple[bool, ...]:
    """False where the candidate advances that target, so sorting ascending prefers it."""
    return tuple(not target.helps(candidate, picked) for target in unmet)


def _archetypes(candidates: Sequence[Candidate]) -> set[str]:
    return {c.opponent_archetype for c in candidates if c.opponent_archetype}


def _as_v2(source: SourceBlob) -> tuple[dict[str, Any], ParsedBlobV2] | None:
    """The decoded body and its v2 parse, or None for anything that is not a v2 blob."""
    try:
        raw = source.decode()
    except ValueError:  # both JSONDecodeError and UnicodeDecodeError are ValueErrors
        return None
    if not isinstance(raw, dict):
        return None
    try:
        blob = parse_blob(raw)
    except ContractError:
        return None
    if not isinstance(blob, ParsedBlobV2):
        return None
    return raw, blob


def _summary_of(blob: dict[str, Any]) -> dict[str, Any]:
    summary = blob.get("summary")
    return summary if isinstance(summary, dict) else {}


def _substring_paths(node: Any, needles: set[str]) -> list[str]:
    """Paths of every string, value or dict key, holding one of `needles` anywhere inside.

    Deliberately blunter than the anonymizer's whole-token check: a fixture is
    published, so a handle inside a longer word still counts as a leak. A key
    that matches is reported as `<key>` so the path never echoes the value.
    """
    hits: list[str] = []
    if not needles:
        return hits

    def holds(value: str) -> bool:
        return any(needle in value for needle in needles)

    def walk(value: Any, path: str) -> None:
        if isinstance(value, str):
            if holds(value):
                hits.append(path or "$")
        elif isinstance(value, dict):
            for name, child in value.items():
                leaks = isinstance(name, str) and holds(name)
                label = "<key>" if leaks else str(name)
                where = f"{path}.{label}" if path else label
                if leaks:
                    hits.append(where)
                walk(child, where)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, f"{path}[{index}]")

    walk(node, "")
    return hits


def _hex_key(value: str) -> bytes:
    try:
        key = bytes.fromhex(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"--seed-key must be hex: {exc}") from exc
    if not key:
        raise argparse.ArgumentTypeError("--seed-key must not be empty")
    return key


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python scripts/refresh_fixtures.py",
        description=(
            "Rebuild tests/fixtures from the application's parsed blobs, "
            "anonymized under a key that exists only for this run."
        ),
    )
    parser.add_argument("--bucket", required=True, help="bucket holding parsed/{userId}/*.json")
    parser.add_argument(
        "--count", type=int, default=DEFAULT_COUNT, help="how many fixtures to write"
    )
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUT, help="directory to replace the fixture set in"
    )
    parser.add_argument(
        "--seed-key",
        type=_hex_key,
        default=None,
        help="hex anonymization key, for reproducing a run while debugging it",
    )
    return parser.parse_args(argv)


def _default_client() -> "S3Client":
    import boto3

    return boto3.client("s3", region_name=os.environ.get("AWS_REGION") or DEFAULT_REGION)


if __name__ == "__main__":
    raise SystemExit(main())
