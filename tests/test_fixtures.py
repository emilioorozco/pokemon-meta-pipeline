"""The committed fixtures: contract-valid, anonymized, scrubbed, and loadable into bronze.

These are real games, so this module is the guard that they carry nothing real:
every handle must already be a 16-hex token, the userId a short token of the
same kind, and none of the keys the refresh script scrubs may be present. A
fixture that fails here must not be committed; regenerate the set with
`uv run python scripts/refresh_fixtures.py --bucket <parsed bucket>`.

The whole module skips cleanly while there are no fixtures yet, so a clone that
has not run the script still has a green suite.
"""

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import jsonschema
import pytest

from pipeline.anonymize import anonymize, assert_no_handles, handles_in, token_for
from pipeline.bronze import BronzeRecord, read_smoke, write_partitions
from pipeline.config import REPO_ROOT
from pipeline.contract import ParsedBlobV2
from scripts.refresh_fixtures import GAME_ID_PREFIX, SCRUBBED_KEYS

FIXTURES_DIR: Final = Path(__file__).parent / "fixtures"
FIXTURE_FILES: Final = sorted(FIXTURES_DIR.glob("game-*.json"))
IDS: Final = [path.name for path in FIXTURE_FILES]
SCHEMA: Final[dict[str, Any]] = json.loads(
    (REPO_ROOT / "contract" / "parsed-blob.schema.json").read_text()
)
HEX16: Final = re.compile(r"^[0-9a-f]{16}$")
USER_ID: Final = re.compile(r"^user-[0-9a-f]{8}$")
OTHER_KEY: Final = b"a key no fixture was ever written under"
NOW: Final = datetime(2026, 9, 21, 12, 0, tzinfo=UTC)

pytestmark = pytest.mark.skipif(
    not FIXTURE_FILES,
    reason="no fixtures yet; run scripts/refresh_fixtures.py against the parsed bucket",
)


def load(path: Path) -> dict[str, Any]:
    blob: dict[str, Any] = json.loads(path.read_text())
    return blob


@pytest.mark.parametrize("path", FIXTURE_FILES, ids=IDS)
def test_a_fixture_satisfies_the_contract_and_the_schema(path: Path) -> None:
    blob = load(path)

    jsonschema.Draft7Validator(SCHEMA).validate(blob)
    parsed = ParsedBlobV2.model_validate(blob)

    assert parsed.summary.game_id[:GAME_ID_PREFIX] in path.name


@pytest.mark.parametrize("path", FIXTURE_FILES, ids=IDS)
def test_a_fixture_names_nobody(path: Path) -> None:
    """Every place a handle can appear holds a token instead; the shape is the guard."""
    blob = load(path)
    summary = blob["summary"]

    assert summary["players"] and all(HEX16.match(player) for player in summary["players"])
    assert blob["statsByPlayer"] and all(HEX16.match(name) for name in blob["statsByPlayer"])
    assert USER_ID.match(summary["userId"])
    for name in ("winner", "opponentName"):
        assert name not in summary or HEX16.match(summary[name])
    for segment in blob["segments"]:
        assert "player" not in segment or HEX16.match(segment["player"])


@pytest.mark.parametrize("path", FIXTURE_FILES, ids=IDS)
def test_a_fixture_is_a_stock_game_with_nothing_identifying_left(path: Path) -> None:
    blob = load(path)
    summary = blob["summary"]

    assert summary["exportVariant"] == "stock"
    assert summary["hasFullDecklists"] is False
    assert summary.get("excludedFromStats") is not True
    assert "opponentDecklist" not in blob
    assert [name for name in SCRUBBED_KEYS if name in summary] == []
    assert blob["segments"]


@pytest.mark.parametrize("path", FIXTURE_FILES, ids=IDS)
def test_anonymizing_a_fixture_again_only_maps_tokens(path: Path) -> None:
    """A second pass under any key is a pure token rename, never a content change."""
    blob = load(path)
    tokens = handles_in(blob)
    assert tokens and all(HEX16.match(token) for token in tokens)

    out = anonymize(blob, OTHER_KEY)

    assert assert_no_handles(out, tokens) == []
    renames = {token_for(token, OTHER_KEY): token for token in tokens}
    assert all(HEX16.match(new) for new in renames)
    restored = re.sub(
        "|".join(re.escape(new) for new in renames),
        lambda match: renames[match.group(0)],
        json.dumps(out, sort_keys=True),
    )
    assert restored == json.dumps(blob, sort_keys=True)


def test_every_fixture_lands_in_bronze(tmp_path: Path) -> None:
    """The set is a usable bronze batch, not just valid JSON."""
    records = [
        BronzeRecord(blob=ParsedBlobV2.model_validate(load(path)), source_key=path.name)
        for path in FIXTURE_FILES
    ]

    written = write_partitions(records, tmp_path / "bronze", NOW)

    assert sum(written.values()) == len(FIXTURE_FILES)
    assert sum(count for _, count in read_smoke(tmp_path / "bronze")) == len(FIXTURE_FILES)
