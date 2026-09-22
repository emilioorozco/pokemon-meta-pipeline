# Fixtures

`game-NN-XXXXXXXX.json` is one real parsed game, anonymized. They are stock
client exports without full decklists, chosen to cover the shapes the pipeline
has to handle: several opponent archetypes, both seats, a concede, the longest
game, a win and a loss. `tests/test_fixtures.py` checks every file and
`tests/test_contract.py` validates it against the contract; both skip when the
directory is empty.

Every handle is a keyed HMAC token, and the key is 32 random bytes drawn for
that one refresh run: it is never printed, never written and never reused, so a
fixture cannot be joined against a bronze row or used to confirm a guessed
handle. `summary.userId` becomes `user-<8 hex>`, and `uploadTokenId`,
`uploadClient`, `deckId`, `deckName`, `deckVersion`, `notes`, `tournamentId`,
`tournamentName`, `round`, `elo` and `matchId` are deleted. `gameId` stays: it
is a content hash of the log, not an account. `opponentDecklist` is never
present, in any form, because a pasted list is still someone else's list.

No fixture may ever contain a real handle. The tests enforce the token shape
(16 hex characters everywhere a handle can appear), so a file that slips through
with a name in it fails the suite rather than reaching a reviewer.

The card catalog fixture the silver tests join against is `tests/catalog.json`,
one directory up, not here: this directory is a directory of blobs that the
backfill ingests wholesale (`--source-dir tests/fixtures`), so any other JSON
file in it would be read as a game and quarantined. Its keys are the identities
silver resolves the fixtures' observed cards to, and its names are invented
("Fixture Card 1"), so a test can tell a catalog hit from a passthrough.

Refresh the whole set (replaces every `game-*.json` here):

```
uv run python scripts/refresh_fixtures.py --bucket <analytics parsed bucket>
```
