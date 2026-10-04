# The pace oracle

Ten files, one per committed fixture game, each holding the ten pace numbers
for **both** seats as the application's own `analyzeGame` computes them. The
keys are the application's, in its own spelling, so a reader can put a file
side by side with its definition and see the same words.

They are here to be disagreed with. `mart_archetype_pace` writes the same ten
definitions a second time, in SQL over silver, because the community average
is a question the nightly build answers and the application cannot. Two
writings of one definition drift, so `tests/test_gold.py` runs the per-seat
intermediate over the same ten games and fails when any number differs from
the one in this directory.

## Where they came from

A one-off script over the application's own function and the ten anonymized
fixture games, run read-only against a checkout of the private application
repository and never committed there. Its two lines worth knowing:

- `analyzeGame(blob, seat)` is called once per seat, not once per game. It
  reads `summary.wentFirst` as a statement about the seat it is handed ("did I
  go first"), and the stored flag is written from the uploader's point of
  view, so the flag is flipped for the seat that is not the uploader. Nothing
  else about the blob is touched.
- The uploader seat's ten numbers are therefore the ones the application
  already pins in its own snapshot test, and they match it exactly. That is
  the check on the script: only the other seat is new here.

## What is in them

Nothing but turn numbers, counts and two ratios. The games are the same stock
exports as `tests/fixtures/`, already anonymized under a key nobody kept: no
handle, no deck name anybody typed, no player. A pace file is ten numbers and
a game identifier, and the identifier is the anonymized one.
