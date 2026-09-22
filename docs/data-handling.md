# Data handling

## Summary

Last updated 2026-09-21. The source is Play Rough Analytics, a private,
invite-only web application where members upload Pokemon Trading Card Game
(TCG) Live battle logs. This pipeline reads only the parsed per-game JSON blobs
from the application's private Amazon Simple Storage Service (S3) bucket,
replaces every player handle with a keyed one-way token before writing anything
to disk, and builds aggregate statistics from the result. Raw log text, real
handles, full decklists and the anonymization key are never published. A
member can have their games deleted by asking an admin; the deletion reaches
every pipeline layer on the next rebuild.

## What is collected

The application, not this pipeline, collects the data. Per uploaded game it
stores the raw log text exactly as exported (`raw/`), a parsed JSON blob with
segments, entries and per-side counters (`parsed/`), and a summary row
(players, winner, archetypes, play date) in its database. Player handles
appear in all three, as structured fields and inside free text such as segment
titles and entry text. A game can also carry a decklist: the uploader's own, and
the opponent's when the opponent chose to share their list in-game, which the
export then prints. The summary marks games that carry both complete lists with
`hasFullDecklists`. See [discovery.md](discovery.md) for the full shape.

## What the pipeline reads

Only `parsed/{userId}/{gameId}.json`, read-only, from the application's private
bucket. The pipeline never reads `raw/` log text. `userId` is the opaque
account id from the authentication provider; it is not a handle and is not
displayed anywhere.

## Anonymization

Anonymization happens in the bronze step, before any Parquet file is written.
Every player handle is replaced with

```
HMAC-SHA256(HANDLE_HMAC_KEY, handle)   truncated to 16 hex characters
```

HMAC is a keyed-hash message authentication code: a hash function (here
SHA-256, Secure Hash Algorithm 256) combined with a secret key, so the output
depends on both the input and the key. The same handle always maps to the same
token, so per-player analysis still works without the handle itself.

A keyed hash is used instead of plain SHA-256 because handles are short and
drawn from a small population. A plain hash of a handle is reversible in
practice: anyone with a list of known player names can hash each one and match
the tokens. With a secret key, the same attack requires the key.

Handles are rewritten in every string value of the blob, not only the
structured fields: segment titles, entry text, actor, owner and winner fields,
counter keys, unparsed lines and extras. After the rewrite, bronze scans the
blob for any remaining raw handle and quarantines the game rather than writing
it. There is no exception list; the operator's own handle is hashed too.

## Key handling and rotation

`HANDLE_HMAC_KEY` is 32 random bytes. It is stored in 1Password and injected
into the process environment with the 1Password command-line interface
(`op run`) for local runs. When a cloud consumer needs it, it moves to AWS
Secrets Manager. The key is never committed, never written to the lake and
never logged; `.env.example` lists the variable name only.

Rotating the key changes every token, which relabels every player. There is no
incremental path, so the rotation procedure is: generate a new 32-byte key and
store it; delete the bronze, silver and gold outputs and any derived model
training set; run a full bronze backfill from S3 with the new key; rebuild
silver and gold; retrain and re-register any model, since the player keys in
its training data no longer match.

## What is never published

"Published" means the public repository, any public artifact (marts written
back to the application, exported files, screenshots) and demos.

- Raw log text. The pipeline does not read it, so it cannot leak it.
- Real handles. Only tokens leave bronze; the leak check enforces this.
- Full decklists. No decklist appears in a public artifact. Inside the lake
  they are kept: an opponent's list reaches a blob only because the opponent
  chose to share it in-game, so it is consented data, and bronze stores it
  along with the uploader's own list. Every game is ingested, and
  `hasFullDecklists` counts those games rather than filtering them.
- Anything from the `dev` environment, which holds test data.
- The HMAC key.

Fixtures committed to the repository are anonymized stock-export games only,
with no opponent decklist in any of them.

## Deletion requests

A member asks an admin to delete a game or all of their games. The admin
deletes the game in the application, which removes its S3 objects. Today the
next backfill run rewrites each touched `play_date` partition in full from
what is in S3, so the deleted game drops out of bronze. Once the event-driven
path exists, the S3 delete event removes the game from bronze without waiting
for a backfill. Either way, silver and gold are rebuilt from bronze and follow.

The minimum a deletion flow needs, and what this design provides: a way to
ask (the admin), a way to find every copy (bronze partition by play date,
silver, gold, any model training set), and a rebuild.

## Retention

Bronze mirrors S3. Nothing the application has deleted is kept past the next
run. Silver, gold and model artifacts are derived and rebuilt from bronze.

## Consent record

All current uploaders are known personally to the operator and gave permission
for their games to be used for analytics. An in-app notice ("Your data" on the
Help page, and a line above the access request form) was added on 2026-09-21;
new members see it when requesting access. It states that uploaded games may
be used for anonymized, aggregate statistics, that names are replaced with
one-way tokens, that raw log text and full decklists are never published, and
that a member can ask an admin to delete their games at any time.

## Open items

- Dev bucket exclusion is by configuration (`PRA_BUCKET` points at one
  environment) and is not yet enforced in code.
- Deletion is batch (next backfill run) until the event-driven path lands.
