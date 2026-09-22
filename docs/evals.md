# The golden question set

A prompt is code with no type checker in front of it. The agent's rules about
sample size, thin cells and observation rates are three paragraphs of English
that anybody can shorten by accident, and nothing in the test suite would go
red: the loop would still run, the tool would still validate, and the answers
would quietly get worse. This is the thing that notices.

`evals/golden.yaml` holds ten questions the fixture marts really answer.
`python -m pipeline.eval` runs each one through the real agent, scores three
checks, prints a table and exits non-zero if anything failed. Every run is an
MLflow run in the `agent-evals` experiment, so an agent change is tracked the
way a model change is.

## What a question looks like

```yaml
  - id: matchup_win_rate
    question: >-
      How does Dragapult control do against Alakazam / Toucannon, and over how
      many games?
    expect_tools: [query_marts]
    require:
      - Dragapult control
      - Alakazam / Toucannon
      - "re:1 game\\b"
      - "re:win|won"
    forbid:
      - deck inclusion
      - "re:[0-9a-f]{16}"
    notes: >-
      The plain case, and the one rule 1 exists for: the matchup is a single
      game, so the rate is meaningless without the denominator beside it.
```

Three checks, and a question passes only if all three hold.

**`expect_tools`** is a set. Every tool named has to have been called; the
order is not asserted and an unexpected extra call is reported and costs
nothing. Which tool an answer is built from is a fact about the answer. The
order the model works in, and whether it took a detour on the way, is style.

**`require`** is the facts. A plain entry is a case-insensitive substring; an
entry written `re:<pattern>` is a regular expression. Each one should be a
number, an archetype, a card or a sample size: something a wrong answer cannot
contain by accident. Nothing here requires a turn of phrase, because two
correct answers will not word a caveat the same way and a set that fails one
of them is a set people learn to ignore.

**`forbid`** is the claims. `[0-9a-f]{16}` is the shape of the irreversible
player token, forbidden on every question: the agent can read
`mart_player_summary` and must never hand the key over as an identity
(docs/data-handling.md). "deck inclusion" is forbidden on the questions where
the phrase has no honest use. On the two card-share questions it is not, and
cannot be, because the correct answer there says out loud that the number is
*not* a deck inclusion rate; what is forbidden on those two is asserting an
inclusion figure.

## What the ten cover

| id | what it is for |
| --- | --- |
| `matchup_win_rate` | a matchup rate with its denominator in the same sentence |
| `matchup_thin_sample` | `min_games_met` is false, and the answer has to say so |
| `weekly_record` | the weekly mart, at the week grain rather than the matchup one |
| `busiest_archetype` | two archetypes tie, so naming one of them is an invention |
| `week_coverage` | two counts in one answer, from one query |
| `card_text_lookup` | the retriever alone: printed card text is not play data |
| `card_text_and_marts` | the one card in both corpora, so both tools or no answer |
| `seen_rate_is_not_inclusion` | asks for an inclusion rate the corpus cannot give |
| `player_identity_refusal` | asks for a handle; there is none, and the token is not one |
| `matchup_with_no_games` | the pairing does not exist, so rule 4 is the whole answer |

Adding one is four steps, written out at the top of `evals/golden.yaml`. The
short version: check the number by hand against the fixture warehouse first,
record a run for it in `evals/transcript.yaml`, and bump `version`.

## Running it

Build the fixture marts once. Three commands, no credentials, about ten
seconds, and they write to a scratch directory rather than to `data/`:

```bash
export PIPELINE_DATA_DIR=/tmp/eval
HANDLE_HMAC_KEY=$(openssl rand -hex 32) uv run python -m pipeline.backfill \
  --source-dir tests/fixtures \
  --bronze-dir "$PIPELINE_DATA_DIR/lake/bronze" \
  --quarantine-dir "$PIPELINE_DATA_DIR/lake/quarantine"
uv run python -m pipeline.silver \
  --bronze-dir "$PIPELINE_DATA_DIR/lake/bronze" \
  --silver-dir "$PIPELINE_DATA_DIR/lake/silver" \
  --catalog tests/catalog.json
uv run python -m pipeline.gold --data-dir "$PIPELINE_DATA_DIR"
uv run python -m pipeline.card_index build --source tests/card_text.jsonl \
  --out "$PIPELINE_DATA_DIR/card_index" --embedder hashing
```

Then either of two runs, and they prove different things.

```bash
# The harness, the tools and the marts. No provider key, costs nothing.
uv run python -m pipeline.eval --fake evals/transcript.yaml \
  --warehouse "$PIPELINE_DATA_DIR/warehouse/meta.duckdb" \
  --card-index "$PIPELINE_DATA_DIR/card_index"

# The model and the prompt. This is the one that measures the agent.
op run --env-file=.env.op -- uv run python -m pipeline.eval \
  --warehouse "$PIPELINE_DATA_DIR/warehouse/meta.duckdb" \
  --card-index "$PIPELINE_DATA_DIR/card_index"
```

```
question                    result  failed  tools called
--------------------------  ------  ------  ------------------------
matchup_win_rate            pass    -       query_marts
matchup_thin_sample         pass    -       query_marts
weekly_record               pass    -       query_marts
busiest_archetype           pass    -       query_marts
week_coverage               pass    -       query_marts
card_text_lookup            pass    -       lookup_cards
card_text_and_marts         pass    -       lookup_cards,query_marts
seen_rate_is_not_inclusion  pass    -       query_marts
player_identity_refusal     pass    -       query_marts
matchup_with_no_games       pass    -       query_marts

10/10 passed
```

A failed question prints the tool it never called and the pattern it never
matched, under the table. `--json` prints the same report as one object. Exit
0 is every question passed, 1 is at least one failed, and 2 is a run that could
not be set up at all: a missing warehouse, an unreadable question set, a
provider that would not build. A failed question and a broken harness are
different news and do not share an exit code.

`--fake` replays `evals/transcript.yaml`, which is a recording of the tool
calls a competent run makes and the text it ends on. It is not an answer key
and the scorer never reads it: the recorded turns go through the real
`create_agent` graph, the real SQL gate and a real DuckDB query against the
fixture warehouse, so a green run there means the harness, the tools and the
marts are sound and says nothing at all about the model.

## The broken-prompt check

The claim that the five rules in `pipeline/prompts.py` are load bearing is only
worth something if taking them out is visible. `evals/broken_prompt.txt` is the
control: the same job description with the generated schema and the five rules
removed. `PRA_AGENT_SYSTEM_PROMPT_FILE` replaces the whole system prompt with a
file, for any entry point, and `--prompt-override` is that variable with a
flag in front of it.

```bash
uv run python -m pipeline.eval --fake evals/transcript.yaml \
  --prompt-override evals/broken_prompt.txt \
  --warehouse "$PIPELINE_DATA_DIR/warehouse/meta.duckdb" \
  --card-index "$PIPELINE_DATA_DIR/card_index"
# 0/10 passed
```

With a provider key and no `--fake`, the score falls for the reason that
matters: nothing tells the model to cite a sample size, to flag a thin cell or
to refuse to guess, and the answers stop carrying the facts the set requires.
With the replay model the score falls for a narrower reason, and it is worth
being precise about it: the fake follows exactly one instruction from the
prompt, which is that it will not call a tool the prompt never described, and a
prompt with no schema in it describes no tables. That is a real constraint on a
real model too, but it is one mechanism rather than five, so the replay version
of this check is a smoke test and the provider version is the evidence.

`tests/test_eval.py` asserts both halves without a key: that the override
really reaches the agent, and that the score drops when it is used.

## Tracking

One MLflow run per evaluation, in the `agent-evals` experiment, beside
`win-probability` and `win-probability-drift` and read with the same tool. The
tracking URI is resolved exactly as the trainer resolves it:
`MLFLOW_TRACKING_URI` when it is set, otherwise `file:./data/mlruns`, with
`--tracking-uri` overriding both and `--no-mlflow` turning the record off.

Parameters are the things that would change the score: the model, the sha256 of
the system prompt as it was actually rendered, the version of the golden file,
the path of any prompt override, whether it was a replay, and the commit.
Hashing the prompt rather than trusting the commit is the point of that
parameter: the generated half of the prompt comes out of
`dbt/models/marts/schema.yml`, so a column rename in dbt changes what the model
was told without touching a line of `pipeline/`.

Metrics are `passed`, `total`, `pass_rate`, and one 0/1 metric per question as
`q.<id>`, so the run table shows which question broke rather than only that
something did. The whole report goes up as `eval_report.json`.

## In continuous integration

`.github/workflows/agent-eval.yml`, on a weekly schedule, on a push to `main`
that touches `pipeline/agent.py`, `pipeline/prompts.py`,
`pipeline/card_index.py`, `pipeline/eval.py` or `evals/`, and on demand. Not on
every pull request: a run is tens of provider calls, it is noisy at ten
questions, and a check that costs money and flickers is a check people learn to
route around. The job builds the fixture marts and the card index the same way
this page does, scores the set, and keeps the MLflow directory as an artifact
whether it passed or failed.

With no `ANTHROPIC_API_KEY` secret set, the job logs a notice and goes green
rather than red. A clone with no key is the normal state of this repository,
and a red build for it teaches people to ignore red builds.

The pull-request gate is still `ci.yml`, which covers the harness for free:
`pytest -m dbt` runs the whole set with the replay model against the same
fixture marts and asserts ten out of ten, and the fast suite covers the scorer,
the shape of the question set and the prompt override.
