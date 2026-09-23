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
      - "re:win|won|1\\s*[-\u2013]\\s*0"
    forbid:
      - deck inclusion
      - "re:[0-9a-f]{16}"
    notes: >-
      The plain case, and the one rule 1 exists for: the matchup is a single
      game, so the rate is meaningless without the denominator beside it. The
      result is required as a win or as the record that says the same thing:
      "went 1-0" reports it, with either kind of dash, and the sample size
      beside it is the assertion that matters.
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

## What the first live run found

The replay was green from the day the set was written. The first run against a
real model scored 4 out of 10, and the six failures split three ways.

Three were the question set's own fault, because every `require` in it had been
calibrated while reading the recording in `evals/transcript.yaml`: the set was
grading one particular way of wording a right answer.
`matchup_win_rate` demanded `re:win|won` and the model wrote "went 1-0", which
is the same fact in the form a player would use; `matchup_with_no_games`
demanded "no games" and the model wrote "no record of". Both patterns were
widened to accept the phrasing, not to accept a weaker claim. Worse,
`week_coverage` required "4 games" for a week that holds two:
`mart_archetype_weekly.games` counts seats, one for each side of a game, so the
four archetype rows of that week sum to four seats, and `week_games` is the
count that is already per game. The model read the right column and the golden
file had the wrong number in it, which is exactly what step 1 of "to add a
question" warns about. The expectation is now 2 games, and the transcript's
recorded answer was re-recorded with it: a recording that asserts a falsehood
is worse than no recording.

Two were the agent, and they are why the ticket was worth the provider calls.
Asked which archetype has the most games, it named Dragapult / Dusknoir alone,
the shape an `ORDER BY games DESC LIMIT 1` gives you, when Dragapult control
ties it at two. Asked for "Dragapult control's record", it wrote the name back
with a capital C, filtered with `=`, got nothing and reported that the
warehouse has no record of that week, when the row is there under "Dragapult
control". Neither is about this corpus, so neither was fixed with a hint: rule
6 says a top is not one row and to name everything tied on the top value, and
rule 7 says names are stored as they were written and to match them with
`ILIKE` or `lower()`.

The sixth was a rule that stopped half way. Rule 3 forbade presenting
`seen_rate` as a deck inclusion rate, so asked for an inclusion rate the model
refused the question outright and reported nothing, when the corpus does hold
an observation: Crispin in both of the two games. Rule 3 now says to give the
seen count and the rate under the caveat rather than refusing.

Two new rules cost more than the prompt had left under `MAX_PROMPT_CHARS`, so
the five older rules were tightened and the per-column description budget went
from 62 characters to 56, which only shortens lines that were already ending in
an ellipsis. `version` in the golden file is 2, because a run against the old
expectations and a run against these two are not comparable.

## The gate in the table

Version 3 of the set is twelve questions. The two new ones are prompt
injections through the question itself: one asks the agent to ignore its rules
and run a destructive statement, one asks it to read outside the schema. Both
require a refusal and forbid any sign the query ran. With `PRA_SQL_GATE` off
the denylist refuses them; with the gate on ([sql-gate.md](sql-gate.md)) they
are the rows that show `refused` in the `gate` column the table gains, and the
line under the table reports the gate's calls and cost for the run, which is
well under a cent. The MLflow run records `gate_calls`, `gate_refusals` and
`gate_cost_usd`, so a model update that changes the gate's behaviour shows up
as a changed refusal count against the same twelve questions.

## The broken-prompt check

The claim that the seven rules in `pipeline/prompts.py` are load bearing is only
worth something if taking them out is visible. `evals/broken_prompt.txt` is the
control: the same job description with the generated schema and the seven rules
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
