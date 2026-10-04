# The agent service

`POST /ask` and `POST /predict` without a laptop. The same FastAPI application
`python -m pipeline.serve` runs, packaged as a Lambda container image and put
behind a function URL, so the application repository can ask the metagame a
question over HTTPS and nothing has to be up in between.

This page is how it is deployed and what it answers with.
[agent-safety.md](agent-safety.md) is the other half: what the agent can
reach, the three layers between a member's question and the warehouse, what
is written down about a question and for how long, and how to turn the whole
thing off.

## What runs where

| piece | where it lives | who owns it |
|---|---|---|
| the application | `pipeline/serve.py`, unchanged | this repository |
| the Lambda entry point | `pipeline/lambda_serve.py` | this repository |
| the image | `Dockerfile.agent` | this repository |
| build and deploy | `.github/workflows/agent-image.yml` | this repository |
| the function, its role, its URL, the key secret | the application's CDK stack | the application repository |

The seam is the image tag, exactly as it is for the event consumer. The stack
creates the function from `:latest` on the image repository; a push to `main`
here builds the image, pushes it under two tags and moves the function to the
commit tag. An agent fix therefore ships from this repository with no deploy in
the application, and a change to the function's memory, timeout or permissions
ships from there with no rebuild here.

There are now two images of the serving application and they are not the same
image. `Dockerfile.serve` is the compose-era one: `python:3.12-slim`, the `ml`
extra only, a long-running uvicorn on port 8000, and `/ask` answering 503
because LangChain is not installed in it. It stays as it is, because it is what
`compose.yaml` builds next to the tracking server and next to Jaeger and
Grafana, and that is still the way to look at a trace. `Dockerfile.agent` is
the deployed one, and it carries the agent.

## The environment the function gets

Everything is set on the function by the application's stack. Nothing below is
written down in this repository, and `scripts/check_history.sh` fails the build
if anything ever is.

| variable | value | why |
|---|---|---|
| `PIPELINE_DATA_DIR` | the `s3://` lake root for the stage | one root, and every path under it is derived; see `pipeline/storage.py` |
| `AGENT_KEYS_SECRET_ARN` | a Secrets Manager secret | its string is one JSON object, `{"ANTHROPIC_API_KEY": "...", "JEV_API_KEY": "..."}` |
| `PRA_SQL_GATE` | `jev` | the judge is on: every generated query is read before it runs. The accepted values are `off` and `jev`, and nothing else; the first deployed function was set to `1` and every `/ask` refused |
| `PRA_SQL_GATE_LOW_CONFIDENCE` | `flag` | a low-confidence allow is recorded rather than turned into a refusal |
| `PRA_SQL_GATE_ON_ERROR` | `refuse` on prod, `allow` on dev | a judge that is down closes the gate in production and opens it in development |
| `PRA_LOG_FORMAT` | `json` | one object per line, which is what a CloudWatch Logs Insights filter reads |
| `AWS_REGION` | the platform's | set by Lambda; boto3 and `pipeline.storage` both read it themselves |

| setting | value | why |
|---|---|---|
| memory | 3008 MB | the embedder and the model are both in memory at once, and CPU on Lambda scales with memory |
| timeout | 60 s | one agent loop, several provider round trips; longer than any question should take and shorter than a stuck one |
| ephemeral storage | 2048 MB | `/tmp` holds the warehouse copy and the synced MLflow store |
| function URL | IAM auth, buffered | the caller signs with SigV4; buffered because the responses are small JSON objects |
| reserved concurrency | 2 | a ceiling on provider spend and on how many copies of the warehouse can be downloaded at once |

The keys are in a secret and not in the function's environment because an
environment variable is readable by anyone who can call `GetFunction`.
`pipeline.lambda_serve` reads the secret once per execution environment, keeps
it, and writes each value into `os.environ` **only if that variable is not
already set**. A key already in the environment therefore wins, which is what
keeps a local run under `.env.op` and the whole test suite free of any AWS
call. No value from the secret is ever logged; the log line names which
variables were filled and nothing else.

## What a cold container does

1. `configure_logging("serve")` installs the JSON formatter.
2. The DuckDB extensions baked into the image are copied into `$HOME`, which
   is `/tmp`. Without this the first `s3://` view in the warehouse would make
   DuckDB fetch `httpfs` over the network at the worst possible moment.
3. The provider keys are read from the secret.
4. `pipeline.serve.create_app` runs with `eager_model=False`, so it builds the
   routes and nothing else.
5. Mangum wraps the application, and the request that started all of this is
   answered.

Neither heavy thing is built here, and for the same reason twice: the cold
path belongs to whichever route the caller actually asked for.

**The model is not loaded.** `create_app`'s `eager_model` says where it is,
and only `pipeline.lambda_serve` passes False. Loading it means
`pipeline.storage.tracking_store` pulling the MLflow file store out of the
lake, and on the first deployed container that was 28.7 of the 29 seconds the
first `/health` took, for a model `/health` does not read. It is loaded by the
first `/model`, `/predict` or `/reload` instead, and `/health` answers
`model_loaded: false` until then without setting it off. `python -m
pipeline.serve` and `compose.yaml` still load it at startup, which is what
makes an unreadable registry a startup failure there rather than a surprise on
the first prediction.

**The agent is not built.** It is built on the first question, because
building it constructs a provider client, imports LangChain and loads the card
index, and a `/predict` that failed because the agent could not be built would
be the wrong failure. So a container that only ever answers `/health` and
`/predict` never imports LangChain or ONNX Runtime. That first build is also the
only warming this function can have: Lambda freezes a container between
invocations, so a background thread does no work and the way to pay once is to
pay inside an invocation and keep the result, which is what both holders on
`app.state` are for.

All of that happens in the handler rather than at import. Lambda gives an
image's init phase about ten seconds and then re-runs the work inside the
invocation; this way the whole 60 s timeout is available for it, and a failure
is reported against a request instead of against an init.

## What `POST /ask` answers with

The body is the answer and the evidence for it, because the application shows
a member what was looked up rather than asking them to take a paragraph on
trust. Four fields were added for that panel and none of the old ones changed.

| field | what it carries |
|---|---|
| `answer` | the prose, as before |
| `tool_calls` | the tally, as before: tool, a one-line `input_summary`, a row count. The evaluation scripts read this and it is not going to change shape |
| `model`, `usage` | the provider's model and token counts, as before, with `cache_read_input_tokens` and `cache_creation_input_tokens` added by the section below |
| `evidence` | `queries` and `cards`, below |
| `gate_summary` | what happened to the answer, over this run's queries |
| `latency_ms` | wall time of the whole call measured inside the service, so a question that had to build the agent reports what the caller waited for |
| `run_id` | which run's data answered |
| `context_used` | whether a page context was really placed in front of the question, for the `about this page` chip; see **Page context** below |
| `context_game_used` | whether the game summary in particular was placed, so a chip that names the game can be exact |
| `context_relevance` | `relevant`, `irrelevant`, `skipped`, or null when no game was sent |

`evidence.queries` is one object per statement, in the order the model wrote
them:

| field | what it carries |
|---|---|
| `sql` | the statement in full, not the shortened `input_summary` |
| `row_count` | how many rows it returned; 0 for a refusal |
| `rows` | the **first 10** rows as JSON values, every string cut to **500** characters |
| `gate` | `off`, `jev:allowed`, `jev:allowed_low`, `jev:refused` or `jev:error` |
| `refused_reason` | why there are no rows, or null |
| `refused_code` | the same reason in one word, or null |

A refused query is still in the list, with no rows and its reason, because
"the agent tried to read the member roster and was not allowed to" is
something a reader should see rather than an absence. The three ways a query
produces nothing all fill `refused_reason`: the validator refused it, the gate
refused it, or DuckDB would not run it. Only the first two are refusals as far
as `gate_summary` is concerned.

`refused_code` is the same thing from a closed set, for an application that
has to switch on it rather than read it. The sentence is written for the model
that has to write a better query; the word is written for the badge.

| code | what happened |
|---|---|
| `table_not_found` | the model named a table that dbt has never built, such as `mart_leaderboard`. Nothing was blocked: a guess missed, and the run almost always goes on to read the right table |
| `table_not_allowed` | a real relation of this warehouse that is off the allowlist, such as `dim_player` or `fct_game_side`. This is the boundary holding, and it is the one a reader should look at |
| `statement_not_allowed` | not a single read-only SELECT: two statements, a write keyword, or a file-reading function |
| `judge_low_confidence` | the Jev gate chose `allow` under its threshold and `PRA_SQL_GATE_LOW_CONFIDENCE=refuse` is set. A threshold to tune rather than a verdict |
| `judge_refused` | the Jev gate really said no |
| `error` | the gate could not be reached or read, or DuckDB would not run an allowed statement. The second of those is not a refusal and does not affect `gate_summary` |

The two table codes are the distinction PLA-198 was filed over. The same
validator catches both, which is why they arrived as one event for so long,
but they are not one event: a guessed name is the agent being imprecise and
correcting itself, and a blocked name is a privacy or correctness boundary
doing what it is there for. An application that draws the same error badge on
both tells a member that a correct answer was refused.

How the validator tells them apart: it reads a list of the dbt project's
models, which are the `.sql` files under `dbt/models/`, and a name that is not
one of them is a name nothing builds. **The list is generated at build time
and committed**, as `pipeline/warehouse_tables.py`, rather than globbed when
the question is asked. This image is why: `Dockerfile.agent` copies
`pipeline/` and the embedding model and not the dbt project, so the glob found
nothing on the deployed function and a naming-convention fallback answered in
its place, which is how two names nobody has ever built a table for came back
`table_not_allowed` on dev. `scripts/generate_warehouse_tables.py` regenerates
the list and a test fails when it has drifted from the glob
([sql-gate.md](sql-gate.md)).

The cheaper half of the fix is upstream of all of it. The prompt's schema
listing now closes with one line, carried in the `query_marts` tool
description too, so the model reads it as it writes a FROM clause:

> This list is complete. There is no leaderboard, rankings, season or summary
> table beyond it, so choose a name from it rather than inferring one.

Those four words are the ones the invented names were built out of. The
application's leaderboard is its rankings system and the warehouse does not
hold it; the nearest readable thing is `mart_player_summary`. A question with
"this season" in it is answered out of `mart_archetype_weekly` like any other,
because there is no season column to filter on. The eval set measures whether
the line works: `guessed_tables` counts the `table_not_found` refusals of a
run ([evals.md](evals.md)).

`evidence.cards` is one object per card the card tool matched, deduplicated by
name, set and number in first-seen order and capped at **10**: `name`,
`set_code`, `number` and `text`, where the text is the card's printed text
without the name and set over it, cut to 500 characters.

`gate_summary` is one word over the whole run, and it describes the answer the
member was given rather than the worst attempt behind it:

- if at least one query was allowed to run, it is the lowest-confidence
  allowed gate among those: `allowed_low` when the gate let one through under
  its threshold or errored and let it through, `allowed` when they ran under
  the gate, `off` when no gate judged them;
- `refused` only when every query that ran was refused, or when no query ran
  because the one that was tried was refused;
- `off` when the run asked the warehouse nothing at all.

So a `table_not_found` refusal followed by a query that ran is never
`refused`. The refused attempt stays in `evidence.queries` with its reason and
its code, so a receipt can still show the detour; what changes is the one word
the application draws a badge from. **This is a contract change for the
application.** A client that treated `gate_summary == "refused"` as "the data
query was refused" was right before and is still right: it now fires only when
the member really got no data. A client that wants to show "the agent tried a
table that does not exist" should read `evidence.queries[].refused_code` for
`table_not_found`, which is information rather than an error.

`run_id` is read from the warehouse: `mart_pipeline_health` carries the gold
stage's last run, which is the run that built these marts, and every stage of
a nightly shares one id, so it is the same id the publish stage wrote on the
rows the application already shows. That mart is a view over the run-metrics
Parquet in the lake, so a container that has the warehouse file and not the
lake cannot read it; the fallback is that warehouse's last-modified time as an
ISO string under the same key, which still answers "which night is this". Null
means there is no warehouse to ask.

**No handle reaches any of this.** The rows are whatever the seven allowlisted
marts hold, and none of their columns is a handle or a user id: the one
person-shaped column the agent can reach at all is the `player_key` of
`mart_player_summary`, which is the same irreversible token the pipeline
uses (docs/data-handling.md), and `dim_player`, silver and staging are off the
allowlist entirely. The bounds are the other half of this: 10 rows, 10 cards
and 500 characters a string keep an answer a response rather than an export.

`python -m pipeline.agent --evidence` prints the same object, so a question
asked on a terminal and the same question asked over HTTP can be compared
without allowing for two renderings; `--json` always carries it.

### Page context

`POST /ask` takes four optional fields beside `question`.

| field | what it is |
|---|---|
| `context` | where the member is in the application and what is on their screen, as plain text. At most **4,000** characters; a longer one is a **422** rather than a truncation, because a summary cut in half is a summary that says something else. Plain prose, not JSON |
| `context_game` | a redacted plain-text summary of the game the member is looking at, built by the application from that member's own log. A few hundred characters to about 1,500, at most **4,000**, and a longer one is a 422 for the same reason. This service never fetches a game |
| `context_first_line` | one sentence describing the same game, such as `Your Dragapult ex game against Gardevoir ex, you went second, lost in 9 turns`. At most **300** characters. It is the only part of the game the relevance decision is shown |
| `job` | the application's own router label, one of `meta`, `my_game`, `my_mistake`, `my_record`, `card_rules`, `out_of_scope`. An enum, so a typo is a 422 rather than a new category in a chart. It changes nothing about the answer today; the per-job playbooks are a later ticket |

**The relevance decision.** A game summary is only worth its place in the
context window when the question is about that game, and "which deck is best
this week" asked from a game page is not. So when `context_game` is present,
one typed Choice call goes to the same Jev client the SQL gate uses, with the
same key, the same base URL and the same five second budget
([sql-gate.md](sql-gate.md)), and asks whether the game on screen bears on the
question. It is given the question and `context_first_line`, and never the
summary, which is what keeps it one short call however long the game was. Its
three verdicts:

| verdict | what was placed | when |
|---|---|---|
| `relevant` | the route sentence, a blank line, then the game summary | the judge said the question is about this game |
| `irrelevant` | the route sentence alone; the summary is dropped | the judge said it is about something else |
| `skipped` | the route sentence, a blank line, then the game summary | no judge is configured (`JEV_API_KEY` unset), the call timed out or errored, or no `context_first_line` came with the game |

`skipped` attaches rather than drops, which is the opposite of how the SQL
gate fails and is deliberate. The gate stands in front of the warehouse, so a
broken gate should refuse. This stands in front of nothing: an irrelevant
game in the context is a few hundred characters the model ignores, and a
missing game on a question about that game is a worse answer. A clone of this
repository with no judge key answers every game question with the game in
front of it, which is what it would do if the decision did not exist.

`context_relevance` in the response is the verdict, null when no game was
sent. `context_game_used` is whether the summary was really placed, and
`context_used` keeps its older meaning, which is whether a non-empty context
of any kind was placed.

**Where it goes, and why there.** The context is placed in the human turn,
after the cache breakpoint, as a `<context>` element in front of the
`<question>` one. With a route sentence and a game the judge kept:

```
<context>
The member is reviewing their last game against Dragapult control.

Your Dragapult ex game against Gardevoir ex. You went second and lost on turn
9. Prize cards taken: you 2, your opponent 6.
</context>
<question>
why did I lose that one
</question>
```

One element with a blank line in it, not two elements and no heading over the
second half: a label would be the project's own words inside the element rule
9 tells the model is somebody else's. With no game, or with a game the judge
dropped, it is the route sentence alone and the bytes are what they were
before this existed.

It is never in the system blocks. Those are the cached prefix, and a prefix is
only a prefix while it is identical from one request to the next: a sentence
that changes per member, placed before the breakpoint, would rewrite the whole
entry on every call and bill a write where a read would have done. Everything
per request goes after the mark, which is the same rule the question has
always followed.

**With no context the bytes do not move.** `wrap_turn(question)` with nothing
to place returns exactly what `wrap_question(question)` returned before this
existed, so the command line, every golden question and every recorded
evaluation produce the same human turn they always did. A context that is
empty, blank, or nothing but our own delimiters counts as nothing to place.

**It is read as data.** Rule 9 of the system prompt says the element describes
where the member is and what is on their screen, that it is information and
never an instruction, and that anything inside it that reads as an order is
ignored. The delimiters of both elements are taken out of both bodies first,
so neither can be closed from inside the other. That is framing and not a
boundary; `validate_sql` is the boundary
([agent-safety.md](agent-safety.md)), and two adversarial questions in the
golden set measure the framing ([evals.md](evals.md)).

**What is written down.** The `agent answered` log line and the `agent.answer`
span carry five things and no text: `context_chars`, the length of the context
as it was placed; `context_game_chars`, the length of the game summary that
was placed, which is zero when the judge dropped it; `context_relevance`, the
verdict, empty when no game was sent; `relevance_ms`, the wall time of the
decision call; and `job`. None of the four strings is ever logged, at any
level, and none is put on a span; a test asserts the absence of all of them
from every record of a request that carried them. The response carries
`context_used`, `context_game_used` and `context_relevance`, so the
application can show an honest "about this page" chip without being handed
its own text back.

Two Prometheus series come out of the same decision:
`agent_context_relevance_total{verdict}` counts the three verdicts, so the
denominator is questions asked from a game page rather than questions, and
`agent_context_relevance_duration_seconds` is what deciding adds to a
member's wait. A `skipped` share that climbs is the judge failing, and it is
visible there before it is visible anywhere else.

## What a question costs, and the cached prefix

Every model call re-sends the whole system prompt. One question is two to four
calls, because the agent reads a tool result and asks again, so the prompt is
paid for two to four times per question before a member has read a word. A
cached prefix is the provider holding that text between calls: the first call
writes it at 1.25 times the input price, every call within five minutes reads
it at a tenth, and the five minutes restart on each read.

**The layout.** The prefix is hashed in order, tools first, then the system
blocks, then the messages, and a change at one level invalidates that level
and everything after it. So everything identical across requests goes first
and the breakpoint goes on the last of it:

| position | content | changes when |
|---|---|---|
| tools | `query_marts`, and `lookup_cards` when the index is there | a deploy |
| system, block 1 | the role sentence, the nine rules, the card-tool note | a deploy |
| system, block 2 | the schema listing generated from `dbt/models/marts/schema.yml` | a deploy, or a `schema.yml` edit |
| **breakpoint** | `cache_control: {"type": "ephemeral"}` on block 2 | |
| human turn | the `<context>` element when there is one, route sentence and game summary inside it, then the `<question>` element, and nothing of ours | every request |

The breakpoint goes on the last **stable** block, not on the last block.
Marking something that varies would rewrite the entry on every call and bill a
write every time instead of a read. Nothing per request is in the two system
blocks: the member's question and the page context travel in the human turn,
which is why one cache entry serves every member.

`pipeline.prompts.system_blocks` builds the blocks and
`pipeline.agent.build_agent` hands them to `create_agent` as a `SystemMessage`
whose content is a list of blocks. langchain-anthropic forwards
`cache_control` on a text block to the provider untouched.

**The minimum, and what it means here.** Claude Haiku 4.5's minimum cacheable
prefix is **4,096 tokens**
([platform.claude.com/docs/en/build-with-claude/prompt-caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)).
Below it the provider caches nothing, marked or not, and returns no error: the
only way to know is the `usage` fields.

Today's prefix is **below that**. The system prompt is 8,444 characters with
the card-tool note and 8,210 without, and the tool schemas are roughly 900
more, so at the four-characters-per-token rule this file already uses for
`MAX_PROMPT_CHARS` the prefix is an **estimated ~2,300 tokens**. That is an
estimate from a character count and not a measurement. The measurement is one
call, and it needs a provider key:

```bash
op run --env-file=.env.dev.op -- uv run python -c '
from langchain_core.messages import HumanMessage, SystemMessage

from pipeline.agent import CARD_TOOL, chat_model, marts_tools
from pipeline.config import CARD_INDEX_DIR, WAREHOUSE_PATH
from pipeline.prompts import system_blocks, wrap_turn
from pipeline.telemetry import build_metrics, build_tracer_provider

tracer = build_tracer_provider("count-tokens").get_tracer("count-tokens")
tools = marts_tools(
    warehouse=WAREHOUSE_PATH,
    card_index=CARD_INDEX_DIR,
    tracer=tracer,
    metrics=build_metrics(),
).tools
has_cards = any(tool.name == CARD_TOOL for tool in tools)
messages = [
    SystemMessage(content=system_blocks(with_card_tool=has_cards)),
    HumanMessage(content=wrap_turn("which decks are winning this week")),
]
print(chat_model().get_num_tokens_from_messages(messages, tools=tools))
'
```

`ChatAnthropic.get_num_tokens_from_messages` is the SDK's `messages.count_tokens`
behind a LangChain name. The tools go in because they are inside the prefix.
The question goes in so the call is shaped like a real one; what is compared
against the 4,096 is the prefix up to the breakpoint, so run it a second time
with the `HumanMessage` dropped and take that number. Write it here, replace
the estimate, and say it is a measurement.

**The honest expectation, until that number is 4,096 or more.** Nothing
caches. `cache_read_input_tokens` is zero on every call and
`cache_creation_input_tokens` is zero too, and that is the correct reading
rather than a bug in the wiring. **Do not pad the prompt to reach the
minimum**: paying for 1,800 tokens of filler on every call to make 2,300
tokens cheaper is a loss, and a prompt written to hit a number is a prompt
nobody can edit. The text that will carry the prefix over the line is text
that earns its own place, the per-job playbooks and the facts glossary of the
router work, and when that lands this section gets the new measurement and the
first non-zero reads.

**Reading the counters.** `pipeline.agent.token_usage` takes the two counts off
LangChain's `usage_metadata["input_token_details"]` and reports them under the
provider's own names, summed over the question's calls. They land in four
places: `usage` in the `/ask` body, `agent.usage.cache_read_input_tokens` and
`agent.usage.cache_creation_input_tokens` on the `agent.answer` span, the
Prometheus counter `agent_prompt_tokens_total{kind="cache_read"|"cache_creation"|"uncached"}`,
and the `usage` object of the service's `agent answered` log line. The three
counter kinds do not overlap: they are the provider's split of the input side,
and `cache_read` over their sum is the hit ratio.

The same ratio from the function's logs, over whatever period the console is
set to:

```
fields @timestamp, usage.cache_read_input_tokens as cache_read,
       usage.cache_creation_input_tokens as cache_write,
       usage.input_tokens as uncached
| filter msg = "agent answered"
| stats sum(cache_read) as read_tokens,
        sum(cache_write) as write_tokens,
        sum(uncached) as uncached_tokens,
        sum(cache_read) / (sum(cache_read) + sum(cache_write) + sum(uncached)) as hit_ratio
  by bin(1d)
```

A day of zeros in `read_tokens` with a non-zero `uncached_tokens` is the
prefix being under the minimum, which is today's expected answer. Zeros in all
three is a day with no questions.

The weekly evaluation reports the same two numbers over a whole run:
`Report.as_dict` carries `usage_totals` and the MLflow run logs
`cache_read_tokens` and `cache_creation_tokens`, so a prompt edit that quietly
moved a per-request string into the cached blocks, or broke the prefix, is a
step in the run table rather than a surprise on a bill.

## What is a failure and what is a state

A lake that cannot be reached while the app is being built raises out of the
handler after one `exception` log line, so it appears in the function's error
metric and in the caller's response rather than as a timeout a minute later.

Three things are **not** in that class, because each is one route's dependency
rather than the function. `/health` stays 200 and the fields carry the news:

| state | `/health` | `/ask` |
|---|---|---|
| the key secret is missing, unreadable, not JSON, or holds no value | 200, `keys_loaded: false`, `missing_keys` naming both | 503, `the agent has no provider key configured; $ANTHROPIC_API_KEY is not set` |
| `PRA_SQL_GATE` is a value the gate does not accept | 200, `agent_ready: false`, `agent_reason` naming the variable and the two accepted values | 503, the same sentence |
| the agent was built and the build failed | 200, `agent_ready: false`, `agent_reason` naming the failure | 503 naming it, and the **next** question builds again |

200 rather than 503 because the function is up, and a health check that went
red for a key nobody had pasted in yet would have whatever watches it
replacing a container that works. `/predict` keeps answering in all three.

A failed agent build is never cached. The first deployed container answered
every `/ask` for the rest of its life in 17 ms with the same
`GateConfigError`, and the fix was one environment variable; now a corrected
variable or a key that has arrived is live on the next question with no
redeployment. A secret read that produced nothing is not latched either, for
the same reason.

## The keepalive ping, and why it calls `/warm`

Something outside this repository pings the function every five minutes so a
reader is not usually the one paying for a cold container. That ping used to
go to `/health`, which does not work, because `/health` deliberately touches
nothing: it kept a container alive with none of the expensive things in it and
the next real question still paid for all of them. The first deployed `/ask`
about a card proved it, hitting the 60 s function timeout, because a first
`lookup_cards` has to load the embedder and embed one query, over image layers
Lambda fetches the first time they are touched. That load used to be the torch
and transformers imports as well, and the two sections below are what it took
to make it a 133 MB graph instead.
`GET /warm` is what the ping calls instead: it builds the agent, which is the
LangChain import and the tool construction, and runs one short fixed string
through the card tool's own embedder, so that the model is resident. It never
calls the provider, because that is a network call per question and warming it
would be spending money on a ping; it never raises, and it answers 200 with
the same dependency fields `/health` carries plus `agent_built`,
`embedder_loaded` and a `seconds` per step, so a ping that found nothing to do
says so in milliseconds and a ping that rebuilt a cold container says what it
cost. The embedder it warms is the one the tool holds, not a second copy:
`pipeline.agent.marts_tools` loads the index once and hands the same object to
both.

## The refresh rule

The nightly replaces two things a running container is holding.
`pipeline.storage.local_file` downloads the warehouse once per process and
keeps the copy; the agent reads the card index out of the lake into memory
once, when it is built. Both are kept on the assumption that nothing rewrites
them underneath a running process. That holds for a command that ends. It does
not hold for a Lambda execution environment, which can live for hours and can
therefore still be answering from yesterday's warehouse, and out of yesterday's
card index, after the nightly has replaced both.

So: one `ObjectWatch` per object, and at most once every ten minutes each one's
ETag is checked with one `HeadObject`. For the warehouse that is the file
itself; for the index it is the `meta.json` the nightly rewrites with the rest
of the directory. When either has changed the agent is discarded, and for the
warehouse the local file is dropped and unlinked as well, so the next question
rebuilds over what is in the lake now.

The index got its watch after the warehouse did, and the reason was a
deployment that spent an afternoon answering card questions with an apology:
the container came up while the lake still held an index of the previous
format, the agent was built without `lookup_cards`, and nothing made it look
again after the nightly rebuilt the index twenty minutes later.

The trade-off is staleness against cost, and ten minutes is where it was put:

- Checking on every invocation would be one `HeadObject` per question to
  detect a change that happens once a day.
- Never checking would mean a warm container answering indefinitely from a
  warehouse that no longer exists.
- Ten minutes means a container that was warm when the nightly landed can
  answer from yesterday's numbers for up to ten more minutes. For a metagame
  summary that is rebuilt once a day, that is invisible.

A `HeadObject` that fails is logged as a warning and treated as no change: a
transient S3 error should not throw away a working warehouse or a working
agent, and the next check is ten minutes away. A question that is already in
flight keeps reading the file it opened, because an unlinked file on Linux
stays readable until the last descriptor closes.

## A missing card tool is a state, not a finished build

An index that is absent, unreadable or of a format this code does not read is
a logged skip in `pipeline.agent.marts_tools`: the SQL half of the agent works
without it. What was missing was anything saying so afterwards. The holder
cached the half-agent as a success, `/warm` reported `agent_built: true` with
`embedder_loaded: false` and no reason, and "what does this attack do" came
back as "I do not have access to card text" for the life of the container.

Now the reason travels with the built agent and two fields carry it:

| field | on | meaning |
|---|---|---|
| `card_tool` | `/health`, `/warm` | whether the agent in hand answers card questions. False before anything has built one, and false when the one that was built came up without `lookup_cards` |
| `card_tool_reason` | `/health`, `/warm` | why there is none, naming what to rebuild; null when there is one |

`agent_ready` still means exactly what it meant, a provider key and a gate
setting the gate accepts, and it stays true while `card_tool` is false. The
two are separate on purpose: "`/ask` would refuse" and "`/ask` would answer,
without card text" are different afternoons, and the second one looked like
neither until it had its own field.

`/warm` is also the retry. A ping that finds the agent incomplete throws it
away and builds again, so an index rebuilt at any point in the night is picked
up within one ping of landing, and `/ask` is left alone, because a question
that rebuilt the graph each time the index was unreadable would pay the
LangChain construction per question to keep failing the same way.
`embedder_loaded` stays false the whole time, and `card_tool_reason` is what
says why.

## The image

`public.ecr.aws/lambda/python:3.12`, `linux/amd64`, the `ml` and `serve`
extras exported from `uv.lock`, `pipeline/` copied in, and the embedding model
baked in. `CMD` is `pipeline.lambda_serve.handler`.

The embedder is baked because the retriever embeds the question locally: a cold
container that had to fetch the model from Hugging Face first would need
egress, would depend on a mirror that can be down, and would spend seconds
nobody is paying for. A build stage runs
`scripts/export_query_embedder.py --out /opt/embedder`, which writes
`model.onnx`, `tokenizer.json` and an `embedder.json` naming the model and its
pooling, and the final stage copies that directory and sets
`PRA_QUERY_EMBEDDER_DIR` to it. The Hugging Face client lives in the build
stage only, so the deployed image cannot reach Hugging Face at all, which is
the stronger version of the offline flags the previous build set after its
download.

### The CUDA runtime is gone, and the image is a quarter of what it was

The image used to be 3.72 GB and most of it was CUDA this function will never
use. `sentence-transformers` pulls torch, and the torch `uv.lock` pinned was
the PyPI `torch==2.14.0`, whose Linux dependency set is the CUDA runtime:
`cuda-toolkit`, `nvidia-cudnn-cu13`, `nvidia-nccl-cu13`, `triton` and the rest,
some 2.5 GB of wheels no kernel here ever runs. Dropping them from the image
was never an option: the PyPI Linux wheel's `libtorch_global_deps.so` links
against those libraries and `import torch` preloads them, so an image without
them fails on the first embedding rather than saving anything. Preloading is
also what made them expensive rather than merely large. Lambda fetches an
image's layers the first time something touches them, so every page of those
libraries was pulled over the network during the first `import torch`, which
on a deployed container is inside the first question.

What was done instead is the index change: `pyproject.toml` declares the
PyTorch CPU index as `explicit`, `[tool.uv.sources]` sends `torch` there for
`sys_platform == 'linux'` only, and `torch` is named in the `agent` extra so
that the source applies to a direct dependency. Linux resolves
`torch==2.14.1+cpu`, which depends on no CUDA package at all; macOS and Windows
keep the PyPI wheel, which was already CPU-only there, so a development machine
installs what it always did. `uv.lock` holds both, one per resolution fork, and
has no `nvidia-*`, `triton`, `cuda-toolkit`, `cuda-bindings` or
`cuda-pathfinder` entry left anywhere in it.

The cost is in `Dockerfile.agent`: the export has to carry `--emit-index-url`
so the second index is named, and the install has to carry
`--index-strategy unsafe-best-match`, because uv otherwise takes a package only
from the first index that carries its name and would look for `2.14.1+cpu` on
PyPI. Every version in the exported file is an exact pin out of `uv.lock`, so
that strategy picks between indexes and never between versions.

Measured on the same laptop, both images built and run the same way, with the
emulator recipe in **Measurements** below:

| measurement | PyPI torch | CPU torch | change |
|---|---|---|---|
| image size | 3,717,661,405 bytes, 3.72 GB | **897,559,363 bytes, 0.90 GB** | 2.82 GB smaller |
| cold start to the first `/health` 200 | 1.12 s | **1.03 s** | noise |
| cold `/warm`, `seconds.embedder_loaded` | 5.87 s | **3.86 s** | 2.0 s |
| cold `/warm`, `seconds.agent_built` | 1.29 s | **0.97 s** | 0.3 s |
| cold `import torch` in the image | 2.32 s on the first ever run, 1.65 to 1.66 s after | **1.08 s, 1.07 to 1.10 s** | about 0.6 s |
| torch version installed | `2.14.0+cu130` | `2.14.1+cpu` | |

Read the three timings as the small half of it. A laptop reads those 2.5 GB
out of its own page cache; Lambda reads them over the network, once, on the
first touch, and the section below is what that looked like. The number that
carries is the image size, because the image is what gets paged in.

### And then torch and transformers went too

The CPU wheel was deployed and it was not enough. On the dev function a cold
`/health` answered in 26.8 s, `GET /warm` hit its 120 s ceiling and came back
as a timeout, and the breakdown said where it went: **105 s of it was the
Python import of torch and transformers**, with the bge weights loading in
under three seconds once that was done. The import is thousands of small files
and Lambda fetches an image's layers the first time something touches them, so
the cost is not the size of the wheels but the number of pages that have to
arrive before the first `import` statement returns. A CPU wheel is a smaller
pile of the same problem.

So the serving image no longer has either of them. `pipeline.query_embedder`
has the design: the same bge network exported to a single ONNX graph, run by
ONNX Runtime, tokenized by the `tokenizers` library out of the model's own
`tokenizer.json`, with the CLS pooling and the L2 normalization applied in this
repository because they are not in the graph. `sentence-transformers` stays on
the build side, where the nightly `build_card_index` embeds the whole corpus on
a runner with no cold start to pay, and the new `serve` extra is the list that
has neither framework in it. The two embedders agree to a minimum cosine of
0.9999999 over five fixture passages and five questions, with identical top-5
retrieval order, which `tests/test_query_embedder.py -m ml` asserts;
`tests/test_serve_imports.py` asserts the absence, by importing every serving
module and running the whole card lookup path in a subprocess and checking
`sys.modules`.

Same laptop, same fixtures, same emulator recipe as **Measurements** below, the
`before` column being the CPU-torch image this replaces:

| measurement | CPU torch | ONNX Runtime | change |
|---|---|---|---|
| image size | 897,556,630 bytes, 0.90 GB | **643,578,235 bytes, 0.64 GB** | 254 MB smaller |
| cold start to the first `/health` 200 | 0.885 s | 0.977 s | noise |
| cold `/warm`, wall clock | 5.51 s | **2.30 s** | 3.2 s |
| cold `/warm`, `seconds.embedder_loaded` | 3.672 s | **0.472 s** | 3.2 s |
| cold `/warm`, `seconds.agent_built` | 0.944 s | 0.932 s | noise |
| warm `/warm` | 39 ms, `embedder_loaded` 0.013 s | 39 ms, `embedder_loaded` 0.013 s | unchanged |
| first `lookup_cards` in the container | 3.419 s | **0.402 s** | 3.0 s |
| second `lookup_cards` | 0.015 s | 0.014 s | unchanged |
| `import pipeline.lambda_serve` | 0.68 to 0.75 s | 0.66 to 0.67 s | unchanged |
| `import torch, transformers` in the image | 1.73 to 1.79 s | not installed | |
| resident after one `/warm` | 684 MiB | **533 MiB** | 151 MiB |

Two of those rows are the point and one of them is a warning.

**`import pipeline.lambda_serve` did not move, and it was never going to.**
torch was not in the handler's import graph before this change either: it was
imported lazily, four frames inside the first `/warm`, which is why
`seconds.embedder_loaded` is the row that fell by 3.2 s and the init stayed
where it was. A dependency list cannot assert that, because a lazy import
resolves at run time out of whatever is installed; the subprocess test is what
asserts it.

**The 3.2 s on a laptop is the floor, and the deployed number is the one that
mattered.** On this machine the frameworks come out of the page cache. On
Lambda they came page by page over the network, which is how 3.2 s here was
105 s there. The image being 254 MB smaller is the same story told as a size:
it is 254 MB that no longer has to arrive.

## What AWS showed

The emulator numbers below were the only ones there were until the function
was deployed and called for real. The first cold request on the dev function,
3008 MB, timestamps from CloudWatch:

- **Init took 2.2 s, the provider keys were read at +0.5 s, "mlflow store
  downloaded" landed at +28.7 s, and `/health` answered at +29 s.** The MLflow
  file store sync was the entire cold start: `tracking_store` downloading
  `mlruns/` one object at a time. The dev store is small and production's is
  several hundred objects, so production would have been worse.
- **The first `POST /ask` took 15.7 s to fail.** That is what building the
  agent costs on a cold container: the torch and LangChain imports and the
  card-index load, over a lazily loaded 3.7 GB image. Warm questions are
  milliseconds.
- **It failed with `GateConfigError: $PRA_SQL_GATE is '1'; it has to be off or
  jev`**, and then kept failing with it in 17 ms for the life of the
  container, because the holder kept the failure. The function's environment
  is now `jev`, and the holder no longer keeps a failure.
- **Before the secret was filled, its placeholder (a random string, not JSON)
  made every route 502 with `SecretError`**, `/health` included. The stack has
  to create the secret before anyone can put a key in it, so that is a state
  every new deployment passes through.

Then the gate value was corrected and the keys filled in, and the first two
questions that reached the model showed the rest of it:

- **"Which archetype has the best win rate" answered 200 in 36 s.** About 15 s
  of that was building the agent on a cold container and the rest was the
  provider and four tool calls.
- **"What does Dragapult ex do?" hit the 60 s timeout and came back 502**,
  because it is the first question that calls `lookup_cards` and so the first
  that pays for the torch import, the bge load and one embedding, over image
  layers that are fetched on first touch. Locally that is four to six seconds.
  `GET /warm` and the five-minute ping onto it are the answer; see the
  keepalive section above.

`/model` and `/predict` answer 503 on dev, because the dev registry has no
`production` alias: the corpus is 13 games and the promotion gate has never
passed a candidate on it. That is correct and is left alone.

Then `/warm` was deployed and called, and it put a number on the thing the
emulator cannot show. On the dev function, 3008 MB, still on the 3.72 GB
image:

- **A cold `/health` answered in 11.5 s, of which 9.1 s was the init phase**,
  which is right under the ten seconds Lambda allows before it re-runs the
  work inside the invocation. Nothing in the init reads the lake or the
  provider; that is the handler module's import graph alone, paged in off the
  image.
- **`/warm` reported `embedder_loaded: 114.5 s`**, and the first `/warm` on a
  fresh container before it hit the 120 s ceiling and came back as a timeout.
  Against 4.5 s on a laptop, that is the whole of the gap: Lambda was fetching
  the CUDA libraries page by page over the network because `import torch`
  preloads them.
- **Once warm, a card question answered in 3.9 s and a SQL question in
  10.3 s**, which is the provider and the tool calls and nothing else.

That is what the CPU-only torch above was done for: the number that was making
the function unusable on a cold container was 2.5 GB of libraries being pulled
over the network so they could be preloaded and then never used.

Then the CPU-torch image was deployed, and it halved the problem rather than
removing it. On the dev function, 3008 MB, on the 0.90 GB image:

- **A cold `/health` answered in 26.8 s**, against 11.5 s on the 3.72 GB image.
  Not an improvement, and the direction is the warning: the init phase is
  whatever has to be paged in before the handler module finishes importing,
  and the variance between cold containers is wider than the change was.
- **`/warm` hit the 120 s ceiling and came back as a timeout again**, and the
  log said where: **105 s of it was the Python import of torch and
  transformers**, with the weights loading in under 3 s after that. The CUDA
  libraries were gone and the frameworks themselves were still thousands of
  small files arriving one page at a time.

Which is the measurement that decided it. The cost was never the model and was
never really the size: it was the number of files an `import` had to touch
before it returned. Dropping both frameworks from the serving image is the
section above, and what the deployed function pays for an embedding now is one
133 MB graph read once.

**Read the emulator numbers as a floor, not as what a member sees.** The two
differences are both large and both in the same direction. The emulator's lake
is a bind mount, so the warehouse open is a local file and the MLflow store is
read in place rather than pulled out of S3, which is the whole of the gap
between 2.3 s and 29 s on that first request. And Lambda lazy-loads image
layers: a page of torch that the emulator reads from the laptop's page cache
is fetched over the network the first time a real container touches it, which
is why the first `/ask` took 15.7 s there against roughly 7 s predicted here.
Nothing below is wrong; it is the shape of the cost and an upper bound on
everything that is not I/O.

## Measurements

Taken on a laptop with the AWS Lambda Runtime Interface Emulator, which the
base image ships, against a lake built from the committed fixtures. They are
not Lambda numbers: the emulator runs the same runtime client and the same
handler, but the filesystem is a bind mount rather than an S3 download and the
CPU is the laptop's rather than the share that comes with 3008 MB. Read them as
the shape of the cost, and as an upper bound on everything that is not I/O.

The lake used is a scratch directory, never `data/`:

```bash
SCRATCH=/tmp/pla171-lake
mkdir -p "$SCRATCH/catalog"
cp tests/card_text.jsonl "$SCRATCH/catalog/card_text.jsonl"
cp tests/catalog.json "$SCRATCH/catalog/cards.json"
HANDLE_HMAC_KEY=scratch-only-throwaway-key \
  uv run python -m pipeline.run_all \
    --source-dir tests/fixtures --data-dir "$SCRATCH" --stop-after build_card_index
```

```bash
docker build --platform linux/amd64 --provenance=false --sbom=false \
  -f Dockerfile.agent -t pipeline-agent:measure .
docker image inspect pipeline-agent:measure --format '{{.Size}}'
```

The container is started with no key secret and a dummy provider key, which is
how the image runs with no AWS at all; `/ask` is never called, because there is
no real key and no intention of spending one.

```bash
docker run -d --rm --name agent-measure --platform linux/amd64 \
  --memory 3008m -p 9000:8080 \
  -e PIPELINE_DATA_DIR="$SCRATCH" -v "$SCRATCH:$SCRATCH" \
  -e ANTHROPIC_API_KEY=not-a-real-key -e JEV_API_KEY=not-a-real-key-either \
  -e PRA_SQL_GATE=jev -e PRA_SQL_GATE_LOW_CONFIDENCE=flag \
  -e PRA_SQL_GATE_ON_ERROR=allow -e PRA_LOG_FORMAT=json \
  pipeline-agent:measure

curl -s -X POST -d @health.json \
  http://localhost:9000/2015-03-31/functions/function/invocations
```

`health.json` is the function URL event shape, the same one
`tests/test_lambda_serve.py::function_url_event` builds. The lake is mounted at
the path it was built at, so the absolute paths MLflow's file store recorded
still resolve and the model really loads.

The `before` column is the eager-model version, the one the first deployed
function ran; `after` is with the model moved off the cold path and the store
sync made concurrent. Same laptop, same fixtures, same image recipe.

| measurement | before | after | how |
|---|---|---|---|
| image size | 3,717,640,668 bytes | 3,717,654,311 bytes, 3.72 GB | `docker image inspect --format '{{.Size}}'` |
| cold start to the first `/health` 200 | 2.3 s | **1.12 s** | wall clock from `docker run` returning to the first invocation that answers, retrying while the emulator's port refuses |
| warm `/health` | 8.1 ms median, 7.6 to 11.4 | 8.0 ms median, 7.4 to 12.1 over 20 | 20 sequential invocations after that, timed by the client |
| the first `/model`, which now does the loading | part of the cold start | 1.43 s | one invocation after the warm run, timed by the client |
| cold `/warm` | no such route | 6.34 s, reported as `agent_built` 0.92 s and `embedder_loaded` 4.50 s | the first invocation of a fresh container, `/warm` instead of `/health` |
| warm `/warm` | no such route | 12 ms, both steps at or near zero | the next invocation of the same container |
| resident memory answering `/health` | 292 MiB | 178 MiB | `docker stats` after 20 `/health` invocations |
| resident memory with the model loaded | 292 MiB | 289 MiB | the same, after the first `/model` |
| peak with the embedder loaded | 1021 MiB | 937 MiB | the same, after a `lookup_cards` in the container |

Both columns are the PyPI-torch image. The rows the two later changes moved
were re-measured on each image and are in **The CUDA runtime is gone** and
**And then torch and transformers went too** above. The short version is that
the image is 0.64 GB rather than 3.72 GB, the cold `/health` has not moved at
any point, and the cold `/warm` is 0.47 s of `embedder_loaded` rather than the
4.50 s here. The rows not repeated there were not re-measured.

A second index has to exist to measure the `before` column now: the format
version went to 3 when `meta.json` gained the pooling, so the CPU-torch image
refuses an index this branch built. The baseline index was built inside the
baseline container, with that image's own `python -m pipeline.card_index
build`, into a copy of the scratch lake. Nothing in `data/` was read or written
for any of it.

The runtime's own accounting agrees and splits the cold start in two: `INIT
REPORT durationMs: 951` for importing the handler module, and `Duration:
1012.76 ms` for the first invocation, which is where the application is built.
Before this change that second number was 1429.87 ms and the model load was
inside it; it is now a separate `Duration: 1427.32 ms` against whichever
invocation first asks for the model, and on this function that is a route the
application repository does not call. A warm invocation reports 2.2 to 4.6 ms,
so most of the 8 ms above is the client and the emulator's HTTP hop.

A `/health` that answers before anything is loaded is also why the resident
figure drops: 178 MiB is the application and its imports, and the 111 MiB on
top of it is LightGBM and the booster, paid by whoever asks for a prediction.

**The cold start here is 1.12 s to a `/health` 200.** It is not 1.12 s on
Lambda, and the section above says by how much and why. What the change
removes from a deployed cold `/health` is the store sync entirely, which was
28.7 of 29 s; what remains on it is the init, the key read and the app build,
which were 2.2 s, 0.5 s and the rest of the 0.3 s. The concurrency in
`pipeline.storage` then applies to whoever does pay for the sync: the pool is
16 wide over the whole tree, and the store is small objects, so the
round-trip-bound part of it comes down by about that factor.

The fixed costs of the `/ask` path, timed inside the warm container rather than
by calling `/ask`, because there is no provider key here and a real question
would be a real charge. This is the work every first question on a cold
container pays for before the model is asked anything:

```bash
docker exec -w /var/task -e PYTHONPATH=/var/task -e PIPELINE_DATA_DIR="$SCRATCH" \
  agent-measure /var/lang/bin/python3.12 -c '
import json, time
from pipeline.card_index import CardIndex
from pipeline.config import CARD_INDEX_DIR, WAREHOUSE_PATH
from pipeline.storage import duckdb_connect

marks, question = {}, "which attacks discard energy from the bench"


def clock(name, fn):
    started = time.perf_counter()
    value = fn()
    marks[name] = round(time.perf_counter() - started, 3)
    return value


index = clock("card_index_load_s", lambda: CardIndex.load(CARD_INDEX_DIR))
clock("first_lookup_cards_s", lambda: index.search(question, k=3))
clock("second_lookup_cards_s", lambda: index.search(question + " again", k=3))
connection = clock("warehouse_open_s", lambda: duckdb_connect(WAREHOUSE_PATH))
clock("one_mart_query_s",
      lambda: connection.execute("select count(*) from main.mart_archetype_weekly").fetchall())
print(json.dumps(marks, sort_keys=True))
'
```

| step | seconds | re-measured | CPU torch | ONNX | what it is |
|---|---|---|---|---|---|
| card index load | 0.31 | 0.32 | 0.302 | 0.288 | reading the index off the lake into memory |
| first `lookup_cards` | 4.14 | 5.78 | 3.419 | **0.402** | loading the embedder and embedding one query |
| second `lookup_cards` | 0.03 | 0.04 | 0.015 | 0.014 | the same query path with the embedder already resident |
| opening the warehouse | 0.15 | 0.26 | 0.162 | 0.144 | `duckdb_connect`, which downloads the file when the lake is `s3://` |
| one mart query | 0.02 | 0.02 | 0.019 | 0.019 | a `count(*)` through that connection |

The first two columns are the PyPI-torch image, run twice; the last two are
the same block on the CPU-torch image and on this one.

The `re-measured` column is the same block run again on the same image; the
spread on the first `lookup_cards` is what a laptop does, not a regression.
Four of the five rows are noise in every column. The one that matters is the
first `lookup_cards`, which is the price of a local embedder paid once per
container, and it is the number both of the last two changes were about: the
CPU wheel took about two seconds off it, and dropping torch and transformers
took the remaining three. What is left, 0.402 s, is reading a 133 MB graph and
running one sequence through it. Nothing here is on the `/predict` path and
nothing here is on the second question.

Two things make these an optimistic floor for the deployed function. The lake
is a bind mount, so the warehouse open is a local file rather than a download
of the real thing out of S3, and the MLflow store is read in place rather than
pulled down; and a laptop core is faster than the share that comes with 3008
MB. The index and the corpus are the fixtures, 12 cards and 40 passages, where
production has a few thousand: the load scales with that and the embedding call
does not. The deployed first `/ask` was 15.7 s against the roughly 7 s this
block predicts, and the difference is image layers being fetched on first
touch.

The three degraded states in the table above were checked against the same
image, one container each, by starting it with no `ANTHROPIC_API_KEY`, then
with `PRA_SQL_GATE=1`, then with `PRA_SQL_GATE=jev` and no `JEV_API_KEY`. All
three answer `/health` 200 with the right fields and `/ask` 503 with the
reason, and none of them reaches a provider, which is why they can be run
with no key.

One thing about the fixture lake is worth saying plainly. `promote` refuses a
candidate that does not beat the archetype win-rate baseline, and on the
committed fixtures nothing does, so a `run_all` over them leaves the
`production` alias unset and `/health` would answer `model_loaded: false`. The
alias was moved by hand in the **scratch** registry, with `MlflowClient`, so
that the measurement includes a real model load. Nothing in `data/` was read or
written at any point.

## Calling it once it is deployed

The function URL is IAM authenticated, so a plain `curl` gets a 403: the
request has to be signed. The simplest signed call is not to the URL at all but
to the function, with the same event body a URL would have produced:

```bash
aws lambda invoke \
  --function-name pra-<stage>-meta-agent \
  --cli-binary-format raw-in-base64-out \
  --payload file://health.json \
  out.json
cat out.json
```

`out.json` holds the function URL response envelope: `statusCode`, `headers`
and `body`, where `body` is the JSON the endpoint returned. For a question,
the same call with the path and method changed and a body on it:

```bash
jq '.rawPath = "/ask"
    | .requestContext.http.path = "/ask"
    | .requestContext.http.method = "POST"
    | .headers["content-type"] = "application/json"
    | .body = "{\"question\": \"which archetype has the best win rate\"}"' \
  health.json > ask.json
aws lambda invoke --function-name pra-<stage>-meta-agent \
  --cli-binary-format raw-in-base64-out --payload file://ask.json out.json
jq -r '.body | fromjson | .answer' out.json
```

To call the URL itself, sign the request: `awscurl --service lambda`, or
anything else that does SigV4, against the function URL the stack prints. The
application repository does this from its own backend with its own role, which
is the only caller that matters.

The second caller is the evaluation. `python -m pipeline.eval --remote <url>`
signs each golden question whose checks hold of any warehouse the same way,
with botocore and whatever
credentials the environment holds, and scores the responses against the same
file a local run uses, which is how "the code in this checkout is correct"
stops being mistaken for "the container members are talking to is correct".
The questions that assert facts of the fixture corpus are reported as skipped
rather than put to a service that answers from the real one.
The `prod` job in `.github/workflows/agent-eval.yml` runs it weekly against
`vars.PIPELINE_AGENT_URL` and skips with a notice when that variable is
unset; the `lambda:InvokeFunctionUrl` grant on the continuous-integration
role is in the application's stack, beside the function it names.
[evals.md](evals.md) has the rest.

## What this does not do

No provisioned concurrency. That is deliberate at this traffic: it is billed
for every hour of the month whether or not anything asks a question, and the
reader waiting on the first question of the day is one person. The five-minute
ping to `/warm` is the cheap version of it, and the cheap version has a hole
in it: a ping keeps one container warm, and a second concurrent question gets
a cold one.

No streaming. The function URL is in buffered mode, so an answer arrives whole.
Response streaming would be worth having the day the agent's answers get long
enough to be worth watching arrive, and today they are a paragraph.

No `/metrics` scrape. The Prometheus exposition is still mounted and still
answers, but nothing scrapes a function that is not running; what the deployed
service reports is its log lines and the Lambda metrics around it. The traces
still work whenever `OTEL_EXPORTER_OTLP_ENDPOINT` names a collector.
