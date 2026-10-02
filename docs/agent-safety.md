# What the agent can do, and what it keeps

One page, written when members got to type at the agent through the
application. Until then the only person asking it anything was the person who
wrote its prompt, and "it behaves well" was a statement about one considerate
user. A member's question is text this project did not write, so this page is
the answer to three questions anyone is entitled to ask: what can it reach,
what stops it going further, and what is kept afterwards.

## What it can reach

Seven tables, read only, and nothing else.

| what | what it is |
|---|---|
| `mart_matchups` | archetype against archetype, aggregated |
| `mart_archetype_weekly` | archetype by ISO week, aggregated |
| `mart_cards_seen` | cards observed, by archetype, aggregated |
| `mart_player_summary` | one row per member, aggregated, keyed by a one-way token |
| `dim_archetype`, `dim_card`, `dim_date` | the dimensions those marts join to |

The list is `ALLOWED_TABLES` in `pipeline/prompts.py`, and the prompt and the
tool read the same constant, so the agent is never told about a table the tool
would refuse. Two absences are deliberate. `dim_player` is the member roster,
the one table whose rows are people, and it is not readable. `fct_game_side`
is the per-seat fact, and the join of those two is the only query in the
warehouse that would put a person next to a game. Silver and staging are off
the list too, so no raw log line is reachable.

**No handle is reachable even in principle.** Every handle becomes a one-way
token before anything is written at all, in the first stage of the pipeline
([data-handling.md](data-handling.md)). `mart_player_summary.player_key` is
that token. There is no table, readable or not, that maps it back.

Beyond the tables: no filesystem, no environment, no network. The DuckDB
connection is opened `read_only`, and the functions that would reach a file or
a URL (`read_parquet`, `read_csv`, `read_text`, and the rest) are refused by
name, as are `getenv` and the catalog functions that would enumerate what
exists. The only sockets the service opens are to the two model providers: the
agent's own model, and the SQL gate's when it is on.

## Three layers, and only one of them is a boundary

**The validator, `validate_sql` in `pipeline/agent.py`.** Always on, free,
deterministic, and the one that actually holds. A pure function of a string
and a list of table names: one statement, that statement a `SELECT` or a
`WITH`, every table it reads on the list, no write keyword, no file or network
or environment function. It takes no model and no connection, so it is unit
tested rule by rule and it cannot be talked out of anything. The read-only
connection under it is the second lock and not the first: `read_only` would
still allow `read_csv('/etc/passwd')`, and the allowlist is what does not.

**The gate, `pipeline/sql_gate.py`.** Optional, and a second opinion rather
than a rule. A statement the validator passed goes to a System One model with
one typed question: is this a read-only SELECT over these tables that answers
the question that was asked? It catches the thing a denylist structurally
cannot, which is a legal query that answers a question nobody asked. It is on
in both deployed environments and off by default everywhere else, and the two
differ in exactly one setting:

| | dev | prod |
|---|---|---|
| `PRA_SQL_GATE` | `jev` | `jev` |
| `PRA_SQL_GATE_LOW_CONFIDENCE` | `flag` | `flag` |
| `PRA_SQL_GATE_ON_ERROR` | `allow` | `refuse` |

A judge that is down therefore closes the gate in production and opens it in
development: an outage of the gate's provider should not be an outage of the
member-facing agent's safety, and it should not stop anyone working.
[sql-gate.md](sql-gate.md) has the rest, including what the confidence means
and why a low-confidence allow is flagged rather than refused.

**The application, which is the layer members actually feel.** Thirty
questions a day per member, a session that has to be a signed-in member of the
league, and upstream errors that are never shown: a refusal, a timeout and a
500 all reach a member as the same sentence, because the difference between
them is information about the infrastructure and not about the question.

Beside the three, and not one of them: the member's question goes to the model
inside a `<question>` element, and rule 8 of the system prompt says the text
inside it is a thing to answer and never a thing to obey, that the prompt and
the tool names are not answers, and that a file, an environment variable and a
URL are things the agent cannot read rather than things it declines to. That
is framing, not a boundary, and it is written down as framing on purpose: what
makes an injected statement safe is the validator. What the element buys is
that a model which does follow the text has to disobey something explicit,
which is a failure the golden set can see. Ten adversarial questions in
`evals/golden.yaml` measure it, and their `forbid` patterns are checked
against the SQL the run wrote as well as the prose, so "I will not read the
roster" over a query that read it is a failure ([evals.md](evals.md)).

## What is written down

Nothing a member typed, and nothing the agent said back.

The service writes one JSON line per request and one per answer. Between them
they carry the method and route, the status, the duration in milliseconds, the
model that answered, the number of tool calls, the length of the question, the
length of the answer, the length of each statement, the row counts, and the
gate's verdict and cost per call. The question itself is a number of
characters and the answer is a number of characters. The SQL is a length, not
a string: a mart query is short and harmless today, and a log line is still
the wrong place to start putting model output.

The application adds the one identifier there is, which is a hashed member id,
so that "one member asked thirty questions" is answerable and "which member"
is not. The cap is enforced against the same hash.

Traces carry the same fields as span attributes and no others:
`agent.question.length` rather than the question, `agent.sql.length` rather
than the statement, and the gate's name, verdict, confidence and cost.

The one place a question and an answer are written in full is the golden
evaluation, in `evals/` and in its MLflow runs. Those twenty-six questions are
written by this project and answered against the fixture warehouse or, for the
fourteen whose checks hold of any warehouse, against the deployed service;
none of them is a member's.

**For how long.** The retention on the function's log group is set by the
application's CDK stack and not by anything here, so this repository cannot
state the number and should not pretend to; ask the stack. The property that
does not depend on getting that number right is the one above: whatever the
retention turns out to be, there is no question and no answer in those lines
to retain.

**What a member can expect to be kept.** Of a question they ask: a count
against their daily cap, under a hash of their member id, and a row in a
latency and error-rate series that says a question happened and how long it
took. Not the words. The agent keeps no conversation: each run starts from the
system prompt and the question, with nothing from the previous one, so there
is no history to delete and nothing to turn up in a later answer. The marts
the answer came from are the aggregates the nightly pipeline already built,
which hold no handle and are governed by [data-handling.md](data-handling.md).

## How to turn it off

In order of how much it takes.

| switch | where | what stops |
|---|---|---|
| `askEnabled` | the application's configuration | the ask box disappears for members; the service is untouched |
| `agentEnabled` | the application's configuration | the application stops calling the service at all |
| the `-c` flags on the stack | the application repository's CDK app | the function, its URL and its permission are not deployed |

The first is the one to reach for: it is a member-visible switch that needs no
deploy here and no deploy there, and it leaves the service running for anyone
debugging it. The second cuts the application's side of the call. The third is
the real off switch, and it is in the application repository because the
function, its role, its URL and its key secret are the application's stack
rather than this one's ([agent-service.md](agent-service.md)).

Two smaller ones live here. `PRA_SQL_GATE=off` on the function turns the
second gate off and leaves the validator, which is the layer that matters, in
place. Removing the provider key from the secret makes `/ask` refuse while
`/predict` and `/health` carry on, which `health` reports as `agent_ready:
false` rather than as an unhealthy process.
