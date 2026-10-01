# The agent service

`POST /ask` and `POST /predict` without a laptop. The same FastAPI application
`python -m pipeline.serve` runs, packaged as a Lambda container image and put
behind a function URL, so the application repository can ask the metagame a
question over HTTPS and nothing has to be up in between.

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
