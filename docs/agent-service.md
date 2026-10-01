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
| `PRA_SQL_GATE` | `1` | the judge is on: every generated query is read before it runs |
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
4. `pipeline.serve.create_app` runs, which pulls the MLflow store out of the
   lake into a temporary directory, loads whichever version holds the
   `production` alias and its archetype code map, and then throws the
   temporary copy away. An empty registry is a state, not a crash: `/health`
   answers with `model_loaded: false`.
5. Mangum wraps the application, and the request that started all of this is
   answered.

The agent is **not** built here. It is built on the first question, because
building it constructs a provider client and opens the warehouse, and a
`/predict` that failed to answer because the agent could not be built would be
the wrong failure. So a container that only ever answers `/health` and
`/predict` never imports torch.

All of that happens in the handler rather than at import. Lambda gives an
image's init phase about ten seconds and then re-runs the work inside the
invocation; this way the whole 60 s timeout is available for it, and a failure
is reported against a request instead of against an init. An error reading the
secret or the lake raises out of the handler after one `exception` log line, so
it appears in the function's error metric and in the caller's response rather
than as a timeout a minute later.

## The refresh rule

`pipeline.storage.local_file` downloads the warehouse once per process and
keeps the copy, on the assumption that nothing rewrites the object underneath a
running process. That holds for a command that ends. It does not hold for a
Lambda execution environment, which can live for hours and can therefore still
be answering from yesterday's warehouse after the nightly has replaced it.

So: the copy is kept for the life of the container, and at most once every ten
minutes the object's ETag is checked with one `HeadObject`. When it has
changed, the local file is dropped and unlinked and the agent over it is
discarded, and the next question rebuilds both from the new object.

The trade-off is staleness against cost, and ten minutes is where it was put:

- Checking on every invocation would be one `HeadObject` per question to
  detect a change that happens once a day.
- Never checking would mean a warm container answering indefinitely from a
  warehouse that no longer exists.
- Ten minutes means a container that was warm when the nightly landed can
  answer from yesterday's numbers for up to ten more minutes. For a metagame
  summary that is rebuilt once a day, that is invisible.

A `HeadObject` that fails is logged as a warning and treated as no change: a
transient S3 error should not throw away a working warehouse, and the next
check is ten minutes away. A question that is already in flight keeps reading
the file it opened, because an unlinked file on Linux stays readable until the
last descriptor closes.

## The image

`public.ecr.aws/lambda/python:3.12`, `linux/amd64`, the `ml`, `agent` and
`serve-lambda` extras exported from `uv.lock`, `pipeline/` copied in, and the
embedding model baked in. `CMD` is `pipeline.lambda_serve.handler`.

The embedder is baked because the retriever embeds the question locally: a cold
container that had to fetch the model from Hugging Face first would need
egress, would depend on a mirror that can be down, and would spend seconds
nobody is paying for. The build runs the same call the card index code makes,
`make_embedder(DEFAULT_MODEL)`, into `/opt/models`, and then sets
`HF_HOME`, `SENTENCE_TRANSFORMERS_HOME`, `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1`, so a cache miss at run time is a clear error rather
than a silent network call.

**The image is big, and most of it is CUDA that this function will never use.**
`sentence-transformers` pulls torch, and the torch that `uv.lock` pins is the
PyPI `torch==2.14.0`, whose Linux dependency set is the CUDA runtime:
`cuda-toolkit`, `nvidia-cudnn`, `nccl`, `triton` and the rest. They are not
dropped from the image, and dropping them would not work: the PyPI Linux
wheel's `libtorch_global_deps.so` links against those libraries and `import
torch` preloads them, so an image without them fails on the first embedding
rather than saving anything. A CPU-only torch is a `[tool.uv.sources]` entry
against the PyTorch CPU index plus a re-lock that every other consumer of this
lock file takes with it, which is a change of its own and was left out of this
one deliberately. It is the single biggest thing that could be done to the
numbers below.

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
  -e ANTHROPIC_API_KEY=not-a-real-key \
  -e PRA_SQL_GATE=1 -e PRA_SQL_GATE_LOW_CONFIDENCE=flag \
  -e PRA_SQL_GATE_ON_ERROR=allow -e PRA_LOG_FORMAT=json \
  pipeline-agent:measure

curl -s -X POST -d @health.json \
  http://localhost:9000/2015-03-31/functions/function/invocations
```

`health.json` is the function URL event shape, the same one
`tests/test_lambda_serve.py::function_url_event` builds. The lake is mounted at
the path it was built at, so the absolute paths MLflow's file store recorded
still resolve and the model really loads.

| measurement | value | how |
|---|---|---|
| image size | 3,717,640,668 bytes, 3.72 GB | `docker image inspect --format '{{.Size}}'` |
| cold start to the first `/health` 200 | 2.3 s | wall clock from `docker run` returning to the first invocation that answers, retrying while the emulator's port refuses |
| warm `/health` | 8.1 ms median, 7.6 to 11.4 ms over 20 | 20 sequential invocations after that, timed by the client |
| peak container memory | 292 MiB serving, 1021 MiB with the embedder loaded | `docker stats` sampled through both runs |

The runtime's own accounting agrees and splits the cold start in two: `INIT
REPORT durationMs: 905` for importing the handler module, and `Duration:
1429.87 ms` for the first invocation, which is where the application is
actually built and the model loaded. A warm invocation reports 2.3 to 3.9 ms,
so most of the 8 ms above is the client and the emulator's HTTP hop. The
peak memory is reached only once the embedder is resident, and 1021 MiB
against the function's 3008 MB leaves the headroom the agent's own working set
needs.

**The cold start is well under the 15 s the ticket set: 2.3 s to a `/health`
200, and about 7 s before a first `/ask` on a cold container reaches the
provider** (2.3 s of start, plus the fixed costs below). The 3.7 GB image is
not the problem it looks like, because Lambda lazy-loads image layers and this
function touches torch only on the `/ask` path.

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

| step | seconds | what it is |
|---|---|---|
| card index load | 0.31 | reading the index off the lake into memory |
| first `lookup_cards` | 4.14 | importing torch, loading the baked bge model, embedding one query |
| second `lookup_cards` | 0.03 | the same query path with the model already resident |
| opening the warehouse | 0.15 | `duckdb_connect`, which downloads the file when the lake is `s3://` |
| one mart query | 0.02 | a `count(*)` through that connection |

Four of those five are noise. The one that matters is the first
`lookup_cards`, and nearly all of it is `import torch` plus building the
model: the second call over the same index is 30 ms. That is the price of a
local embedder, paid once per container, and it is the number a CPU-only torch
would move. Nothing here is on the `/predict` path and nothing here is on the
second question.

Two things make these an optimistic floor for the deployed function. The lake
is a bind mount, so the warehouse open is a local file rather than a download
of the real thing out of S3, and the MLflow store is read in place rather than
pulled down; and a laptop core is faster than the share that comes with 3008
MB. The index and the corpus are the fixtures, 12 cards and 40 passages, where
production has a few thousand: the load scales with that and the embedding call
does not.

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

## What this does not do

No provisioned concurrency, so the first question after an idle period pays the
cold start. That is deliberate at this traffic: provisioned concurrency is
billed for every hour of the month whether or not anything asks a question, and
the reader waiting on the first question of the day is one person.

No streaming. The function URL is in buffered mode, so an answer arrives whole.
Response streaming would be worth having the day the agent's answers get long
enough to be worth watching arrive, and today they are a paragraph.

No `/metrics` scrape. The Prometheus exposition is still mounted and still
answers, but nothing scrapes a function that is not running; what the deployed
service reports is its log lines and the Lambda metrics around it. The traces
still work whenever `OTEL_EXPORTER_OTLP_ENDPOINT` names a collector.
