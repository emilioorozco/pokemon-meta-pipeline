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
building it constructs a provider client, imports LangChain and torch and
loads the card index, and a `/predict` that failed because the agent could not
be built would be the wrong failure. So a container that only ever answers
`/health` and `/predict` never imports torch. That first build is also the
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
`lookup_cards` has to import torch, load the baked bge model and embed one
query, over image layers Lambda fetches the first time they are touched.
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
over the network so they could be preloaded and then never used. A rebuild on
the new image has not been deployed yet, so there is no `after` column here;
what there is is 2.82 GB that no longer has to arrive.

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

Both columns are the PyPI-torch image. The four rows the CPU-only torch moved
were re-measured on both images and are in **The CUDA runtime is gone** above:
the image is 0.90 GB rather than 3.72 GB, the cold `/health` and the warm
`/warm` are unchanged, and the cold `/warm` is 3.86 s of `embedder_loaded`
rather than 4.50 to 5.87 s. The rows not repeated there were not re-measured.

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

| step | seconds | re-measured | what it is |
|---|---|---|---|
| card index load | 0.31 | 0.32 | reading the index off the lake into memory |
| first `lookup_cards` | 4.14 | 5.78 | importing torch, loading the baked bge model, embedding one query |
| second `lookup_cards` | 0.03 | 0.04 | the same query path with the model already resident |
| opening the warehouse | 0.15 | 0.26 | `duckdb_connect`, which downloads the file when the lake is `s3://` |
| one mart query | 0.02 | 0.02 | a `count(*)` through that connection |

Nothing in this change touches any of them, and the `re-measured` column is
the same block run again on top of it; the spread on the first
`lookup_cards` is what a laptop does, not a regression. Four of the five are
noise anyway. The one that matters is the first `lookup_cards`, and nearly
all of it is `import torch` plus building the model: the second call over the
same index is tens of milliseconds. That is the price of a local embedder,
paid once per container, and it is the number the CPU-only torch moved: on
this laptop by about two seconds, and on Lambda by whatever 2.5 GB of libraries
costs to fetch before they can be preloaded. Nothing here is on the `/predict`
path and nothing here is on the second question.

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
