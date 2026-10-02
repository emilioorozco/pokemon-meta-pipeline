# The nightly run

`.github/workflows/nightly.yml` runs the batch half of the pipeline on a
schedule, on a GitHub-hosted runner, against the lake in S3. There is no
server, no scheduler and no machine that has to be up: the workflow is the
cron, the runner exists for the length of the run, and the credentials exist
for less.

## What runs, and when

| when | cron (UTC) | what it runs |
|---|---|---|
| every day | `17 10 * * *` | silver, gold, train, promote, drift, build_card_index, quality gate, publish |
| every Sunday | `17 9 * * 0` | the same, with `backfill` in front of it |

Both are the same job over the same list, `python -m pipeline.run_all`, which
is the same command a person runs on a laptop. The difference is one
environment variable. The nightly run sets `PRA_INGEST_MODE=consumer`, which
`run_all` reads and turns the backfill into a logged skip: the event consumer
(`pipeline.lambda_consumer`, and `docs/stages.md` section 1c) lands each parsed
game as the producer writes it, so bronze is already current and a backfill
would be a second pass over rows that are there. The weekly run sets
`PRA_INGEST_MODE=backfill` instead and lists the bucket, as the safety net for
a notification that never fired, a window the consumer was broken for, or
anything else the queue lost. A `run_all` summary says which happened, with
the reason next to the skip.

Two stages report a skip on most nights and that is not a fault.
`build_card_index` needs the card-text corpus, which `scripts/fetch_card_text.py`
downloads and which is not on the runner; `train`, `promote` and `drift` skip
when `features_turn` is empty. Each says so in the summary.

## Running it by hand

```bash
gh workflow run nightly.yml -f stage=dev
gh workflow run nightly.yml -f stage=prod -f full_backfill=true
gh run watch                                   # or the Actions tab
```

`stage` picks the GitHub Environment, `dev` or `prod`, and defaults to `prod`,
which is what the two schedules use. `full_backfill` puts the backfill in front
of a run that would not otherwise have it, which is the way to catch up after
the consumer has been down.

Runs queue rather than cancel each other: the concurrency group is per
environment and `cancel-in-progress` is false, because two runs writing the
same lake would race and a half-written set of silver partitions is worse than
a late run.

## What it expects, and where that is set

Every identifier is a variable on a **GitHub Environment** named `dev` or
`prod` (Settings, Environments, then Variables). None of them is in this
repository, and none of them may be: `scripts/check_history.sh` fails the build
on a bucket name, a table name, a role ARN or an account id in any file or in
any added line.

| variable | what it is |
|---|---|
| `AWS_REGION` | the region everything is in |
| `PIPELINE_CI_ROLE_ARN` | the role the OIDC token is traded for, trusted to this repository |
| `PIPELINE_READER_ROLE_ARN` | the role chained from it, holding the lake, the parsed bucket and the insights table |
| `PIPELINE_DATA_DIR` | the lake root, an `s3://bucket/prefix` |
| `PRA_BUCKET` | the parsed-blob bucket the backfill lists |
| `PRA_PREFIX` | the prefix under it |
| `PRA_INSIGHTS_TABLE` | the DynamoDB table the publish writes |
| `HANDLE_HMAC_KEY_SECRET_ID` | the Secrets Manager secret holding the anonymization key |

An environment with any of them unset skips the whole job with a
`::notice::` naming what is missing, so a fork and a clone see a green run
rather than a red one for not having an AWS account. `tests/test_nightly_workflow.py`
asserts that the list checked in the gate is exactly the list the rest of the
file reads, so a variable added below cannot be forgotten above.

### Credentials, and the one-hour ceiling

Two steps, no stored key. The job asks GitHub for an OpenID Connect token,
trades it for `PIPELINE_CI_ROLE_ARN`, and from that session assumes
`PIPELINE_READER_ROLE_ARN` with `role-chaining: true`. The split keeps the
trust policy (who may assume anything here) apart from the permission set
(what a pipeline run may touch), so widening the second is not an edit to the
first.

A role-chained session is capped at one hour by AWS, whatever duration is
asked for, which is why the workflow asks for `role-duration-seconds: 3600`
and caps itself at `timeout-minutes: 55`. A run that needed longer would lose
its credentials part way through a stage. The fix then is not a longer session;
it is a machine that is not a shared runner, which is what
[docs/orchestration-on-aws.md](orchestration-on-aws.md) designs.

### The anonymization key

`HANDLE_HMAC_KEY` is not a GitHub secret and should not become one. Only the
stages that write bronze anonymize anything, which is the backfill and nothing
else, so the nightly run never holds the key at all. The weekly run reads it
from Secrets Manager with the credentials it already has, masks it in the next
line, and exports it for that job only. One copy of the key, in one place, with
one rotation.

## Reading last night's run

Actions tab, the `Nightly` workflow, the latest run. The job summary is the
page to read first: it names the run identifier, the environment, whether the
backfill ran, what the quality gate said, whether the publish ran, and it
quotes the whole `run_all` stage table underneath.

That run identifier is the thread through everything else. It is the value in
the `run_id` column of every `run_metrics` row the run wrote (one per stage,
queryable through the `ops` dbt models), it is on every log line the run
emitted, and it is the identifier shown on `/insights` for the numbers the
publish put there. An answer to "where did this number come from" is one
filter.

Two artifacts are attached to every run, kept for thirty days:
`run_summary.txt`, the stage table with the run identifier and the exit code on
top, and `drift_report.md`, fetched back out of the lake because that is where
the drift stage writes it. Both are uploaded even when the run failed, because
the run that failed is the one whose output somebody needs.

## When it is red

**The quality gate refused.** `python -m pipeline.quality_gate` exits 1 when
any stage's last run failed or any stage's quarantine rate is over the
threshold, and `run_all` stops at the first failing stage, so the publish never
ran. Nothing was written to the insights table and the last good run's numbers
are still what `/insights` serves. Read the job summary for the stage the gate
named, fix that, and either wait for the next night or run the workflow by
hand. There is nothing to roll back.

**Something broke.** A stage that crashed, an expired credential, a runner that
ran out of disk. The exit code in the summary is the failing stage's own, and
the step log has the traceback. Reruns are safe: every stage replaces the
partition it writes rather than appending to it, so running the same night
twice lands the same rows. Use the Actions tab's rerun, or
`gh run rerun <id>`.

**It skipped with a notice.** An environment variable is unset. The notice
names which.

## Why a cron in CI and not an orchestrator

Because the batch stages are a straight line with one failure rule, they run
once a night, and the machine they need exists for forty minutes a day.
Airflow, or the Step Functions state machine in
[docs/orchestration-on-aws.md](orchestration-on-aws.md), buys retries per task,
a real graph and an execution history, and costs a scheduler that has to be
running at three in the morning whether or not anything happened, plus the
cluster, the task role and the database that come with it. A scheduled workflow
buys none of that and costs nothing, and it is the honest choice right up to
the day a stage needs its own retry, its own machine, or more than an hour. The
design document is the plan for that day, and the stage commands do not change
when it arrives: an orchestrator would call exactly the commands this workflow
calls.
