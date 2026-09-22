"""The daily pipeline as an Airflow DAG: one task per stage command, nothing else.

Every task is a `BashOperator` running `python -m pipeline.<stage>` in the
repository, which is bind mounted at `/opt/pipeline` by `compose.yaml`. That is
the whole design. No `PythonOperator` importing the package into the scheduler's
interpreter, no logic in this file that is not scheduling: the stages are
command-line programs with their own arguments, exit codes and tests, and a DAG
that called them as functions would be a second way to invoke them that could
drift from the first. The commands here are the commands in the README, and
`python -m pipeline.run_all` runs the same list without Airflow at all.

The interpreter is `PIPELINE_PYTHON`, a virtual environment the image builds at
`/opt/pipeline-venv` with the project's dependencies in it, separate from the
one Airflow runs in. Airflow and this pipeline both pin large parts of the same
dependency tree (Jinja, SQLAlchemy, Flask, pandas) and resolving them together
is a fight nobody needs to have; keeping them apart costs one environment
variable. The code still comes from the bind mount, because the tasks run with
`cd /opt/pipeline` and the current directory is first on `sys.path`, so editing
`pipeline/silver.py` on the host changes what the next task run executes.

**The run identifier.** Every task exports `PRA_RUN_ID={{ run_id }}`, so all the
stages of one DAG run write their `run_metrics` rows and their log lines under
Airflow's own identifier for that run. Finding out what the 06:00 run did is one
filter on either, and it is the same string in the Airflow UI and in the
warehouse.

**Parameters.** `source_dir` empty (the default) reads the production S3 bucket;
set it to `tests/fixtures` to run the committed games instead, which needs no
AWS account. `ingest_mode` is `backfill` or `consumer`; `consumer` means the
event-driven ingest is landing bronze already, so the first task becomes a
logged no-op rather than a second pass over the same bucket.

**The graph.** Linear, because the data is: bronze feeds silver feeds the
warehouse feeds the model. The one branch is `build_card_index`, which needs the
warehouse and nothing after it, so it runs beside the model stages instead of
delaying them. `quality_gate` is last because it judges every row the run wrote.
"""

from __future__ import annotations

import os

import pendulum
from airflow.models.dag import DAG
from airflow.models.param import Param

try:  # Airflow 3 moved the operator out of core and into the standard provider.
    from airflow.providers.standard.operators.bash import BashOperator
except ImportError:  # pragma: no cover - whichever import the installed version has
    from airflow.operators.bash import BashOperator  # type: ignore[no-redef]

DAG_ID = "play_rough_pipeline"
# Where compose mounts the repository, and the interpreter that has the
# project's dependencies. Both are overridable from the environment so the same
# file works in a container that lays them out differently.
PIPELINE_DIR = os.environ.get("PIPELINE_DIR", "/opt/pipeline")
PYTHON = os.environ.get("PIPELINE_PYTHON", "/opt/pipeline-venv/bin/python")

# Passed to every task. `PRA_RUN_ID` is the whole reason this dictionary exists;
# the rest is the configuration the stages read, forwarded from the container so
# the DAG holds no secret and no path of its own. A variable the container does
# not set arrives as an empty string, which every stage already treats as unset.
TASK_ENV = {
    "PRA_RUN_ID": "{{ run_id }}",
    "PRA_INGEST_MODE": "{{ params.ingest_mode }}",
    "PIPELINE_DATA_DIR": os.environ.get("PIPELINE_DATA_DIR", f"{PIPELINE_DIR}/data"),
    "MLFLOW_TRACKING_URI": os.environ.get("MLFLOW_TRACKING_URI", ""),
    "HANDLE_HMAC_KEY": os.environ.get("HANDLE_HMAC_KEY", ""),
    "PRA_BUCKET": os.environ.get("PRA_BUCKET", ""),
    "PRA_PREFIX": os.environ.get("PRA_PREFIX", ""),
    "PRA_SPARK_MASTER": os.environ.get("PRA_SPARK_MASTER", ""),
    # Logs that a collector reads, not a terminal: the tasks have no teletype,
    # so this only pins what would already be the default.
    "PRA_LOG_FORMAT": "json",
    "AWS_REGION": os.environ.get("AWS_REGION", ""),
    "AWS_PROFILE": os.environ.get("AWS_PROFILE", ""),
}

DEFAULT_ARGS = {
    "owner": "pipeline",
    "retries": 0,
    "retry_delay": pendulum.duration(minutes=5),
}


def stage(command: str) -> str:
    """One stage command, run from the repository with the project's interpreter."""
    return f"cd {PIPELINE_DIR} && {PYTHON} -m pipeline.{command}"


with DAG(
    dag_id=DAG_ID,
    description="Bronze to silver to gold to model, once a day, one run identifier.",
    default_args=DEFAULT_ARGS,
    schedule="@daily",
    start_date=pendulum.datetime(2026, 9, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    tags=["pipeline", "play-rough-analytics"],
    params={
        "source_dir": Param(
            "",
            type="string",
            title="Source directory",
            description=(
                "Read the parsed blobs from this directory instead of S3. Empty means S3; "
                "tests/fixtures runs the committed games with no AWS account."
            ),
        ),
        "ingest_mode": Param(
            "backfill",
            type="string",
            enum=["backfill", "consumer"],
            title="Ingest mode",
            description=(
                "backfill lists the bucket and lands what it finds; consumer means the "
                "event-driven ingest is already landing bronze, so the first task is a no-op."
            ),
        ),
    },
) as dag:
    dag.doc_md = __doc__

    # The one task that talks to the network and the only one worth retrying on
    # its own: a listing or a GET can fail for a reason that is gone five
    # minutes later, while a failed Spark job or a failed dbt test will fail the
    # same way however many times it is retried.
    backfill = BashOperator(
        task_id="backfill",
        bash_command=(
            "{% if params.ingest_mode == 'consumer' %}"
            "echo 'ingest_mode=consumer: the SQS consumer lands bronze, nothing to backfill'"
            "{% else %}"
            + stage("backfill")
            + "{% if params.source_dir %} --source-dir {{ params.source_dir }}{% endif %}"
            "{% endif %}"
        ),
        env=TASK_ENV,
        append_env=True,
        retries=1,
        retry_delay=pendulum.duration(minutes=5),
    )

    spark_silver = BashOperator(
        task_id="spark_silver",
        bash_command=stage("silver"),
        env=TASK_ENV,
        append_env=True,
    )

    # Three tasks over one command. `--steps` splits dbt's build from dbt's
    # tests so a failed test is a node a reader can retry by itself, and
    # `--stage-name` keeps the three `run_metrics` rows apart, since all three
    # run under the same run identifier and would otherwise write one file.
    dbt_run = BashOperator(
        task_id="dbt_run",
        bash_command=stage("gold --steps run --stage-name gold_run"),
        env=TASK_ENV,
        append_env=True,
    )

    dbt_test = BashOperator(
        task_id="dbt_test",
        bash_command=stage("gold --steps test --stage-name gold_test"),
        env=TASK_ENV,
        append_env=True,
    )

    # `dbt_run` already built the feature table, since `dbt run` builds the
    # whole project. This rebuilds the `ml` tag alone, which makes the model's
    # input a named node in the graph rather than something that happened inside
    # another task, and makes a feature change one task to rerun.
    build_features = BashOperator(
        task_id="build_features",
        bash_command=stage("gold --steps run --select tag:ml --stage-name gold_features"),
        env=TASK_ENV,
        append_env=True,
    )

    train = BashOperator(
        task_id="train",
        bash_command=stage("train"),
        env=TASK_ENV,
        append_env=True,
    )

    promote = BashOperator(
        task_id="promote",
        bash_command=stage("promote --candidate latest"),
        env=TASK_ENV,
        append_env=True,
    )

    drift = BashOperator(
        task_id="drift",
        bash_command=stage("drift"),
        env=TASK_ENV,
        append_env=True,
    )

    # The retriever index is a later ticket. The task is here so the finished
    # shape of the run is visible in the graph, and it checks for the module
    # rather than assuming it: the day `pipeline/card_index.py` lands, this
    # starts calling it with no change to the DAG.
    build_card_index = BashOperator(
        task_id="build_card_index",
        bash_command=(
            f"cd {PIPELINE_DIR} && "
            f"if {PYTHON} -c 'import importlib.util, sys; "
            'sys.exit(0 if importlib.util.find_spec("pipeline.card_index") else 1)\'; then '
            + stage("card_index")
            + "; else echo 'pipeline.card_index is not implemented yet, skipping'; fi"
        ),
        env=TASK_ENV,
        append_env=True,
    )

    # The only task allowed to fail a green run: it reads `mart_pipeline_health`
    # and refuses when a stage's last run failed or its quarantine rate is over
    # the threshold.
    quality_gate = BashOperator(
        task_id="quality_gate",
        bash_command=stage("quality_gate"),
        env=TASK_ENV,
        append_env=True,
    )

    (
        backfill
        >> spark_silver
        >> dbt_run
        >> dbt_test
        >> build_features
        >> train
        >> promote
        >> drift
        >> quality_gate
    )
    dbt_test >> build_card_index
