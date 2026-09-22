"""The DAG file, read as source rather than loaded as a DAG.

Airflow is not installed in this project's environment and is deliberately not
going to be. `pyproject.toml` carries it as an optional `orchestration` extra,
which resolves to Airflow 3 under this repository's `requires-python`, while the
image that actually runs the DAG is `apache/airflow:2.10.5-python3.12`. A
`DagBag` test here would therefore load the file under a different Airflow than
the one that schedules it, on an interpreter Airflow does not support, and would
be slower and less honest than the thing it replaced. The real load test is the
scheduler parsing the file in the container, which is what
`docker compose up airflow` does on every start.

So this parses the file with `ast` and asserts the parts a typo would break
silently: the task identifiers, the edges between them, and the handful of
scheduling arguments whose wrong value is a DAG that runs at the wrong time, or
twice, or over every day since the start date. Those are facts about the source
text, and the source text is what the scheduler reads too.
"""

import ast
from pathlib import Path
from typing import Final

import pytest

DAG_FILE: Final = (
    Path(__file__).parent.parent / "orchestration" / "airflow" / "dags" / "play_rough_pipeline.py"
)

EXPECTED_TASKS: Final = {
    "backfill",
    "spark_silver",
    "dbt_run",
    "dbt_test",
    "build_features",
    "train",
    "promote",
    "drift",
    "build_card_index",
    "quality_gate",
    "publish",
}

# The chain the ticket specifies, plus the one branch.
EXPECTED_EDGES: Final = {
    ("backfill", "spark_silver"),
    ("spark_silver", "dbt_run"),
    ("dbt_run", "dbt_test"),
    ("dbt_test", "build_features"),
    ("build_features", "train"),
    ("train", "promote"),
    ("promote", "drift"),
    ("drift", "quality_gate"),
    ("quality_gate", "publish"),
    ("dbt_test", "build_card_index"),
}


@pytest.fixture(scope="module")
def tree() -> ast.Module:
    return ast.parse(DAG_FILE.read_text(encoding="utf-8"), filename=str(DAG_FILE))


def operators(tree: ast.Module) -> dict[str, ast.Call]:
    """Every `name = BashOperator(...)` in the file, keyed by the variable name.

    Keyed by variable rather than by `task_id` on purpose: the edges below are
    written in terms of the variables, so a task whose variable and identifier
    disagree would build a graph that does not match the one this asserts, and
    the mismatch is what the next test catches.
    """
    found = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            callee = node.value.func
            if isinstance(callee, ast.Name) and callee.id == "BashOperator":
                (target,) = node.targets
                assert isinstance(target, ast.Name)
                found[target.id] = node.value
    return found


def keyword(call: ast.Call, name: str) -> ast.expr | None:
    """One keyword argument's expression, or None."""
    return next((word.value for word in call.keywords if word.arg == name), None)


def literal(call: ast.Call, name: str) -> object:
    """One keyword argument as a Python value, for the ones that are constants."""
    node = keyword(call, name)
    assert node is not None, name
    return ast.literal_eval(node)


def assignment(tree: ast.Module, name: str) -> ast.expr:
    """The right-hand side of the one module-level `name = ...` in the file."""
    (found,) = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == name
    ]
    return found


def constant(tree: ast.Module, name: str) -> object:
    """A module-level constant's value."""
    return ast.literal_eval(assignment(tree, name))


def chain(node: ast.expr) -> list[str]:
    """The names of a `a >> b >> c` expression, left to right."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.RShift):
        return chain(node.left) + chain(node.right)
    assert isinstance(node, ast.Name), ast.dump(node)
    return [node.id]


def edges(tree: ast.Module) -> set[tuple[str, str]]:
    """Every dependency the file declares with `>>`."""
    found: set[tuple[str, str]] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.BinOp):
            names = chain(node.value)
            found.update(zip(names, names[1:], strict=False))
    return found


def dag_call(tree: ast.Module) -> ast.Call:
    """The `with DAG(...)` call."""
    for node in ast.walk(tree):
        if isinstance(node, ast.With):
            (item,) = node.items
            assert isinstance(item.context_expr, ast.Call)
            return item.context_expr
    raise AssertionError("no `with DAG(...)` block in the DAG file")


def test_the_dag_file_is_valid_python() -> None:
    assert DAG_FILE.is_file(), DAG_FILE
    ast.parse(DAG_FILE.read_text(encoding="utf-8"))


def test_every_task_is_there_and_named_after_its_variable(tree: ast.Module) -> None:
    found = operators(tree)
    assert set(found) == EXPECTED_TASKS
    for name, call in found.items():
        assert literal(call, "task_id") == name


def test_the_dependencies_are_the_chain_with_one_branch(tree: ast.Module) -> None:
    assert edges(tree) == EXPECTED_EDGES


def test_every_task_calls_a_stage_command(tree: ast.Module) -> None:
    """A task that does not shell out to `python -m pipeline.*` is not a thin task."""
    source = DAG_FILE.read_text(encoding="utf-8")
    assert "-m pipeline." in source
    for name, call in operators(tree).items():
        command = keyword(call, "bash_command")
        assert command is not None, name
        rendered = ast.dump(command)
        assert "stage" in rendered or "pipeline." in rendered, name


def test_every_task_is_given_the_run_identifier(tree: ast.Module) -> None:
    """One DAG run, one `PRA_RUN_ID`, which is what makes the metrics joinable."""
    for name, call in operators(tree).items():
        env = keyword(call, "env")
        assert isinstance(env, ast.Name) and env.id == "TASK_ENV", name

    env_dict = assignment(tree, "TASK_ENV")
    assert isinstance(env_dict, ast.Dict)
    keys = [key.value for key in env_dict.keys if isinstance(key, ast.Constant)]
    assert "PRA_RUN_ID" in keys
    assert {"PIPELINE_DATA_DIR", "MLFLOW_TRACKING_URI", "HANDLE_HMAC_KEY"} <= set(keys)


def test_the_schedule_is_daily_and_does_not_catch_up(tree: ast.Module) -> None:
    call = dag_call(tree)
    dag_id = keyword(call, "dag_id")
    assert isinstance(dag_id, ast.Name) and dag_id.id == "DAG_ID"
    assert constant(tree, "DAG_ID") == "play_rough_pipeline"
    assert literal(call, "schedule") == "@daily"
    assert literal(call, "catchup") is False
    assert literal(call, "max_active_runs") == 1


def test_the_two_parameters_are_declared_with_their_defaults(tree: ast.Module) -> None:
    params = keyword(dag_call(tree), "params")
    assert isinstance(params, ast.Dict)
    names = [key.value for key in params.keys if isinstance(key, ast.Constant)]
    assert names == ["source_dir", "ingest_mode"]
    # An empty `source_dir` means S3; the fixture run passes tests/fixtures.
    source_dir, ingest_mode = params.values
    assert isinstance(source_dir, ast.Call) and ast.literal_eval(source_dir.args[0]) == ""
    assert isinstance(ingest_mode, ast.Call)
    assert ast.literal_eval(ingest_mode.args[0]) == "backfill"


def test_the_ingest_task_is_the_one_that_retries(tree: ast.Module) -> None:
    """A listing or a GET can fail transiently; a dbt test cannot."""
    found = operators(tree)
    assert literal(found["backfill"], "retries") == 1
    for name, call in found.items():
        if name != "backfill":
            assert keyword(call, "retries") is None, name


def test_the_owner_is_the_pipeline(tree: ast.Module) -> None:
    defaults = assignment(tree, "DEFAULT_ARGS")
    assert isinstance(defaults, ast.Dict)
    args = {
        key.value: value
        for key, value in zip(defaults.keys, defaults.values, strict=True)
        if isinstance(key, ast.Constant)
    }
    assert ast.literal_eval(args["owner"]) == "pipeline"
