"""The nightly workflow: the contract between the file and the operator.

Nothing here runs the workflow, and nothing here talks to AWS. What it checks
is the half of the file that is a promise to somebody outside it: the set of
environment variables an operator has to set, the permissions the OpenID
Connect exchange needs, and the triggers that decide when it runs. Each of the
three is a line that can be edited without breaking anything visible until the
night it matters.

The variable check is the one worth having. The gate step at the top of the job
exists so an unconfigured environment skips with a notice instead of failing
half way through, and it only works when it names every variable the rest of
the file reads. A variable added below and forgotten in the gate is a run that
starts and then fails on an empty string, which is exactly the failure the gate
was written to prevent.
"""

import re
from typing import Any

import pytest
import yaml

from pipeline.config import REPO_ROOT

WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "nightly.yml"

DAILY_CRON = "17 10 * * *"
WEEKLY_CRON = "17 9 * * 0"


@pytest.fixture(scope="module")
def source() -> str:
    return WORKFLOW_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def workflow(source: str) -> dict[str, Any]:
    document = yaml.safe_load(source)
    assert isinstance(document, dict)
    return document


def job(workflow: dict[str, Any]) -> dict[str, Any]:
    """The one job. Named here so a rename is one failure, not twelve."""
    jobs = workflow["jobs"]
    assert list(jobs) == ["run"]
    result = jobs["run"]
    assert isinstance(result, dict)
    return result


def steps(workflow: dict[str, Any]) -> list[dict[str, Any]]:
    return list(job(workflow)["steps"])


def step_named(workflow: dict[str, Any], name: str) -> dict[str, Any]:
    (found,) = [step for step in steps(workflow) if step.get("name") == name]
    return found


def triggers(workflow: dict[str, Any]) -> dict[str, Any]:
    """The `on:` block, which YAML 1.1 reads as the boolean `True`."""
    block = workflow["on"] if "on" in workflow else workflow[True]  # type: ignore[index]
    assert isinstance(block, dict)
    return block


def test_the_gate_step_names_exactly_the_variables_the_file_reads(
    workflow: dict[str, Any], source: str
) -> None:
    """Three lists of names, one set: the file's, the gate's environment, the gate's loop."""
    used = set(re.findall(r"vars\.([A-Z0-9_]+)", source))
    gate = step_named(workflow, "Check the pipeline variables are set")
    declared = set(gate["env"])
    loop = re.search(r"for name in (.*?); do", gate["run"], re.DOTALL)
    assert loop is not None, "the gate step no longer loops over the variable names"
    checked = set(loop.group(1).replace("\\", " ").split())

    assert used == declared == checked
    # And the set itself, so adding one is a deliberate edit here as well.
    assert used == {
        "AWS_REGION",
        "PIPELINE_CI_ROLE_ARN",
        "PIPELINE_READER_ROLE_ARN",
        "PIPELINE_DATA_DIR",
        "PRA_BUCKET",
        "PRA_PREFIX",
        "PRA_INSIGHTS_TABLE",
        "HANDLE_HMAC_KEY_SECRET_ID",
    }


def test_every_step_after_the_gate_waits_for_it(workflow: dict[str, Any]) -> None:
    """A skipped run must skip all of it, including the steps that run `always()`."""
    after = steps(workflow)[2:]
    assert after, "the job has no steps after the gate"
    for step in after:
        assert "steps.configured.outputs.ready == 'true'" in str(step.get("if", ""))


def test_the_job_may_ask_for_an_oidc_token_and_nothing_more(workflow: dict[str, Any]) -> None:
    assert workflow["permissions"] == {"contents": "read", "id-token": "write"}


def test_the_reader_role_is_chained_and_capped_at_the_hour(workflow: dict[str, Any]) -> None:
    """A chained session cannot outlive an hour, and the job has to finish inside it."""
    chained = step_named(workflow, "Assume the pipeline reader role")
    assert chained["uses"] == "aws-actions/configure-aws-credentials@v4"
    assert chained["with"]["role-chaining"] is True
    assert chained["with"]["role-duration-seconds"] == 3600
    assert job(workflow)["timeout-minutes"] < 60


def test_both_schedules_are_there(workflow: dict[str, Any]) -> None:
    crons = [entry["cron"] for entry in triggers(workflow)["schedule"]]
    assert crons == [DAILY_CRON, WEEKLY_CRON]


def test_the_weekly_schedule_is_the_one_that_backfills(workflow: dict[str, Any]) -> None:
    """The schedule string is the only thing a scheduled event carries to tell them apart."""
    plan = step_named(workflow, "Decide what this run is")
    assert WEEKLY_CRON in plan["run"]
    assert DAILY_CRON not in plan["run"]
    assert plan["env"]["SCHEDULE"] == "${{ github.event.schedule }}"


def test_a_hand_run_chooses_its_environment_and_whether_to_backfill(
    workflow: dict[str, Any],
) -> None:
    inputs = triggers(workflow)["workflow_dispatch"]["inputs"]
    assert set(inputs) == {"stage", "full_backfill"}
    assert inputs["stage"]["type"] == "choice"
    assert sorted(inputs["stage"]["options"]) == ["dev", "prod"]
    assert inputs["stage"]["default"] == "prod"
    assert inputs["full_backfill"]["type"] == "boolean"
    assert inputs["full_backfill"]["default"] is False
    assert job(workflow)["environment"] == "${{ inputs.stage || 'prod' }}"


def test_the_key_is_read_only_by_the_run_that_writes_bronze(workflow: dict[str, Any]) -> None:
    """Nothing downstream of bronze anonymizes anything, so the nightly never holds the key."""
    key_step = step_named(workflow, "Read the anonymization key")
    assert "steps.plan.outputs.backfill == 'true'" in key_step["if"]
    assert "::add-mask::" in key_step["run"]
    assert "HANDLE_HMAC_KEY=" in key_step["run"]


def test_the_summary_and_the_report_survive_a_failed_run(workflow: dict[str, Any]) -> None:
    """The night that went wrong is the night whose output somebody needs."""
    run = step_named(workflow, "Run the pipeline")
    assert "--summary-path artifacts/run_summary.txt" in run["run"]
    upload = step_named(workflow, "Upload the run's output")
    assert upload["uses"].startswith("actions/upload-artifact@")
    assert upload["with"]["retention-days"] == 30
    assert upload["with"]["path"] == "artifacts/"
    for name in ("Collect the drift report", "Upload the run's output", "Write the job summary"):
        assert "always()" in step_named(workflow, name)["if"]


def test_the_job_summary_reports_the_run_id_the_gate_and_the_publish(
    workflow: dict[str, Any],
) -> None:
    body = step_named(workflow, "Write the job summary")["run"]
    assert "$GITHUB_STEP_SUMMARY" in body
    for field in ("run id", "quality gate", "publish", "environment"):
        assert field in body


def test_the_file_names_no_identifier(source: str) -> None:
    """Belt to `scripts/check_history.sh`'s braces, said once where it is easy to read."""
    assert not re.search(r"arn:aws:", source)
    assert not re.search(r"s3://[a-z]", source)
    assert not re.search(r"\b[0-9]{12}\b", source)


def test_the_workflow_uses_the_same_toolchain_as_the_test_job(workflow: dict[str, Any]) -> None:
    """One Java and one uv across the repository: a version that drifts is a run that differs."""
    ci = yaml.safe_load((REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text())
    check = ci["jobs"]["check"]["steps"]
    expected = {
        step["name"]: (step["uses"], step["with"].get("java-version"))
        for step in check
        if step.get("name") in {"Install Java", "Install uv"}
    }
    for name, (uses, java) in expected.items():
        step = step_named(workflow, name)
        assert step["uses"] == uses
        if java is not None:
            assert step["with"]["java-version"] == java
