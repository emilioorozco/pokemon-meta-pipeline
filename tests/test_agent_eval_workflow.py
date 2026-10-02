"""The agent eval workflow: the contract between the file and the operator.

Nothing here runs the workflow and nothing here talks to AWS or to a provider.
What it checks is the half of the file that is a promise to somebody outside
it: that the two jobs measure the two different things they claim to, that the
deployed-agent job skips with a notice rather than failing when the repository
has no deployment to point it at, and that it asks for the one permission the
OpenID Connect exchange needs and no more.

The variable gate is the check worth having, and it is the same one
`tests/test_nightly_workflow.py` makes against the nightly file. A variable
added to a step and forgotten in the gate is a job that starts, assumes a
role, and fails on an empty string in the middle, which is exactly the failure
the gate was written to prevent.
"""

import re
from typing import Any

import pytest
import yaml

from pipeline.config import REPO_ROOT

WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "agent-eval.yml"

LOCAL_JOB = "golden"
REMOTE_JOB = "prod"


@pytest.fixture(scope="module")
def source() -> str:
    return WORKFLOW_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def workflow(source: str) -> dict[str, Any]:
    document = yaml.safe_load(source)
    assert isinstance(document, dict)
    return document


def job(workflow: dict[str, Any], name: str) -> dict[str, Any]:
    """One of the two jobs. Named here so a rename is one failure, not eight."""
    jobs = workflow["jobs"]
    assert list(jobs) == [LOCAL_JOB, REMOTE_JOB]
    result = jobs[name]
    assert isinstance(result, dict)
    return result


def steps(workflow: dict[str, Any], name: str) -> list[dict[str, Any]]:
    return list(job(workflow, name)["steps"])


def step_named(workflow: dict[str, Any], job_name: str, name: str) -> dict[str, Any]:
    (found,) = [step for step in steps(workflow, job_name) if step.get("name") == name]
    return found


def test_the_two_jobs_measure_the_commit_and_the_deployment(workflow: dict[str, Any]) -> None:
    """One builds an agent here, the other asks the one members talk to."""
    local = job(workflow, LOCAL_JOB)
    remote = job(workflow, REMOTE_JOB)
    assert remote["needs"] == LOCAL_JOB
    # `always()`, because a failed local run is when the deployed one matters
    # most: the two disagreeing is the news this job exists to carry.
    assert remote["if"] == "always()"
    assert remote["environment"] == "prod"
    # The local job builds the fixture marts; the remote one must not, because
    # the warehouse that answers is the function's own.
    assert any("fixture marts" in str(step.get("name", "")) for step in local["steps"])
    assert not any("fixture marts" in str(step.get("name", "")) for step in remote["steps"])


def test_only_the_deployed_job_may_ask_for_an_oidc_token(workflow: dict[str, Any]) -> None:
    assert workflow["permissions"] == {"contents": "read"}
    assert job(workflow, REMOTE_JOB)["permissions"] == {
        "contents": "read",
        "id-token": "write",
    }
    assert "permissions" not in job(workflow, LOCAL_JOB)


def test_the_gate_step_names_exactly_the_variables_the_file_reads(
    workflow: dict[str, Any], source: str
) -> None:
    """Three lists of names, one set: the file's, the gate's environment, the gate's loop."""
    used = set(re.findall(r"vars\.([A-Z0-9_]+)", source))
    gate = step_named(workflow, REMOTE_JOB, "Check the agent variables are set")
    declared = set(gate["env"])
    loop = re.search(r"for name in (.*?); do", gate["run"], re.DOTALL)
    assert loop is not None, "the gate step no longer loops over the variable names"
    checked = set(loop.group(1).replace("\\", " ").split())

    assert used == declared == checked
    # And the set itself, so adding one is a deliberate edit here as well.
    assert used == {"AWS_REGION", "PIPELINE_CI_ROLE_ARN", "PIPELINE_AGENT_URL"}


def test_an_unset_variable_is_a_notice_and_a_green_job(workflow: dict[str, Any]) -> None:
    """A clone with no deployment behind it must not see a red build."""
    gate = step_named(workflow, REMOTE_JOB, "Check the agent variables are set")
    assert "::notice" in gate["run"]
    assert "ready=false" in gate["run"]
    after = steps(workflow, REMOTE_JOB)[2:]
    assert after, "the job has no steps after the gate"
    for step in after:
        assert "steps.configured.outputs.ready == 'true'" in str(step.get("if", ""))


def test_the_deployed_job_scores_the_same_file_over_the_url(workflow: dict[str, Any]) -> None:
    run = step_named(workflow, REMOTE_JOB, "Score the deployed agent")["run"]
    assert '--remote "$PIPELINE_AGENT_URL"' in run
    assert "--golden evals/golden.yaml" in run
    # The pass bar is the command's exit code, which is the local job's bar
    # too, so `pipefail` has to be on or `tee` would swallow a red run.
    assert "set -eu -o pipefail" in run
    assert "tee artifacts/remote_eval_report.json" in run


def test_the_report_survives_a_failed_run(workflow: dict[str, Any]) -> None:
    keep = step_named(workflow, REMOTE_JOB, "Keep the report")
    assert keep["if"].startswith("always()")
    assert keep["uses"].startswith("actions/upload-artifact@")
    assert keep["with"]["path"] == "artifacts/"


def test_the_header_says_what_a_run_costs(source: str) -> None:
    """A job that spends money on a schedule says so where it is turned on."""
    header = source.split("name: Agent eval")[0]
    assert "cent" in header
    assert "Haiku" in header


def test_the_file_names_no_identifier(source: str) -> None:
    """Every account-shaped thing is a GitHub Environment variable, never a literal."""
    assert "lambda-url" not in source
    assert "amazonaws.com" not in source
    assert not re.search(r"\barn:aws:", source)
    assert not re.search(r"\b\d{12}\b", source)
