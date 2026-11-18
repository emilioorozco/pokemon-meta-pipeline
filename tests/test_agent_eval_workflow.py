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
MEMBER_STEP = "Read a member's token from the warehouse"
MEMBER_VARIABLES = {"PIPELINE_READER_ROLE_ARN", "PIPELINE_DATA_DIR"}


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
    """Three lists of names, one set: the gate's environment, the gate's loop, the rest.

    The two the member-token step reads are the exception, and on purpose:
    without them the chips are still asked, with the placeholder token, so
    they are checked by that step rather than by the gate that skips the job.
    """
    used = set(re.findall(r"vars\.([A-Z0-9_]+)", source))
    gate = step_named(workflow, REMOTE_JOB, "Check the agent variables are set")
    declared = set(gate["env"])
    loop = re.search(r"for name in (.*?); do", gate["run"], re.DOTALL)
    assert loop is not None, "the gate step no longer loops over the variable names"
    checked = set(loop.group(1).replace("\\", " ").split())
    member = step_named(workflow, REMOTE_JOB, MEMBER_STEP)
    optional = {name for name in member["env"] if name in MEMBER_VARIABLES}

    assert declared == checked
    assert used == declared | optional
    # And the sets themselves, so adding one is a deliberate edit here as well.
    assert declared == {"AWS_REGION", "PIPELINE_CI_ROLE_ARN", "PIPELINE_AGENT_URL"}
    assert optional == MEMBER_VARIABLES
    for name in MEMBER_VARIABLES:
        assert f'"${{{name}:-}}"' in member["run"]


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


def test_the_offered_set_is_asked_of_the_same_deployment(workflow: dict[str, Any]) -> None:
    """The second step of the deployed job, and the one that may not fail it yet."""
    step = step_named(workflow, REMOTE_JOB, "Ask the questions the application offers")
    assert step["continue-on-error"] is True
    # Asked on a red week as well as a green one.
    assert step["if"].startswith("always()")
    run = step["run"]
    assert "--offered" in run
    assert '--remote "$PIPELINE_AGENT_URL"' in run
    assert "tee artifacts/offered_eval_report.json" in run
    # A real member's token, from a secret and never from a variable or the
    # file, and optional: an unset secret falls back to the runner's own.
    assert step["env"]["OFFERED_PLAYER_TOKEN"] == "${{ secrets.OFFERED_PLAYER_TOKEN }}"
    assert '--player-token "${OFFERED_PLAYER_TOKEN:-${MEMBER_PLAYER_TOKEN:-}}"' in run
    # Nobody has measured this set, so the first runs are a baseline and the
    # news is a notice. A step that went red on day one is a step somebody
    # turns off before the baseline is read.
    report = step_named(workflow, REMOTE_JOB, "Say how the offered questions went")
    assert report["if"].startswith("always()")
    assert "::notice title=Offered questions::" in report["run"]


def test_the_member_token_is_read_with_the_reader_role_and_masked(
    workflow: dict[str, Any],
) -> None:
    """The chips' member, from the warehouse, without moving the CI session."""
    names = [step.get("name") for step in steps(workflow, REMOTE_JOB)]
    assert names.index(MEMBER_STEP) < names.index("Ask the questions the application offers")
    step = step_named(workflow, REMOTE_JOB, MEMBER_STEP)
    assert step["if"].startswith("always()")
    # A token that could not be read is the placeholder run, not a red one.
    assert step["continue-on-error"] is True
    run = step["run"]
    # The secret, when set, still decides.
    assert step["env"]["OFFERED_PLAYER_TOKEN"] == "${{ secrets.OFFERED_PLAYER_TOKEN }}"
    assert run.index("OFFERED_PLAYER_TOKEN") < run.index("sts assume-role")
    # The reader role in this shell only: no second credentials action, which
    # would replace the CI session the two scoring steps sign the URL with,
    # and no session tags, which the reader role's trust policy does not grant.
    credential_steps = [
        step
        for step in steps(workflow, REMOTE_JOB)
        if str(step.get("uses", "")).startswith("aws-actions/configure-aws-credentials")
    ]
    assert len(credential_steps) == 1
    assert "aws sts assume-role" in run
    assert "--tags" not in run
    # Masked before it is exported, and the credentials masked too.
    assert run.index('echo "::add-mask::$token"') < run.index("MEMBER_PLAYER_TOKEN=")
    assert "::add-mask::$AWS_SECRET_ACCESS_KEY" in run
    assert "::add-mask::$AWS_SESSION_TOKEN" in run
    assert "--member-token" in run
    assert '--warehouse "${PIPELINE_DATA_DIR%/}/warehouse/meta.duckdb"' in run
