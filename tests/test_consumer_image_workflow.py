"""The consumer image workflow: the contract between the file and the operator.

Nothing here runs the workflow, and nothing here talks to AWS. What it checks
is the half of the file that is a promise to somebody outside it: that a push
to main rebuilds and redeploys *both* environments rather than whichever one
happened to own the repository variables, that each build declares the
environment whose OpenID Connect subject the CI role is trusted against, and
that the gate at the top names every variable the rest of the file reads.

The last of those is the one worth having, for the same reason it is worth
having in `tests/test_nightly_workflow.py`: a variable added below and
forgotten in the gate is a run that starts and then fails on an empty string,
which is exactly the failure the gate was written to prevent.
"""

import re
from typing import Any

import pytest
import yaml

from pipeline.config import REPO_ROOT

WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "consumer-image.yml"

ENVIRONMENTS = ["dev", "prod"]


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
    assert list(jobs) == ["plan", "image"]
    result = jobs[name]
    assert isinstance(result, dict)
    return result


def steps(workflow: dict[str, Any]) -> list[dict[str, Any]]:
    return list(job(workflow, "image")["steps"])


def step_named(workflow: dict[str, Any], name: str) -> dict[str, Any]:
    (found,) = [step for step in steps(workflow) if step.get("name") == name]
    return found


def triggers(workflow: dict[str, Any]) -> dict[str, Any]:
    """The `on:` block, which YAML 1.1 reads as the boolean `True`."""
    block = workflow["on"] if "on" in workflow else workflow[True]  # type: ignore[index]
    assert isinstance(block, dict)
    return block


def test_the_build_fans_out_over_both_environments(workflow: dict[str, Any]) -> None:
    """A push to main has to reach prod as well as dev, and one failing is not the other's."""
    image = job(workflow, "image")
    assert image["strategy"]["fail-fast"] is False
    assert image["strategy"]["matrix"] == {
        "environment": "${{ fromJSON(needs.plan.outputs.environments) }}"
    }
    assert image["needs"] == "plan"
    assert image["environment"] == "${{ matrix.environment }}"


def test_the_plan_job_defaults_to_every_environment(workflow: dict[str, Any]) -> None:
    """The matrix is JSON from a shell script, so the lists it can emit are the contract."""
    plan = job(workflow, "plan")
    assert plan["outputs"] == {"environments": "${{ steps.pick.outputs.environments }}"}
    (pick,) = plan["steps"]
    assert pick["env"] == {"ONLY": "${{ inputs.environment }}"}
    assert '["dev","prod"]' in pick["run"]
    assert "$GITHUB_OUTPUT" in pick["run"]


def test_a_hand_run_may_narrow_the_matrix_to_one_environment(workflow: dict[str, Any]) -> None:
    """A choice input always carries a value, so "both" is an option rather than an absence."""
    inputs = triggers(workflow)["workflow_dispatch"]["inputs"]
    assert set(inputs) == {"environment"}
    environment = inputs["environment"]
    assert environment["type"] == "choice"
    assert environment["required"] is False
    assert environment["default"] == "both"
    assert environment["options"] == ["both", *ENVIRONMENTS]
    # Anything that is not one environment's name means all of them.
    plan_run = job(workflow, "plan")["steps"][0]["run"]
    assert "|".join(ENVIRONMENTS) + ")" in plan_run


def test_the_gate_step_names_exactly_the_variables_the_file_reads(
    workflow: dict[str, Any], source: str
) -> None:
    """Three lists, one set: what the file reads, what the gate binds, what its loop tests."""
    used = set(re.findall(r"vars\.([A-Z0-9_]+)", source))
    gate = step_named(workflow, "Check the deployment variables are set")
    bound = {name: re.findall(r"vars\.([A-Z0-9_]+)", value) for name, value in gate["env"].items()}
    loop = re.search(r"for name in (.*?); do", gate["run"], re.DOTALL)
    assert loop is not None, "the gate step no longer loops over the variable names"
    checked = set(loop.group(1).replace("\\", " ").split())

    # Every variable the file reads is bound to exactly one local name, and
    # every local name the gate binds is one the loop tests for emptiness.
    assert {variable for names in bound.values() for variable in names} == used
    assert all(len(names) == 1 for names in bound.values())
    assert set(bound) == checked
    # And the set itself, so adding one is a deliberate edit here as well.
    assert used == {
        "AWS_REGION",
        "PIPELINE_CI_ROLE_ARN",
        "CONSUMER_ECR_REPOSITORY",
        "CONSUMER_FUNCTION_NAME",
    }


def test_every_step_after_the_gate_waits_for_it(workflow: dict[str, Any]) -> None:
    """A skipped environment must skip all of it, not just the AWS calls."""
    after = steps(workflow)[2:]
    assert after, "the job has no steps after the gate"
    for step in after:
        assert "steps.configured.outputs.ready == 'true'" in str(step.get("if", ""))


def test_two_pushes_queue_per_environment_and_never_cancel(workflow: dict[str, Any]) -> None:
    """Both tag `:latest`, so dev and prod may overlap and two dev builds may not."""
    concurrency = job(workflow, "image")["concurrency"]
    assert "matrix.environment" in concurrency["group"]
    assert concurrency["cancel-in-progress"] is False
    assert "concurrency" not in workflow, "a workflow-level group would serialize the environments"


def test_the_image_stays_a_single_platform_manifest(workflow: dict[str, Any]) -> None:
    """Lambda refuses an OCI image index, which is what an attestation turns the build into."""
    build = step_named(workflow, "Build and push the image")["run"]
    assert "--provenance=false" in build
    assert "--sbom=false" in build
    assert "$REPOSITORY:${{ github.sha }}" in build
    assert "$REPOSITORY:latest" in build


def test_the_deploy_waits_for_lambda_to_finish_pulling(workflow: dict[str, Any]) -> None:
    deploy = step_named(workflow, "Point the function at the new image")["run"]
    assert "aws lambda update-function-code" in deploy
    assert "aws lambda wait function-updated" in deploy


def test_the_job_may_ask_for_an_oidc_token_and_nothing_more(workflow: dict[str, Any]) -> None:
    assert workflow["permissions"] == {"contents": "read", "id-token": "write"}


def test_the_file_names_no_identifier(source: str) -> None:
    """Belt to `scripts/check_history.sh`'s braces, said once where it is easy to read."""
    assert not re.search(r"arn:aws:", source)
    assert not re.search(r"s3://[a-z]", source)
    assert not re.search(r"\b[0-9]{12}\b", source)
