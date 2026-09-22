"""Gold stage: one command that builds and tests the dbt project over silver.

dbt is a command-line tool, so this is a thin wrapper rather than a library
call: it pins the project and profiles directories (both are `dbt/` in this
repository, so a fresh clone needs no `~/.dbt`), passes the lake location
through `PIPELINE_DATA_DIR` so the models and the Python package cannot
disagree about where the data is, and runs the steps in order. Orchestration
later calls one command per stage, and this is the gold one.

The counts that reach `run_metrics` are read back out of `target/run_results.json`
rather than scraped from dbt's output: dbt writes that file after every
invocation, and parsing the artifact it already produces beats matching on a
line of console text that changes between minor versions. It is also why dbt's
output is left streaming to the terminal instead of being captured.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from pipeline.config import PIPELINE_DATA_DIR, REPO_ROOT
from pipeline.observability import STATUS_FAILED, configure_logging, emit_summary, stage_run

logger = logging.getLogger(__name__)

DBT_DIR = REPO_ROOT / "dbt"
STAGE = "gold"
RUN_RESULTS = Path("target") / "run_results.json"
# What dbt calls a node that did what it was asked to: `success` for a model,
# `pass` for a test.
OK_STATUSES = frozenset({"success", "pass"})


@dataclass
class GoldSummary:
    """What the build did, per dbt step, in the order the steps ran."""

    steps: dict[str, tuple[int, int]] = field(default_factory=dict)

    @property
    def models_run(self) -> int:
        """Nodes the `run` step attempted."""
        return self.steps.get("run", (0, 0))[0]

    @property
    def models_passed(self) -> int:
        """Nodes the `run` step built without an error."""
        return self.steps.get("run", (0, 0))[1]

    @property
    def tests_run(self) -> int:
        """Data tests the `test` step attempted."""
        return self.steps.get("test", (0, 0))[0]

    @property
    def tests_passed(self) -> int:
        """Data tests that passed."""
        return self.steps.get("test", (0, 0))[1]

    def __str__(self) -> str:
        lines = [f"models: {self.models_passed}/{self.models_run} built"]
        if "test" in self.steps:
            lines.append(f"tests:  {self.tests_passed}/{self.tests_run} passed")
        return "\n".join(lines)


def dbt_executable() -> str:
    """The `dbt` beside this interpreter, so a virtual environment run finds its own."""
    local = Path(sys.executable).parent / "dbt"
    return str(local) if local.is_file() else "dbt"


def read_run_results(dbt_dir: Path) -> tuple[int, int]:
    """Nodes attempted and nodes that passed, from the artifact the last step wrote.

    `(0, 0)` when there is no artifact or it cannot be read: a missing count is
    worth a zero in a metrics row, never a failed build.
    """
    path = dbt_dir / RUN_RESULTS
    try:
        results = json.loads(path.read_text(encoding="utf-8"))["results"]
    except (OSError, ValueError, KeyError):
        return (0, 0)
    return (len(results), sum(1 for node in results if node.get("status") in OK_STATUSES))


def run_gold(
    *,
    data_dir: Path,
    target: str = "dev",
    dbt_dir: Path = DBT_DIR,
    summary: GoldSummary | None = None,
) -> int:
    """Run `dbt deps` (when there are packages), `dbt run` and `dbt test`.

    `summary` is filled in as the steps run when one is passed. It is an
    argument rather than the return value because the exit code is what every
    caller and every test already reads, and a build that failed halfway still
    has counts worth recording.
    """
    (data_dir / "warehouse").mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PIPELINE_DATA_DIR": str(data_dir)}
    steps = ["run", "test"]
    if (dbt_dir / "packages.yml").is_file():
        steps.insert(0, "deps")
    for step in steps:
        argv = [dbt_executable(), step, "--project-dir", str(dbt_dir)]
        if step != "deps":
            argv += ["--profiles-dir", str(dbt_dir), "--target", target]
        code = subprocess.run(argv, env=env, check=False).returncode
        if summary is not None and step != "deps":
            summary.steps[step] = read_run_results(dbt_dir)
        if code:
            return code
    return 0


def main(argv: list[str] | None = None) -> int:
    """Build the gold models from the command line."""
    parser = argparse.ArgumentParser(
        prog="python -m pipeline.gold", description="Build and test the dbt gold models."
    )
    parser.add_argument("--target", default="dev", help="dbt profile target (default: dev)")
    parser.add_argument(
        "--data-dir", type=Path, default=PIPELINE_DATA_DIR, metavar="PATH", help="lake root"
    )
    args = parser.parse_args(argv)
    configure_logging(STAGE)

    summary = GoldSummary()
    # Pinned to the directory this run was pointed at, not to the ambient
    # `PIPELINE_DATA_DIR`: `--data-dir` moves the lake the models read, so a run
    # whose metrics landed somewhere else would be a row about a warehouse it
    # does not sit beside.
    with stage_run(STAGE, directory=args.data_dir / "lake" / "run_metrics") as metrics:
        code = run_gold(data_dir=args.data_dir, target=args.target, summary=summary)
        metrics.rows_in = summary.models_run
        metrics.rows_out = summary.models_passed
        metrics.rows_quarantined = 0
        metrics.extra = {
            "tests_run": summary.tests_run,
            "tests_passed": summary.tests_passed,
            "target": args.target,
            "exit_code": code,
        }
        if code:
            metrics.status = STATUS_FAILED
            metrics.error = f"dbt exited {code}"

    emit_summary(
        logger,
        "gold summary",
        {
            "models_run": summary.models_run,
            "models_passed": summary.models_passed,
            "tests_run": summary.tests_run,
            "tests_passed": summary.tests_passed,
            "exit_code": code,
        },
        text=str(summary),
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
