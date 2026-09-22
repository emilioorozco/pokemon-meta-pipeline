"""Gold stage: one command that builds and tests the dbt project over silver.

dbt is a command-line tool, so this is a thin wrapper rather than a library
call: it pins the project and profiles directories (both are `dbt/` in this
repository, so a fresh clone needs no `~/.dbt`), passes the lake location
through `PIPELINE_DATA_DIR` so the models and the Python package cannot
disagree about where the data is, and runs the steps in order. Orchestration
later calls one command per stage, and this is the gold one.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from pipeline.config import PIPELINE_DATA_DIR, REPO_ROOT

DBT_DIR = REPO_ROOT / "dbt"


def dbt_executable() -> str:
    """The `dbt` beside this interpreter, so a virtual environment run finds its own."""
    local = Path(sys.executable).parent / "dbt"
    return str(local) if local.is_file() else "dbt"


def run_gold(*, data_dir: Path, target: str = "dev", dbt_dir: Path = DBT_DIR) -> int:
    """Run `dbt deps` (when there are packages), `dbt run` and `dbt test`."""
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
    return run_gold(data_dir=args.data_dir, target=args.target)


if __name__ == "__main__":
    raise SystemExit(main())
