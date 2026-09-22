"""The golden question set, scored: does the agent still answer these correctly?

    uv run python -m pipeline.eval --fake evals/transcript.yaml
    op run --env-file=.env.op -- uv run python -m pipeline.eval

Ten questions in `evals/golden.yaml`, each with the tools its answer has to
call and the facts its answer has to contain. The command runs them through the
real `Agent`, scores three checks per question, prints a table and exits
non-zero when anything failed. The score of a run is logged to MLflow, so a
prompt change is tracked the way a model change is.

**It grades facts, not prose.** Every assertion is a substring or a regular
expression naming a number, an archetype, a card or a sample size, because two
correct answers to "how does this matchup look" will not share a sentence and a
harness that demands one is a harness people delete. What a wrong answer cannot
do is contain the right number over the right denominator, so that is what is
required. The same reasoning makes `expect_tools` a set rather than a sequence
and makes an unexpected extra tool call a note in the report rather than a
failure: the order a model works in is style, and calling the tool at all is
not.

**The forbidden half is the interesting half.** This corpus invites two
specific wrong answers, and both are asserted against. `seen_rate` is the share
of games in which a card was observed, and reporting it as the share of decks
that run the card is the mistake the whole `mart_cards_seen` model is shaped to
avoid; two questions are written to invite exactly that and forbid the claim.
And `mart_player_summary` is keyed by an irreversible sixteen-character token,
so `[0-9a-f]{16}` is forbidden in every answer: an agent that hands over the
token as an identity has crossed the boundary docs/data-handling.md draws, and
a regular expression catches it whatever sentence it is wrapped in.

**Two ways to run it, and they prove different things.** With a provider key
and no `--fake`, this is a measurement of the model and the prompt, and it is
what the weekly workflow runs. With `--fake evals/transcript.yaml` it replays
recorded tool calls through the real agent graph, the real SQL gate and a real
DuckDB query against the fixture warehouse, so it measures the harness, the
tools and the marts and costs nothing. The second is what the tests run and
what anyone can run on a clean checkout; it is not evidence about the model,
and the transcript file says so at the top.

**The broken-prompt check.** The claim that the seven rules in
`pipeline.prompts` do the work is only worth something if removing them is
visible. `--prompt-override evals/broken_prompt.txt` (which is nothing more
than setting `PRA_AGENT_SYSTEM_PROMPT_FILE`) swaps in a prompt with the schema
and the rules cut out, and the score falls. With a real provider it falls
because the model stops citing sample sizes and starts guessing; with the
replay model it falls because the fake refuses to query a table the prompt
never described, which is the one respect in which it behaves like a model.
docs/evals.md has both procedures.

Exit codes are the point of a continuous-integration command: 0 when every
question passed, 1 when any question failed, 2 when the run could not be set
up at all, which is a missing warehouse, an unreadable golden file or a
provider that would not build. A failed question and a broken harness are
different news and should not share an exit code.
"""

import argparse
import hashlib
import json
import logging
import os
import re
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import yaml
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable

from pipeline.agent import (
    CARD_TOOL,
    SQL_TOOL,
    Agent,
    build_agent,
    chat_model,
    default_card_index,
    referenced_tables,
)
from pipeline.config import REPO_ROOT, WAREHOUSE_PATH, default_tracking_uri
from pipeline.observability import configure_logging, emit_summary, git_commit, stage_run
from pipeline.prompts import PROMPT_FILE_VAR, system_prompt

logger = logging.getLogger(__name__)

STAGE: Final = "agent_eval"

GOLDEN_PATH: Final = REPO_ROOT / "evals" / "golden.yaml"
TRANSCRIPT_PATH: Final = REPO_ROOT / "evals" / "transcript.yaml"
BROKEN_PROMPT_PATH: Final = REPO_ROOT / "evals" / "broken_prompt.txt"

# One experiment for every eval run, beside `win-probability` and
# `win-probability-drift`, so an agent change lands in the same place a model
# change does and the two are read with the same tool.
DEFAULT_EXPERIMENT: Final = "agent-evals"
REPORT_ARTIFACT: Final = "eval_report.json"

# The tools a question may name. Not `Agent.tool_names`, because the golden
# file has to be checkable without building an agent, and a typo in it should
# be a load error rather than a question that can never pass.
VALID_TOOLS: Final[tuple[str, ...]] = (SQL_TOOL, CARD_TOOL)

# A `require` or `forbid` entry starting with this is a regular expression;
# everything else is a plain substring. Both are matched case insensitively,
# because capitalisation is wording and this file grades facts.
REGEX_PREFIX: Final = "re:"

# What the replay model says instead of playing a recorded turn whose tool the
# system prompt never described. It contains no number and no archetype, so a
# question answered with it fails its `require` checks as well as its tools.
BLIND_ANSWER: Final = (
    "I was not told which tables I can read, so I cannot answer this from the warehouse."
)

CHECK_TOOLS: Final = "tools"
CHECK_REQUIRE: Final = "require"
CHECK_FORBID: Final = "forbid"
CHECK_ERROR: Final = "error"


class GoldenError(ValueError):
    """The golden file or the transcript is not the shape the runner needs."""


# ------------------------------------------------------------- the golden --


@dataclass(frozen=True)
class Question:
    """One golden question: what to ask, what to call, and what must be true."""

    id: str
    question: str
    expect_tools: tuple[str, ...] = ()
    require: tuple[str, ...] = ()
    forbid: tuple[str, ...] = ()
    notes: str = ""


@dataclass(frozen=True)
class Golden:
    """A loaded golden file: its version and its questions, in file order."""

    version: int
    questions: tuple[Question, ...]
    path: Path


def matches(pattern: str, text: str) -> bool:
    """Whether one `require` or `forbid` entry is present in an answer."""
    if pattern.startswith(REGEX_PREFIX):
        return re.search(pattern[len(REGEX_PREFIX) :], text, re.IGNORECASE) is not None
    return pattern.casefold() in text.casefold()


def _patterns(raw: Any, *, where: str) -> tuple[str, ...]:
    """A `require` or `forbid` list, with every regular expression compiled once.

    Compiled here and thrown away, so that a pattern with an unbalanced bracket
    in it is a load error naming the question rather than a traceback in the
    middle of question seven.
    """
    if raw is None:
        return ()
    if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
        raise GoldenError(f"{where}: expected a list of strings, got {type(raw).__name__}")
    patterns = tuple(str(item) for item in raw)
    for pattern in patterns:
        if not pattern.strip():
            raise GoldenError(f"{where}: an empty pattern matches everything")
        if pattern.startswith(REGEX_PREFIX):
            try:
                re.compile(pattern[len(REGEX_PREFIX) :])
            except re.error as failure:
                raise GoldenError(
                    f"{where}: {pattern!r} is not a regular expression: {failure}"
                ) from failure
    return patterns


def load_golden(path: Path = GOLDEN_PATH) -> Golden:
    """Read and check the golden file. Raises `GoldenError` on anything wrong.

    Every check here is one a question could otherwise fail silently for: a
    duplicate id would overwrite a metric, a tool name with a typo in it could
    never be called, and a question with no `require` at all would pass on an
    empty answer.
    """
    try:
        parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as failure:
        raise GoldenError(f"{path}: {failure}") from failure
    except yaml.YAMLError as failure:
        raise GoldenError(f"{path}: not valid YAML: {failure}") from failure
    if not isinstance(parsed, dict):
        raise GoldenError(f"{path}: expected a mapping with `version` and `questions`")
    version = parsed.get("version")
    if not isinstance(version, int):
        raise GoldenError(f"{path}: `version` has to be an integer, got {version!r}")
    raw_questions = parsed.get("questions")
    if not isinstance(raw_questions, list) or not raw_questions:
        raise GoldenError(f"{path}: `questions` has to be a non-empty list")

    questions: list[Question] = []
    seen: set[str] = set()
    for position, raw in enumerate(raw_questions, start=1):
        if not isinstance(raw, dict):
            raise GoldenError(f"{path}: question {position} is not a mapping")
        identifier = str(raw.get("id", "")).strip()
        where = f"{path}: question {position} ({identifier or 'unnamed'})"
        if not identifier:
            raise GoldenError(f"{where}: every question needs an `id`")
        if identifier in seen:
            raise GoldenError(f"{where}: duplicate id")
        seen.add(identifier)
        text = str(raw.get("question", "")).strip()
        if not text:
            raise GoldenError(f"{where}: `question` is empty")
        tools = raw.get("expect_tools") or []
        if not isinstance(tools, list):
            raise GoldenError(f"{where}: `expect_tools` has to be a list")
        for tool in tools:
            if tool not in VALID_TOOLS:
                raise GoldenError(f"{where}: {tool!r} is not a tool ({', '.join(VALID_TOOLS)})")
        require = _patterns(raw.get("require"), where=f"{where} require")
        if not require:
            raise GoldenError(f"{where}: `require` is empty, so any answer would pass")
        questions.append(
            Question(
                id=identifier,
                question=text,
                expect_tools=tuple(str(tool) for tool in tools),
                require=require,
                forbid=_patterns(raw.get("forbid"), where=f"{where} forbid"),
                notes=str(raw.get("notes", "")).strip(),
            )
        )
    return Golden(version=version, questions=tuple(questions), path=path)


# ------------------------------------------------------------- the scorer --


@dataclass(frozen=True)
class Result:
    """One question's outcome: what was called, what was missing, and why."""

    question: Question
    answer: str
    tools_called: tuple[str, ...] = ()
    missing_tools: tuple[str, ...] = ()
    unexpected_tools: tuple[str, ...] = ()
    missing_required: tuple[str, ...] = ()
    present_forbidden: tuple[str, ...] = ()
    error: str | None = None

    @property
    def failed_checks(self) -> tuple[str, ...]:
        """The names of the checks this question failed, in reporting order."""
        failed = []
        if self.error is not None:
            failed.append(CHECK_ERROR)
        if self.missing_tools:
            failed.append(CHECK_TOOLS)
        if self.missing_required:
            failed.append(CHECK_REQUIRE)
        if self.present_forbidden:
            failed.append(CHECK_FORBID)
        return tuple(failed)

    @property
    def passed(self) -> bool:
        return not self.failed_checks

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.question.id,
            "passed": self.passed,
            "failed_checks": list(self.failed_checks),
            "tools_called": list(self.tools_called),
            "missing_tools": list(self.missing_tools),
            "unexpected_tools": list(self.unexpected_tools),
            "missing_required": list(self.missing_required),
            "present_forbidden": list(self.present_forbidden),
            "error": self.error,
            "answer": self.answer,
        }


def score(question: Question, answer: str, tools_called: Sequence[str]) -> Result:
    """Score one answer against one question. Pure, and the unit the tests hit.

    A tool is expected or it is not; a tool called and not expected is recorded
    as unexpected and costs nothing, because the golden set says what an answer
    must be built from and not what it may not look at along the way.
    """
    called = tuple(tools_called)
    unique = set(called)
    return Result(
        question=question,
        answer=answer,
        tools_called=called,
        missing_tools=tuple(tool for tool in question.expect_tools if tool not in unique),
        unexpected_tools=tuple(
            sorted(unique - set(question.expect_tools)) if question.expect_tools else ()
        ),
        missing_required=tuple(
            pattern for pattern in question.require if not matches(pattern, answer)
        ),
        present_forbidden=tuple(pattern for pattern in question.forbid if matches(pattern, answer)),
    )


@dataclass(frozen=True)
class Report:
    """A whole run: how it was set up, and how each question went."""

    golden: Golden
    results: tuple[Result, ...]
    model: str
    prompt_sha256: str
    prompt_override: str | None = None
    fake: str | None = None
    warehouse: str = ""
    card_index: str | None = None
    commit: str | None = None

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for result in self.results if result.passed)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "golden_version": self.golden.version,
            "golden_path": str(self.golden.path),
            "model": self.model,
            "prompt_sha256": self.prompt_sha256,
            "prompt_override": self.prompt_override,
            "fake": self.fake,
            "warehouse": self.warehouse,
            "card_index": self.card_index,
            "git_commit": self.commit,
            "passed": self.passed,
            "total": self.total,
            "pass_rate": round(self.pass_rate, 4),
            "questions": [result.as_dict() for result in self.results],
        }


# --------------------------------------------------------------- the fake --


@dataclass(frozen=True)
class Turn:
    """One recorded tool call: the tool and the arguments it was given."""

    tool: str
    args: dict[str, Any]


@dataclass(frozen=True)
class Transcript:
    """Recorded runs, keyed by question id: what a competent run called and said."""

    version: int
    runs: dict[str, tuple[tuple[Turn, ...], str]]
    path: Path

    def for_question(self, identifier: str) -> tuple[tuple[Turn, ...], str]:
        try:
            return self.runs[identifier]
        except KeyError:
            raise GoldenError(
                f"{self.path}: no recorded run for {identifier!r}. Add its turns, or run "
                f"the evaluation against a provider instead of --fake."
            ) from None


def load_transcript(path: Path = TRANSCRIPT_PATH) -> Transcript:
    """Read the recorded runs. Raises `GoldenError` on anything unplayable."""
    try:
        parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as failure:
        raise GoldenError(f"{path}: {failure}") from failure
    except yaml.YAMLError as failure:
        raise GoldenError(f"{path}: not valid YAML: {failure}") from failure
    if not isinstance(parsed, dict) or not isinstance(parsed.get("runs"), dict):
        raise GoldenError(f"{path}: expected a mapping with `version` and `runs`")
    runs: dict[str, tuple[tuple[Turn, ...], str]] = {}
    for identifier, raw in parsed["runs"].items():
        where = f"{path}: run {identifier}"
        if not isinstance(raw, dict):
            raise GoldenError(f"{where}: expected a mapping with `turns` and `answer`")
        answer = str(raw.get("answer", "")).strip()
        if not answer:
            raise GoldenError(f"{where}: `answer` is empty")
        turns: list[Turn] = []
        for position, turn in enumerate(raw.get("turns") or [], start=1):
            if not isinstance(turn, dict) or turn.get("tool") not in VALID_TOOLS:
                raise GoldenError(f"{where}: turn {position} has to name a tool")
            args = turn.get("args") or {}
            if not isinstance(args, dict):
                raise GoldenError(f"{where}: turn {position} `args` has to be a mapping")
            turns.append(Turn(tool=str(turn["tool"]), args=dict(args)))
        runs[str(identifier)] = (tuple(turns), answer)
    version = parsed.get("version")
    return Transcript(version=version if isinstance(version, int) else 0, runs=runs, path=path)


def prompt_describes(turn: Turn, prompt: str) -> bool:
    """Whether the system prompt told the model about what this turn calls.

    The one judgement the replay model makes for itself. A model cannot query
    `mart_matchups` when nothing in its prompt says the table exists, and it
    cannot search card text when nothing says the tool is there; so a
    transcript whose turns are not supported by the prompt in front of it is
    not replayed. That is what makes `--prompt-override` with the schema
    removed lower the score without a provider key, and it is deliberately the
    only thing the fake infers: everything else it does is recorded.
    """
    if turn.tool == CARD_TOOL:
        return CARD_TOOL in prompt
    sql = str(turn.args.get("sql", ""))
    return all(table in prompt for table in referenced_tables(sql))


class ReplayChatModel(BaseChatModel):
    """Plays one recorded run back: the tool calls in order, then the answer.

    A second fake beside `tests/agent_fakes.ScriptedChatModel`, and the two are
    not the same thing. That one is written in Python inside a test, one script
    per assertion. This one is driven by a committed file, keyed by question,
    and lives in the package because `python -m pipeline.eval --fake` has to
    work from a command line and in a workflow, not only from pytest. The
    injectable `agent_factory` is there for the other direction: a test that
    wants the scripted model drives the same loop with it.
    """

    turns: list[Turn]
    answer: str
    model_name: str = "replay"
    index: int = 0

    @property
    def _llm_type(self) -> str:
        return "replay"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> Runnable[Any, Any]:
        """Accept the tools and ignore them: the turns are already recorded."""
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        prompt = "\n".join(
            str(message.content) for message in messages if isinstance(message, SystemMessage)
        )
        if self.index == 0 and not all(prompt_describes(turn, prompt) for turn in self.turns):
            logger.info(
                "the prompt does not describe the tools this run needs",
                extra={"prompt_length": len(prompt), "turns": len(self.turns)},
            )
            self.index = len(self.turns)
            return _one(AIMessage(content=BLIND_ANSWER))
        if self.index < len(self.turns):
            turn = self.turns[self.index]
            self.index += 1
            return _one(
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": turn.tool,
                            "args": dict(turn.args),
                            "id": f"replay-{self.index}",
                            "type": "tool_call",
                        }
                    ],
                )
            )
        return _one(AIMessage(content=self.answer))


def _one(message: AIMessage) -> ChatResult:
    return ChatResult(generations=[ChatGeneration(message=message)])


# -------------------------------------------------------------- the runner --


AgentFactory = Callable[[Question], Agent]


def live_factory(*, warehouse: Path, card_index: Path | None, model: str | None) -> AgentFactory:
    """One real agent, built once, asked every question. Needs a provider key."""
    built = build_agent(model=chat_model(model), warehouse=warehouse, card_index=card_index)

    def factory(question: Question) -> Agent:
        return built

    return factory


def replay_factory(
    transcript: Transcript, *, warehouse: Path, card_index: Path | None
) -> AgentFactory:
    """A fresh agent per question, with that question's recorded run behind it.

    Per question rather than once, because a replay model carries its position
    in the script and two questions must not share one. Everything else the
    agent is made of is the same object the live path builds.
    """

    def factory(question: Question) -> Agent:
        turns, answer = transcript.for_question(question.id)
        return build_agent(
            model=ReplayChatModel(turns=list(turns), answer=answer),
            warehouse=warehouse,
            card_index=card_index,
        )

    return factory


def run_question(question: Question, agent: Agent) -> Result:
    """Ask one question and score what came back.

    A failure inside the loop is scored as a failed question rather than
    allowed out: one provider error in the middle of a set should cost that
    question and let the other nine report, because the table is more useful
    than the traceback.
    """
    try:
        answer = agent.ask(question.question)
    except Exception as failure:  # noqa: BLE001 - one bad question must not end the run
        logger.exception("a question could not be answered", extra={"question_id": question.id})
        return Result(question=question, answer="", error=f"{type(failure).__name__}: {failure}")
    return score(question, answer.answer, [call.tool for call in answer.tool_calls])


def run_evals(
    golden: Golden,
    agent_factory: AgentFactory,
    *,
    warehouse: Path,
    card_index: Path | None = None,
    prompt_override: Path | None = None,
    fake: Path | None = None,
) -> Report:
    """Every question, in file order, with one report at the end."""
    results: list[Result] = []
    model = ""
    for question in golden.questions:
        agent = agent_factory(question)
        model = agent.model_name
        result = run_question(question, agent)
        logger.info(
            "question scored",
            extra={
                "question_id": question.id,
                "passed": result.passed,
                "failed_checks": list(result.failed_checks),
                "tools": list(result.tools_called),
            },
        )
        results.append(result)
    # The prompt as it was rendered for this run, override included, hashed so
    # two runs can be told apart by what the model was told rather than by a
    # commit that may have changed nothing the agent reads.
    rendered = system_prompt(with_card_tool=card_index is not None)
    return Report(
        golden=golden,
        results=tuple(results),
        model=model,
        prompt_sha256=hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
        prompt_override=str(prompt_override) if prompt_override else None,
        fake=str(fake) if fake else None,
        warehouse=str(warehouse),
        card_index=str(card_index) if card_index else None,
        commit=git_commit(),
    )


# -------------------------------------------------------------- reporting --


def render(report: Report) -> str:
    """The report as the table the command line prints.

    One row per question, then a line per failure saying which pattern or which
    tool was missing. The detail lines are under the table rather than in it
    because a regular expression does not fit in a column and the thing a
    reader wants first is which question, not why.
    """
    rows = [
        (
            result.question.id,
            "pass" if result.passed else "FAIL",
            ",".join(result.failed_checks) or "-",
            ",".join(result.tools_called) or "-",
        )
        for result in report.results
    ]
    headers = ("question", "result", "failed", "tools called")
    widths = [max(len(row[column]) for row in (*rows, headers)) for column in range(4)]
    lines = [
        "  ".join(header.ljust(width) for header, width in zip(headers, widths, strict=True)),
        "  ".join("-" * width for width in widths),
    ]
    lines += [
        "  ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True)).rstrip()
        for row in rows
    ]
    lines.append("")
    lines.append(f"{report.passed}/{report.total} passed")
    for result in report.results:
        if result.passed:
            continue
        lines.append(f"  {result.question.id}:")
        if result.error:
            lines.append(f"    error: {result.error}")
        for tool in result.missing_tools:
            lines.append(f"    never called: {tool}")
        for pattern in result.missing_required:
            lines.append(f"    missing: {pattern}")
        for pattern in result.present_forbidden:
            lines.append(f"    forbidden: {pattern}")
    return "\n".join(lines)


def log_to_mlflow(report: Report, *, tracking_uri: str, experiment: str) -> str | None:
    """One MLflow run per evaluation, in the `agent-evals` experiment.

    The same shape the training code uses, for the same reason: a score without
    a record of what produced it is a number somebody remembers wrongly a month
    later. The parameters are what would change the score (the model, the
    hash of the prompt that was actually rendered, the version of the golden
    file, the commit) and the metrics are the score, plus one 0/1 metric per
    question so the run table shows which question broke rather than only that
    something did.

    MLflow arrives with the `ml` extra and the agent's own extra does not pull
    it, so an install with neither logs a warning and carries on. The exit code
    is about the questions, and bookkeeping must not change it.
    """
    try:
        import mlflow
    except ImportError:
        logger.warning("mlflow is not installed, so this run was not tracked")
        return None
    if tracking_uri.startswith("file:"):
        # The same opt-in `pipeline.train` makes: MLflow 3 keeps the plain
        # directory store behind a flag, and a directory is what the default is.
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name="golden") as run:
        mlflow.log_params(
            {
                "model": report.model,
                "prompt_sha256": report.prompt_sha256,
                "prompt_override": report.prompt_override or "none",
                "golden_version": report.golden.version,
                "fake": report.fake or "none",
                "git_commit": report.commit or "unknown",
            }
        )
        mlflow.log_metrics(
            {
                "passed": float(report.passed),
                "total": float(report.total),
                "pass_rate": report.pass_rate,
                **{f"q.{result.question.id}": float(result.passed) for result in report.results},
            }
        )
        mlflow.log_dict(report.as_dict(), REPORT_ARTIFACT)
        return str(run.info.run_id)


# ------------------------------------------------------------ entry point --


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m pipeline.eval",
        description="Score the agent against the golden question set.",
        epilog=(
            "With a provider key this measures the model: "
            "`op run --env-file=.env.op -- uv run python -m pipeline.eval`. "
            "With --fake it replays recorded turns through the real tools and needs no key."
        ),
    )
    parser.add_argument(
        "--golden",
        type=Path,
        default=GOLDEN_PATH,
        metavar="PATH",
        help=f"the question set to score (default: {GOLDEN_PATH.name})",
    )
    parser.add_argument(
        "--warehouse",
        type=Path,
        default=WAREHOUSE_PATH,
        metavar="PATH",
        help="the DuckDB warehouse the questions are answered from",
    )
    parser.add_argument(
        "--card-index",
        type=Path,
        default=None,
        metavar="DIR",
        help=f"a built card index; without one the agent has no {CARD_TOOL} tool",
    )
    parser.add_argument(
        "--model", default=None, metavar="NAME", help="provider model, as `pipeline.agent` takes it"
    )
    parser.add_argument(
        "--fake",
        type=Path,
        default=None,
        metavar="PATH",
        help=f"replay recorded turns instead of calling a provider ({TRANSCRIPT_PATH.name})",
    )
    parser.add_argument(
        "--prompt-override",
        type=Path,
        default=None,
        metavar="PATH",
        help=f"replace the system prompt with this file (sets ${PROMPT_FILE_VAR})",
    )
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT, help="MLflow experiment name")
    parser.add_argument(
        "--tracking-uri",
        default=None,
        metavar="URI",
        help="MLflow tracking URI (default: MLFLOW_TRACKING_URI, else file:./data/mlruns)",
    )
    parser.add_argument(
        "--no-mlflow", action="store_true", help="score the questions and record nothing"
    )
    parser.add_argument("--json", action="store_true", help="print the report as one JSON object")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Score the golden set. 0 when every question passed, 1 when any did not."""
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(STAGE)

    if args.prompt_override is not None:
        if not args.prompt_override.is_file():
            sys.stderr.write(f"{parser.prog}: no prompt file at {args.prompt_override}\n")
            return 2
        # The flag is a surface on the environment variable rather than a
        # second mechanism: one hook, so what the evaluation measures is what a
        # person reproducing it by hand would get.
        os.environ[PROMPT_FILE_VAR] = str(args.prompt_override)

    card_index = default_card_index(args.card_index)
    if args.card_index is not None and card_index is None:
        sys.stderr.write(f"{parser.prog}: no card index directory at {args.card_index}\n")
        return 2
    if not args.warehouse.is_file():
        sys.stderr.write(
            f"{parser.prog}: no warehouse at {args.warehouse}. "
            "Build the fixture marts first; docs/evals.md has the three commands.\n"
        )
        return 2

    try:
        golden = load_golden(args.golden)
        if args.fake is not None:
            factory = replay_factory(
                load_transcript(args.fake), warehouse=args.warehouse, card_index=card_index
            )
        else:
            factory = live_factory(
                warehouse=args.warehouse, card_index=card_index, model=args.model
            )
    except GoldenError as failure:
        sys.stderr.write(f"{parser.prog}: {failure}\n")
        return 2
    except Exception as failure:  # noqa: BLE001 - a missing key arrives here too
        logger.exception("the evaluation could not be set up")
        sys.stderr.write(f"{parser.prog}: {type(failure).__name__}: {failure}\n")
        return 2

    with stage_run(STAGE) as metrics:
        report = run_evals(
            golden,
            factory,
            warehouse=args.warehouse,
            card_index=card_index,
            prompt_override=args.prompt_override,
            fake=args.fake,
        )
        metrics.rows_in = report.total
        metrics.rows_out = report.passed
        metrics.rows_quarantined = report.total - report.passed
        metrics.extra = {
            "golden_version": golden.version,
            "pass_rate": report.pass_rate,
            "model": report.model,
            "prompt_sha256": report.prompt_sha256,
            "failed": [result.question.id for result in report.results if not result.passed],
        }
        if not args.no_mlflow:
            metrics.extra["mlflow_run_id"] = log_to_mlflow(
                report,
                tracking_uri=args.tracking_uri or default_tracking_uri(),
                experiment=args.experiment,
            )

    if args.json:
        sys.stdout.write(json.dumps(report.as_dict(), indent=2) + "\n")
    else:
        emit_summary(
            logger,
            "eval summary",
            {
                "golden_version": golden.version,
                "passed": report.passed,
                "total": report.total,
                "pass_rate": round(report.pass_rate, 4),
                "model": report.model,
            },
            text=render(report),
        )
    return 0 if report.passed == report.total else 1


if __name__ == "__main__":
    raise SystemExit(main())
