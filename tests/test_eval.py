"""The golden evaluation: the scorer, the shape of the set, and one real run.

Three suites, split the way the rest of the project splits them.

The fast half is the scorer and the loader, and it is most of the value. Both
are pure functions over strings and dictionaries, so every rule of the grading
is one assertion with no warehouse, no model and no network: what a regular
expression entry matches, that an extra tool call is not a failure, that a
forbidden player token fails a question whose facts are all present, and that
a golden file with a duplicate id or an unknown tool in it is refused at load
rather than silently scoring nothing.

The committed set is checked here too, because `evals/golden.yaml` is data and
data rots: ten questions, unique ids, every tool name real, every question
answerable, every question forbidding the player-token shape, and a recorded
run in `evals/transcript.yaml` for each one.

The `dbt` half runs the loop for real over the fixture warehouse, and it is
the one that would catch a harness that scores nothing: ten out of ten with the
recorded turns replayed through the real tools, fewer than ten when the answers
stop carrying the facts, and fewer than ten when the system prompt is replaced
with the deliberately broken one.
"""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Final

import pytest

from pipeline import card_index
from pipeline import eval as evals
from pipeline.prompts import PROMPT_FILE_VAR, system_prompt
from tests.agent_fakes import final, scripted, tool_call

CORPUS: Final = Path(__file__).parent / "card_text.jsonl"
# The shape of an irreversible player token, which every question forbids.
TOKEN_PATTERN: Final = "re:[0-9a-f]{16}"


def question(**overrides: object) -> evals.Question:
    """A golden question with every field defaulted to something harmless."""
    fields: dict[str, object] = {
        "id": "example",
        "question": "how many games",
        "expect_tools": (evals.SQL_TOOL,),
        "require": ("re:\\b12 games",),
        "forbid": (TOKEN_PATTERN,),
    }
    fields.update(overrides)
    return evals.Question(**fields)  # type: ignore[arg-type]


# ------------------------------------------------------------------ fast --


def test_a_plain_entry_is_a_substring_and_case_does_not_matter() -> None:
    """Capitalisation is wording, and this file grades facts."""
    assert evals.matches("Dragapult control", "the Dragapult Control deck won")
    assert not evals.matches("Dragapult control", "the Dragapult deck won")


def test_an_entry_marked_as_a_regular_expression_is_one() -> None:
    assert evals.matches("re:\\b4 games", "that week holds 4 games")
    assert not evals.matches("re:\\b4 games", "that week holds 14 games")
    # Without the marker the same text is looked for literally, brackets included.
    assert not evals.matches("\\b4 games", "that week holds 4 games")


def test_a_question_passes_when_the_tools_the_facts_and_the_silence_all_hold() -> None:
    result = evals.score(question(), "12 games in the corpus", [evals.SQL_TOOL])
    assert result.passed
    assert result.failed_checks == ()


def test_an_expected_tool_that_was_never_called_fails_the_question() -> None:
    result = evals.score(question(), "12 games in the corpus", [])
    assert not result.passed
    assert result.failed_checks == (evals.CHECK_TOOLS,)
    assert result.missing_tools == (evals.SQL_TOOL,)


def test_an_unexpected_tool_is_reported_and_costs_nothing() -> None:
    """Which tools the answer is built from is a fact; the detour is not."""
    result = evals.score(question(), "12 games in the corpus", [evals.SQL_TOOL, evals.CARD_TOOL])
    assert result.passed
    assert result.unexpected_tools == (evals.CARD_TOOL,)


def test_a_missing_fact_names_the_pattern_that_was_missing() -> None:
    result = evals.score(question(), "quite a lot of games", [evals.SQL_TOOL])
    assert result.failed_checks == (evals.CHECK_REQUIRE,)
    assert result.missing_required == ("re:\\b12 games",)


def test_a_player_token_fails_a_question_whose_facts_are_all_there() -> None:
    """The boundary check: a right answer that hands over a token is a wrong answer."""
    result = evals.score(
        question(), "12 games, the best of them by 0123456789abcdef", [evals.SQL_TOOL]
    )
    assert not result.passed
    assert result.failed_checks == (evals.CHECK_FORBID,)
    assert result.present_forbidden == (TOKEN_PATTERN,)


def test_every_failed_check_is_reported_rather_than_the_first() -> None:
    result = evals.score(question(), "player 0123456789abcdef", [])
    assert result.failed_checks == (
        evals.CHECK_TOOLS,
        evals.CHECK_REQUIRE,
        evals.CHECK_FORBID,
    )


# ------------------------------------------------------- the committed set --


@pytest.fixture(scope="module")
def golden() -> evals.Golden:
    return evals.load_golden()


def test_the_golden_set_is_ten_questions_with_unique_identifiers(golden: evals.Golden) -> None:
    assert len(golden.questions) == 10
    identifiers = [entry.id for entry in golden.questions]
    assert len(set(identifiers)) == len(identifiers)
    assert golden.version >= 1


def test_every_question_is_answerable_and_names_only_real_tools(golden: evals.Golden) -> None:
    for entry in golden.questions:
        assert entry.question.strip(), entry.id
        assert entry.require, entry.id
        # An identifier becomes an MLflow metric name, `q.<id>`.
        assert entry.id.replace("_", "").isalnum(), entry.id
        for tool in entry.expect_tools:
            assert tool in evals.VALID_TOOLS, (entry.id, tool)


def test_every_question_forbids_the_player_token_shape(golden: evals.Golden) -> None:
    """The one assertion that has to hold on every question, not just the player one."""
    for entry in golden.questions:
        assert TOKEN_PATTERN in entry.forbid, entry.id


def test_the_set_covers_both_tools_and_the_questions_with_no_good_answer(
    golden: evals.Golden,
) -> None:
    by_id = {entry.id: entry for entry in golden.questions}
    assert evals.CARD_TOOL in by_id["card_text_lookup"].expect_tools
    assert set(by_id["card_text_and_marts"].expect_tools) == {evals.SQL_TOOL, evals.CARD_TOOL}
    # The refusal question asserts nothing about tools: reading the member
    # summary to describe the distribution is as correct as not reading it.
    assert by_id["player_identity_refusal"].expect_tools == ()
    assert by_id["matchup_with_no_games"].require


def test_the_transcript_has_a_recorded_run_for_every_question(golden: evals.Golden) -> None:
    transcript = evals.load_transcript()
    for entry in golden.questions:
        turns, answer = transcript.for_question(entry.id)
        assert answer
        for turn in turns:
            assert turn.tool in evals.VALID_TOOLS, entry.id


def test_a_recorded_query_is_one_the_sql_gate_would_allow() -> None:
    """The transcript is replayed through the real tool, so its SQL has to be legal."""
    from pipeline.agent import validate_sql

    transcript = evals.load_transcript()
    for turns, _ in transcript.runs.values():
        for turn in turns:
            if turn.tool == evals.SQL_TOOL:
                assert validate_sql(str(turn.args["sql"])) is None


# --------------------------------------------------------------- loading --


def write_golden(path: Path, body: str) -> Path:
    path.write_text(f"version: 1\nquestions:\n{body}", encoding="utf-8")
    return path


ONE_QUESTION: Final = """\
  - id: only
    question: how many games
    expect_tools: [query_marts]
    require: ["re:\\\\b12 games"]
"""


def test_a_well_formed_file_loads(tmp_path: Path) -> None:
    golden = evals.load_golden(write_golden(tmp_path / "g.yaml", ONE_QUESTION))
    assert [entry.id for entry in golden.questions] == ["only"]
    assert golden.questions[0].forbid == ()


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        (ONE_QUESTION + ONE_QUESTION, "duplicate id"),
        (
            "  - id: only\n    question: q\n    require: [x]\n    expect_tools: [nope]\n",
            "not a tool",
        ),
        ("  - id: only\n    question: ''\n    require: [x]\n", "`question` is empty"),
        ("  - id: ''\n    question: q\n    require: [x]\n", "needs an `id`"),
        ("  - id: only\n    question: q\n", "`require` is empty"),
        ("  - id: only\n    question: q\n    require: ['re:[']\n", "not a regular expression"),
        ("  - id: only\n    question: q\n    require: ['  ']\n", "empty pattern"),
    ],
)
def test_a_golden_file_that_could_not_score_anything_is_refused(
    tmp_path: Path, body: str, expected: str
) -> None:
    """Every one of these would otherwise be a question that quietly never fails."""
    with pytest.raises(evals.GoldenError, match=expected):
        evals.load_golden(write_golden(tmp_path / "g.yaml", body))


def test_a_file_that_is_not_a_question_set_at_all_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "g.yaml"
    path.write_text("just a string\n", encoding="utf-8")
    with pytest.raises(evals.GoldenError, match="expected a mapping"):
        evals.load_golden(path)
    with pytest.raises(evals.GoldenError, match="questions"):
        evals.load_golden(write_golden(path, "  []\n"))


def test_a_transcript_missing_a_run_says_which_one(tmp_path: Path) -> None:
    path = tmp_path / "t.yaml"
    path.write_text("version: 1\nruns:\n  only:\n    answer: fine\n", encoding="utf-8")
    transcript = evals.load_transcript(path)
    assert transcript.for_question("only") == ((), "fine")
    with pytest.raises(evals.GoldenError, match="missing_one"):
        transcript.for_question("missing_one")


def test_a_transcript_turn_has_to_name_a_tool(tmp_path: Path) -> None:
    path = tmp_path / "t.yaml"
    path.write_text(
        "version: 1\nruns:\n  only:\n    answer: fine\n    turns:\n      - tool: nope\n",
        encoding="utf-8",
    )
    with pytest.raises(evals.GoldenError, match="has to name a tool"):
        evals.load_transcript(path)


# ------------------------------------------------------- the prompt hook --


@pytest.fixture
def no_override() -> Iterator[None]:
    """Make sure a leaked environment variable cannot change what these assert."""
    with pytest.MonkeyPatch.context() as patch:
        patch.delenv(PROMPT_FILE_VAR, raising=False)
        yield


def test_the_generated_prompt_is_used_when_nothing_overrides_it(no_override: None) -> None:
    assert "mart_matchups" in system_prompt()


def test_an_override_file_replaces_the_whole_prompt(tmp_path: Path, no_override: None) -> None:
    path = tmp_path / "prompt.txt"
    path.write_text("be brief", encoding="utf-8")
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv(PROMPT_FILE_VAR, str(path))
        # Not merged with the generated one: "the rules are missing" has to
        # mean the rules really are missing.
        assert system_prompt() == "be brief"
        assert system_prompt(with_card_tool=True) == "be brief"


def test_the_committed_broken_prompt_has_neither_the_schema_nor_the_rules() -> None:
    text = evals.BROKEN_PROMPT_PATH.read_text(encoding="utf-8")
    assert "mart_matchups" not in text
    assert "seen_rate" not in text
    assert "sample size" not in text


def test_the_replay_model_will_not_call_a_tool_the_prompt_never_described() -> None:
    """The one judgement the fake makes, and the reason a broken prompt scores lower."""
    good = system_prompt(with_card_tool=True)
    broken = evals.BROKEN_PROMPT_PATH.read_text(encoding="utf-8")
    sql_turn = evals.Turn(tool=evals.SQL_TOOL, args={"sql": "select 1 from mart_matchups"})
    card_turn = evals.Turn(tool=evals.CARD_TOOL, args={"query": "bench damage"})
    assert evals.prompt_describes(sql_turn, good)
    assert evals.prompt_describes(card_turn, good)
    assert not evals.prompt_describes(sql_turn, broken)
    assert not evals.prompt_describes(card_turn, broken)


# ------------------------------------------------------------- reporting --


def report_of(*results: evals.Result) -> evals.Report:
    return evals.Report(
        golden=evals.Golden(version=1, questions=(), path=Path("evals/golden.yaml")),
        results=results,
        model="replay",
        prompt_sha256="0" * 64,
    )


def test_the_table_names_the_failure_and_ends_with_the_score() -> None:
    report = report_of(
        evals.score(question(id="good"), "12 games", [evals.SQL_TOOL]),
        evals.score(question(id="bad"), "no idea", []),
    )
    rendered = evals.render(report)
    assert "1/2 passed" in rendered
    assert "good" in rendered and "pass" in rendered
    assert f"never called: {evals.SQL_TOOL}" in rendered
    assert "missing: re:\\b12 games" in rendered


def test_the_report_is_json_and_carries_every_question() -> None:
    report = report_of(evals.score(question(id="good"), "12 games", [evals.SQL_TOOL]))
    payload = json.loads(json.dumps(report.as_dict()))
    assert payload["passed"] == 1
    assert payload["pass_rate"] == 1.0
    assert [entry["id"] for entry in payload["questions"]] == ["good"]


def test_a_question_that_raised_is_a_failure_and_not_an_end_to_the_run() -> None:
    class Exploding:
        model_name = "boom"

        def ask(self, question_text: str) -> None:
            raise RuntimeError("the provider said no")

    result = evals.run_question(question(id="boom"), Exploding())  # type: ignore[arg-type]
    assert result.failed_checks[0] == evals.CHECK_ERROR
    assert "the provider said no" in str(result.error)


# ------------------------------------------------------------ setup errors --


def test_a_missing_warehouse_is_exit_two_rather_than_ten_failures(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A broken harness and a broken agent are different news."""
    code = evals.main(["--warehouse", str(tmp_path / "nothing.duckdb"), "--no-mlflow"])
    assert code == 2
    assert "no warehouse" in capsys.readouterr().err


def test_a_missing_prompt_override_is_exit_two(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = evals.main(["--prompt-override", str(tmp_path / "gone.txt"), "--no-mlflow"])
    assert code == 2
    assert "no prompt file" in capsys.readouterr().err


def test_a_missing_card_index_is_exit_two(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "meta.duckdb").write_bytes(b"")
    code = evals.main(
        [
            "--warehouse",
            str(tmp_path / "meta.duckdb"),
            "--card-index",
            str(tmp_path / "nothing"),
            "--no-mlflow",
        ]
    )
    assert code == 2
    assert "no card index" in capsys.readouterr().err


# --------------------------------------------------------------------- ml --


@pytest.mark.ml
def test_a_run_is_logged_to_mlflow_with_one_metric_per_question(tmp_path: Path) -> None:
    """The record, so an agent change is comparable the way a model change is."""
    import mlflow

    report = report_of(
        evals.score(question(id="good"), "12 games", [evals.SQL_TOOL]),
        evals.score(question(id="bad"), "no idea", []),
    )
    uri = f"file:{tmp_path / 'mlruns'}"
    run_id = evals.log_to_mlflow(report, tracking_uri=uri, experiment="agent-evals-test")
    assert run_id

    mlflow.set_tracking_uri(uri)
    logged = mlflow.get_run(run_id)
    assert logged.data.metrics["passed"] == 1.0
    assert logged.data.metrics["pass_rate"] == 0.5
    assert logged.data.metrics["q.good"] == 1.0
    assert logged.data.metrics["q.bad"] == 0.0
    assert logged.data.params["prompt_sha256"] == "0" * 64
    assert logged.data.params["golden_version"] == "1"


# ------------------------------------------------------------------- dbt --


@pytest.fixture(scope="module")
def hashed_index(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The fixture card corpus indexed with the embedder that needs no download."""
    out = tmp_path_factory.mktemp("card-index") / "index"
    cards = card_index.read_cards(CORPUS)
    assert card_index.build_index(cards, card_index.HashingEmbedder(), out) == len(cards)
    return out


@pytest.mark.dbt
def test_the_whole_set_passes_against_the_fixture_marts(
    gold_from_fixtures: Path, hashed_index: Path, golden: evals.Golden
) -> None:
    """Ten out of ten, with every query really run against the warehouse dbt built.

    The recorded turns go through the real graph, the real SQL gate and real
    DuckDB, so this fails if a mart is renamed, if a number in the fixture
    corpus moves, or if the gate starts refusing a query the set depends on.
    """
    report = evals.run_evals(
        golden,
        evals.replay_factory(
            evals.load_transcript(), warehouse=gold_from_fixtures, card_index=hashed_index
        ),
        warehouse=gold_from_fixtures,
        card_index=hashed_index,
    )
    assert report.passed == report.total == 10, evals.render(report)
    assert report.model == "replay"
    # Every question really called something, and the card tool was really used.
    called = {tool for result in report.results for tool in result.tools_called}
    assert called == {evals.SQL_TOOL, evals.CARD_TOOL}


@pytest.mark.dbt
def test_answers_without_the_facts_score_below_ten(
    gold_from_fixtures: Path, golden: evals.Golden
) -> None:
    """The scorer is load bearing: a run that queries and then says nothing fails.

    This is the injected-factory path, with `tests/agent_fakes` in the model
    slot rather than the transcript, so both ways of running the loop without a
    provider are covered.
    """
    from pipeline.agent import build_agent

    def factory(entry: evals.Question) -> object:
        return build_agent(
            model=scripted(
                tool_call(evals.SQL_TOOL, "call-1", sql="select games from mart_matchups"),
                final("It depends. Roughly half, I would say."),
            ),
            warehouse=gold_from_fixtures,
        )

    report = evals.run_evals(
        golden,
        factory,  # type: ignore[arg-type]
        warehouse=gold_from_fixtures,
    )
    assert report.passed == 0
    assert all(evals.CHECK_REQUIRE in result.failed_checks for result in report.results)


@pytest.mark.dbt
def test_the_broken_prompt_drops_the_score(
    gold_from_fixtures: Path, hashed_index: Path, golden: evals.Golden, tmp_path: Path
) -> None:
    """The claim the rules are load bearing, run as an experiment rather than asserted.

    With a provider key the fall comes from the model: nothing tells it to cite
    a sample size or to refuse to guess. Here it comes from the replay model
    declining to query tables the prompt never described, which is the same
    mechanism a real model is subject to and the only inference the fake makes.
    """
    warehouse, index = gold_from_fixtures, hashed_index
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv(PROMPT_FILE_VAR, str(evals.BROKEN_PROMPT_PATH))
        report = evals.run_evals(
            golden,
            evals.replay_factory(evals.load_transcript(), warehouse=warehouse, card_index=index),
            warehouse=warehouse,
            card_index=index,
            prompt_override=evals.BROKEN_PROMPT_PATH,
        )
    assert report.passed < 10
    assert report.prompt_sha256 != _good_prompt_sha()
    assert report.prompt_override is not None


@pytest.mark.dbt
def test_the_command_line_prints_the_table_and_exits_zero(
    gold_from_fixtures: Path, hashed_index: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = evals.main(
        [
            "--fake",
            str(evals.TRANSCRIPT_PATH),
            "--warehouse",
            str(gold_from_fixtures),
            "--card-index",
            str(hashed_index),
            "--no-mlflow",
        ]
    )
    printed = capsys.readouterr().out
    assert code == 0, printed
    assert "10/10 passed" in printed
    assert "matchup_win_rate" in printed


@pytest.mark.dbt
def test_the_json_report_is_machine_readable(
    gold_from_fixtures: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Without a card index the card question fails, which is the exit code to check."""
    code = evals.main(
        [
            "--fake",
            str(evals.TRANSCRIPT_PATH),
            "--warehouse",
            str(gold_from_fixtures),
            "--no-mlflow",
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert code == 1
    assert payload["total"] == 10
    assert payload["card_index"] is None
    failed = {entry["id"] for entry in payload["questions"] if not entry["passed"]}
    assert "card_text_lookup" in failed


def _good_prompt_sha() -> str:
    import hashlib

    with pytest.MonkeyPatch.context() as patch:
        patch.delenv(PROMPT_FILE_VAR, raising=False)
        return hashlib.sha256(system_prompt(with_card_tool=True).encode("utf-8")).hexdigest()
