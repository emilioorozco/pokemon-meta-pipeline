"""A scripted chat model and a scripted SQL gate, so the tests need no API key.

Not a test module: it is imported by `tests/test_agent.py`,
`tests/test_card_index.py` and `tests/test_eval.py`, which need the same fakes.

LangChain ships several fake chat models and none of them can be given tools:
`bind_tools` on `BaseChatModel` raises `NotImplementedError`, and
`create_agent` binds the tools before it does anything else, so a fake that
cannot be bound cannot be used in an agent at all. This is the smallest
subclass that can: it returns the messages it was handed, in order, and accepts
whatever tools it is offered without looking at them.

That is the point of it. Nothing about the tool call is mocked. The scripted
`AIMessage` carries a real `tool_calls` entry, LangChain routes it to the real
`StructuredTool`, the tool validates the SQL for real, runs it against a real
DuckDB warehouse built by a real dbt run, and the `ToolMessage` that comes back
is what the next scripted turn is produced after. The only thing that is not
real is the decision about which SQL to write, which is exactly the part that
costs a key and is not deterministic.
"""

from collections.abc import Sequence
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable

from pipeline.sql_gate import GATE_JEV, GateDecision


class ScriptedChatModel(BaseChatModel):
    """Returns pre-written assistant turns in order, tool calls and all.

    `seen` records the message lists it was called with, so a test can assert
    that the tool's output really came back to the model rather than only that
    the tool ran.
    """

    responses: list[AIMessage]
    model_name: str = "scripted-fake"
    index: int = 0
    seen: list[list[BaseMessage]] = []

    @property
    def _llm_type(self) -> str:
        return "scripted-fake"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> Runnable[Any, Any]:
        """Accept the tools and ignore them: the turns are already written."""
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.seen.append(list(messages))
        if self.index >= len(self.responses):
            raise AssertionError(
                f"the scripted model ran out of turns after {len(self.responses)}: "
                f"the agent asked for one more than the script has"
            )
        response = self.responses[self.index]
        self.index += 1
        return ChatResult(generations=[ChatGeneration(message=response)])


def tool_call(name: str, call_id: str, **args: Any) -> AIMessage:
    """An assistant turn that calls one tool and says nothing else."""
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )


def final(text: str) -> AIMessage:
    """An assistant turn that is the answer."""
    return AIMessage(content=text)


def scripted(*responses: AIMessage) -> ScriptedChatModel:
    """A model that will produce these turns, in this order, once each."""
    return ScriptedChatModel(responses=list(responses), seen=[])


class FakeGate:
    """A `SqlGate` that answers however the test told it to, and remembers being asked.

    It stands in for a provider the test suite has no key for and must not
    call. What it is not is a mock of the wiring: the decision it returns goes
    through the real tool, the real span, the real counter and the real
    evaluation report, so a test that asserts the eval table says `refused` is
    asserting the whole path from a verdict to a column.

    The adapters that really talk to Jev are tested separately, in
    `tests/test_sql_gate.py`, against a faked HTTP layer rather than a faked
    gate: that is where the request shape, the parsing and the error paths are
    covered, and this is where the effect of a verdict is.
    """

    def __init__(
        self,
        *,
        allowed: bool = True,
        confidence: float = 0.95,
        reason: str = "the fake gate said so",
        cost_usd: float = 0.000_012,
        input_tokens: int = 286,
        errored: bool = False,
    ) -> None:
        self.name = GATE_JEV
        self.decision = GateDecision(
            allowed=allowed,
            confidence=confidence,
            reason=reason,
            cost_usd=cost_usd,
            input_tokens=input_tokens,
            gate=GATE_JEV,
            errored=errored,
        )
        self.judged: list[tuple[str, str]] = []

    def judge(self, question: str, sql: str, schema_summary: str) -> GateDecision:
        self.judged.append((question, sql))
        return self.decision
