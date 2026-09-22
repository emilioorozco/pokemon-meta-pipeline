"""A scripted chat model, so the agent tests run the real loop with no API key.

Not a test module: it is imported by `tests/test_agent.py` and
`tests/test_card_index.py`, which both need the same fake.

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
