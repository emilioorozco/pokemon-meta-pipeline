"""The serving import graph has no deep learning framework in it any more.

This is the test for the whole of PLA-171. On the deployed function, 105 s of
every cold `GET /warm` was the Python import of torch and transformers, paged
in over image layers Lambda fetches on first touch; the bge weights loaded in
under three seconds afterwards. The fix was to stop importing them, so the
thing to assert is an absence, and a dependency list cannot assert it: a lazy
`import torch` four frames inside a factory resolves at run time out of
whatever happens to be installed, and on a laptop something always is.

So: a subprocess, because `sys.modules` in the test process is already full of
everything the suite has touched; every module the handler reaches, including
the card lookup path, which is the one that used to do it; and the three names
that must not appear. `tests/test_lambda_consumer.py` makes the same shape of
check about the other image, and `tests/test_lambda_serve.py` makes a narrower
one about the handler module alone.
"""

import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Final

# Named rather than globbed because the point is these three and not "anything
# large": onnxruntime is a hundred megabytes and is exactly what replaced them.
FORBIDDEN: Final = ("torch", "transformers", "sentence_transformers")

PROGRAM: Final = textwrap.dedent(
    """
    import sys

    import pipeline.agent
    import pipeline.card_index
    import pipeline.lambda_serve
    import pipeline.query_embedder
    import pipeline.serve
    from opentelemetry import trace

    from pipeline.card_index import CardIndex, HashingEmbedder, build_index, read_cards
    from pipeline.telemetry import build_metrics

    # The card lookup path for real, not just its import: build a small index,
    # load it, make the tool and call it. The embedder is the hashing one, so
    # this downloads nothing, and every line of the retriever that a question
    # touches has run by the end of it.
    index_dir = sys.argv[2]
    cards = read_cards(sys.argv[1])
    build_index(cards, HashingEmbedder(), index_dir)
    index = CardIndex.load(index_dir)
    tool = pipeline.card_index.make_lookup_cards_tool(
        index_dir,
        tracer=trace.get_tracer(__name__),
        metrics=build_metrics(),
        index=index,
    )
    assert tool.invoke({"query": "bench damage", "k": 2})
    assert pipeline.card_index.warm_index(index)

    # And the agent's tool set over the same index, which is what `/warm`
    # builds and the one place the serving process constructs an embedder.
    toolset = pipeline.agent.marts_tools(
        tracer=trace.get_tracer(__name__),
        metrics=build_metrics(),
        card_index=index_dir,
    )
    assert len(toolset.tools) == 2

    print(",".join(sorted(n for n in sys.modules if n.split(".")[0] in %r)))
    """
).strip() % (FORBIDDEN,)


CORPUS: Final = Path(__file__).parent / "card_text.jsonl"


def test_the_serving_modules_and_the_card_lookup_import_no_framework(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-c", PROGRAM, str(CORPUS), str(tmp_path / "card_index")],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "", result.stdout
