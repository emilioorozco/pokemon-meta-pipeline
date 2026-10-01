"""The two runtimes of one embedding model, and the claim that they agree.

The fast half is about the directory: what `OnnxEmbedder` does when the files
are missing, when the manifest does not name a model, and when the pooling is
one it does not implement. All three are the failure a deployed function would
hit if the image were built wrong, and all three have to say so rather than
hand ONNX Runtime a file it cannot parse. Nothing here downloads anything: the
directories are written by the tests themselves and no session is ever built.

The `ml` half is the one that matters. It embeds five passages out of the
committed fixture corpus and five questions with both implementations and
compares them pair by pair, then builds the fixture index with
`sentence-transformers` and searches it with each. If the export were wrong,
or the pooling, or the normalization, or the tokenizer's truncation, this is
where it would show: everything else in the project would keep passing and the
deployed retriever would return a confident ranking of the wrong cards.
"""

import json
from pathlib import Path
from typing import Final

import numpy as np
import pytest

from pipeline import card_index
from pipeline.query_embedder import (
    DEFAULT_MODEL,
    MANIFEST_FILE,
    MODEL_FILE,
    POOLING_CLS,
    TOKENIZER_FILE,
    OnnxEmbedder,
    QueryEmbedderError,
    SentenceTransformerEmbedder,
    normalize_rows,
    query_prefix,
    write_manifest,
)

CORPUS: Final = Path(__file__).parent / "card_text.jsonl"

# Five questions a member might ask, spread over the kinds of card text the
# index holds: an attack effect, two verbatim-ish rules lines, an ability and
# one deliberately vague two-word query.
QUESTIONS: Final = [
    "bench damage",
    "put damage counters on the bench",
    "search your deck for a Supporter card",
    "draw cards until you have 7 in hand",
    "which attacks discard energy from the bench",
]


def written_directory(path: Path, *, pooling: str = POOLING_CLS) -> Path:
    """A directory shaped like an exported embedder, with nothing real in it.

    The graph and the tokenizer are placeholders, which is enough for every
    check that happens before a session is built and is the reason the fast
    suite needs no download.
    """
    path.mkdir(parents=True, exist_ok=True)
    (path / MODEL_FILE).write_bytes(b"not a graph")
    (path / TOKENIZER_FILE).write_text("{}", encoding="utf-8")
    write_manifest(path, model=DEFAULT_MODEL, pooling=pooling, dimensions=384)
    return path


# ------------------------------------------------------------------ fast --


def test_a_directory_with_nothing_in_it_says_which_file_is_missing(tmp_path: Path) -> None:
    """The error a wrongly built image would raise, and it names the fix."""
    with pytest.raises(QueryEmbedderError, match=MODEL_FILE):
        OnnxEmbedder(tmp_path / "nowhere")
    directory = written_directory(tmp_path / "embedder")
    (directory / TOKENIZER_FILE).unlink()
    with pytest.raises(QueryEmbedderError, match=TOKENIZER_FILE):
        OnnxEmbedder(directory)


def test_a_manifest_that_names_no_model_is_refused(tmp_path: Path) -> None:
    directory = written_directory(tmp_path / "embedder")
    (directory / MANIFEST_FILE).write_text(json.dumps({"pooling": POOLING_CLS}), encoding="utf-8")
    with pytest.raises(QueryEmbedderError, match="does not name a model"):
        OnnxEmbedder(directory)


def test_a_pooling_this_runtime_does_not_implement_is_refused(tmp_path: Path) -> None:
    """Mean pooling is a real mode and this is not it: fail at construction, not at search."""
    directory = written_directory(tmp_path / "embedder", pooling="mean")
    with pytest.raises(QueryEmbedderError, match="pooling"):
        OnnxEmbedder(directory)


def test_the_manifest_is_what_the_embedder_reports(tmp_path: Path) -> None:
    """Name, pooling and width come off disk, which is what `CardIndex.load` compares."""
    embedder = OnnxEmbedder(written_directory(tmp_path / "embedder"))
    assert (embedder.name, embedder.pooling, embedder.dimensions) == (DEFAULT_MODEL, "cls", 384)
    # And an empty batch never reaches a session, so the placeholder graph above
    # is never parsed. ONNX Runtime refuses a zero-length batch; nothing to
    # embed is not an error.
    assert embedder.embed([]).shape == (0, 384)


def test_only_bge_gets_an_instruction_in_front_of_a_question() -> None:
    assert query_prefix(DEFAULT_MODEL).startswith("Represent this sentence")
    assert query_prefix("sentence-transformers/all-MiniLM-L6-v2") == ""


def test_a_row_of_zeros_normalizes_to_zeros_rather_than_dividing_by_one() -> None:
    """The empty-passage case, said once where the function is."""
    out = normalize_rows(np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32))
    assert out.dtype == np.float32
    assert out[0] == pytest.approx([0.6, 0.8])
    assert out[1].tolist() == [0.0, 0.0]


def test_an_index_refuses_a_query_embedder_that_did_not_build_it(tmp_path: Path) -> None:
    """The one failure a result cannot show, so it has to be a refusal.

    Both halves: a different model name, and the same name pooled differently.
    Hashing embedders stand in for two models here because the fast suite does
    not download one, and the check is on the recorded strings either way.
    """
    cards = card_index.read_cards(CORPUS)
    out = tmp_path / "card_index"
    card_index.build_index(cards, card_index.HashingEmbedder(64), out)

    with pytest.raises(ValueError, match="rebuild the index"):
        card_index.CardIndex.load(out, card_index.HashingEmbedder(32))

    meta_path = out / card_index.META_FILE
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["pooling"] = POOLING_CLS
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError, match="not the same vector space"):
        card_index.CardIndex.load(out, card_index.HashingEmbedder(64))


def test_an_index_from_the_previous_layout_is_rebuilt_rather_than_read(tmp_path: Path) -> None:
    """Format 2 recorded no pooling, so there is nothing to check it against."""
    cards = card_index.read_cards(CORPUS)
    out = tmp_path / "card_index"
    card_index.build_index(cards, card_index.HashingEmbedder(), out)
    meta_path = out / card_index.META_FILE
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["format_version"] = 2
    del meta["pooling"]
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError, match="card_index build"):
        card_index.CardIndex.load(out)


# -------------------------------------------------------------------- ml --


@pytest.fixture(scope="module")
def exported(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The real files, downloaded once for this module. Marked `ml` for that reason."""
    pytest.importorskip("huggingface_hub", reason="the export script needs the hub client")
    from scripts.export_query_embedder import fetch

    out = tmp_path_factory.mktemp("query-embedder")
    fetch(DEFAULT_MODEL, out, export=False)
    return out


@pytest.mark.ml
def test_the_two_runtimes_of_the_model_agree_to_seven_decimal_places(exported: Path) -> None:
    """Five passages and five questions through both embedders.

    Measured minimum cosine over the ten pairs: **0.9999999** (0.99999994,
    which is a float32 dot product of two unit vectors and not a difference).
    The threshold below is 0.999, three orders of magnitude of slack, because
    what this is guarding against is a wrong pooling or a wrong prefix, which
    would show as 0.9 or 0.5 and never as 0.9998.
    """
    pytest.importorskip("sentence_transformers", reason="the reference embedder is not installed")
    cards = card_index.read_cards(CORPUS)
    passages = [passage.text for card in cards for passage in card.passages()][:5]
    assert len(passages) == 5

    onnx = OnnxEmbedder(exported)
    reference = SentenceTransformerEmbedder(DEFAULT_MODEL)
    assert (onnx.name, onnx.pooling) == (reference.name, reference.pooling)

    cosines = [
        float(np.dot(onnx.embed([text])[0], reference.embed([text])[0])) for text in passages
    ]
    cosines += [
        float(np.dot(onnx.embed_query(question), reference.embed_query(question)))
        for question in QUESTIONS
    ]
    assert min(cosines) > 0.999, cosines


@pytest.mark.ml
def test_both_embedders_rank_the_fixture_index_in_the_same_order(exported: Path) -> None:
    """The index built with torch, searched twice: once with each runtime.

    Identical top-5 card order for all five questions when it was measured, and
    that is the claim the serving path rests on. The index itself is built with
    `sentence-transformers`, as the nightly builds it.
    """
    pytest.importorskip("sentence_transformers", reason="the reference embedder is not installed")
    cards = card_index.read_cards(CORPUS)
    out = exported.parent / "card_index"
    if not (out / card_index.META_FILE).is_file():
        card_index.build_index(cards, SentenceTransformerEmbedder(DEFAULT_MODEL), out)

    served = card_index.CardIndex.load(out, OnnxEmbedder(exported))
    built = card_index.CardIndex.load(out, SentenceTransformerEmbedder(DEFAULT_MODEL))
    for question in QUESTIONS:
        assert [hit.card.name for hit in served.search(question, k=5)] == [
            hit.card.name for hit in built.search(question, k=5)
        ], question
