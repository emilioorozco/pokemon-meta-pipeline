"""Two ways to run one embedding model: ONNX Runtime to serve, torch to build.

The index and the question have to land in the same vector space, and until
now there was one implementation on both sides: `sentence-transformers` over
`BAAI/bge-small-en-v1.5`, which means `import torch` and `import transformers`
in every process that answers a question. On a laptop that is a second. On the
deployed Lambda function it was **105 s of every cold `GET /warm`**: thousands
of small files, paged in over image layers that Lambda fetches the first time
something touches them. The weights themselves loaded in under three seconds
once the imports were done, so the cost was the frameworks and not the model.

So the serving image no longer has them. `OnnxEmbedder` runs the same network,
exported once to a single ONNX graph, through ONNX Runtime, and tokenizes with
the `tokenizers` library reading the model's own `tokenizer.json`. Those two
wheels are about 60 MB together and neither imports a deep learning framework.
`SentenceTransformerEmbedder` stays, because the nightly `build_card_index`
and `python -m pipeline.card_index build` embed the whole corpus and the
reference implementation is the right thing to build an index with: it is the
definition of what the vectors mean, and the build runs on a runner with no
cold start to pay.

**They have to agree, and they do.** `tests/test_query_embedder.py` embeds five
card passages and five questions with both and compares them; the smallest
cosine between a pair is 0.9999999, and the top-5 retrieval order over the
fixture passages is identical. That is the same graph and the same weights in
two runtimes, so anything less would have been a bug in the export.

**The parts that have to match the index exactly** are the pooling and the
normalization, because they are not in the graph. bge-small-en-v1.5 is CLS
pooling then L2 normalization (`1_Pooling/config.json` says
`pooling_mode_cls_token`, and the model's third module is a `Normalize`), so
the ONNX side takes `last_hidden_state[:, 0, :]` and divides by the row norm.
Truncation is 512 tokens, which is the model's `max_seq_length`; no card
passage comes close, and a query that did would be truncated the same way on
both sides. The query instruction prefix is bge's and is applied by model name,
for the reason `pipeline.card_index` gives where the constant is used.

**The files, and where they come from.** `BAAI/bge-small-en-v1.5` publishes
`onnx/model.onnx` in its own Hugging Face repository, 133,093,490 bytes, fp32
and in one file with no external data beside it, so nothing has to be exported:
`scripts/export_query_embedder.py` downloads that and `tokenizer.json` and
writes a small `embedder.json` next to them saying which model and which
pooling they are. A directory with an external data file in it works too, since
ONNX Runtime looks for it beside the graph; there just is not one here.

`pipeline.config.QUERY_EMBEDDER_DIR` is where they are looked for, which is
`PRA_QUERY_EMBEDDER_DIR` when it is set and `.models/query-embedder` under the
repository otherwise. `Dockerfile.agent` fetches them in a build stage that is
not part of the runtime image and sets the variable to where it put them. The
fast test suite never reads any of this: it builds its index with
`HashingEmbedder` and constructs an `OnnxEmbedder` only over a directory a test
wrote itself, so `uv run pytest` downloads nothing.

**`embedder.json` is why a mismatch can be caught.** It names the model and the
pooling the files implement, so `CardIndex.load` can compare them against what
`meta.json` records the index was built with and refuse the pair outright. A
query vector from a different model is not wrong in any way a result can show:
it ranks something, confidently, and the ranking means nothing.
"""

import json
import logging
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final, Protocol

import numpy as np

from pipeline.config import QUERY_EMBEDDER_DIR

logger = logging.getLogger(__name__)

DEFAULT_MODEL: Final = "BAAI/bge-small-en-v1.5"
FALLBACK_MODEL: Final = "sentence-transformers/all-MiniLM-L6-v2"

# bge is trained with an instruction on the query side and none on the document
# side, and skipping it costs real accuracy: without the prefix, "put damage
# counters on the bench" does not rank the card that does exactly that first,
# and with it, it does. Only the bge family wants it, which is why it is applied
# by model name rather than to everything.
BGE_QUERY_PREFIX: Final = "Represent this sentence for searching relevant passages: "

# What a model does with the token vectors before anything is normalized. The
# strings are the ones `sentence-transformers` reports for its own pooling
# module, so an index built through it records a value this module understands.
POOLING_CLS: Final = "cls"
POOLING_MEAN: Final = "mean"
# Not a pooled model at all: `HashingEmbedder` produces a row directly.
POOLING_NONE: Final = "none"

# The files a serving embedder needs. The graph, the tokenizer, and the note
# saying what they are; an external data file, if a future model has one, sits
# beside the graph and ONNX Runtime finds it without being told.
MODEL_FILE: Final = "model.onnx"
TOKENIZER_FILE: Final = "tokenizer.json"
MANIFEST_FILE: Final = "embedder.json"

# The model's own `max_seq_length`. Card passages are a sentence or two and
# questions are shorter, so this truncates nothing in practice; it is here so
# that the one input long enough to be cut is cut the same way on both sides.
MAX_LENGTH: Final = 512

# One thread. This runs inside a Lambda function sized for an agent loop, where
# the work is one short sequence and the thread pool costs more to start than
# the matrix multiply saves, and inside a laptop's test suite beside everything
# else pytest is doing. ONNX Runtime defaults to one thread per core.
THREADS: Final = 1

ONNX_OUTPUT: Final = "last_hidden_state"


class QueryEmbedderError(RuntimeError):
    """The serving embedder's files are missing, incomplete or not what they claim."""


class Embedder(Protocol):
    """What an index needs from an embedder: a name, a pooling, a width, a matrix.

    `name` and `pooling` are what `meta.json` records and what the loader
    checks, so every implementation has to be able to answer both. `embed` is
    the document side and `embed_query` the query side; they differ for bge,
    which wants an instruction in front of a question and nothing in front of a
    passage.
    """

    @property
    def name(self) -> str:
        """The identifier written into `meta.json` and checked on load."""

    @property
    def pooling(self) -> str:
        """How token vectors become one row: `cls`, `mean`, or `none`."""

    @property
    def dimensions(self) -> int:
        """How wide a vector is."""

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """One L2-normalized row per text, as float32. Documents, not queries."""

    def embed_query(self, text: str) -> np.ndarray:
        """One vector for a search query, in the same space as the documents."""


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """Rows scaled to unit length, so a dot product is a cosine.

    A zero row stays zero rather than becoming a division by zero: a passage
    with no text is a passage that matches nothing, which is the honest answer.
    """
    lengths = np.linalg.norm(matrix, axis=1, keepdims=True)
    lengths[lengths == 0] = 1.0
    normalized: np.ndarray = (matrix / lengths).astype(np.float32)
    return normalized


def query_prefix(model_name: str) -> str:
    """The instruction a question gets in front of it, which only bge wants."""
    return BGE_QUERY_PREFIX if "bge" in model_name.lower() else ""


def write_manifest(directory: Path, *, model: str, pooling: str, dimensions: int) -> Path:
    """Record what the files in a directory are, beside the files.

    Written by `scripts/export_query_embedder.py` and read by `OnnxEmbedder`.
    Separate from the model's own `config.json` because the two facts that
    matter here, which checkpoint this is and how its token vectors are pooled,
    live in two different files in the source repository and in neither of the
    two files that get copied.
    """
    path = directory / MANIFEST_FILE
    path.write_text(
        json.dumps(
            {
                "model": model,
                "pooling": pooling,
                "dimensions": dimensions,
                "max_length": MAX_LENGTH,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


class OnnxEmbedder:
    """The serving embedder: one ONNX graph, one tokenizer, no framework.

    The session and the tokenizer are built on first use rather than in the
    constructor, exactly as the torch one does it, so that making the object is
    free and a process that never embeds anything never reads 133 MB off disk.
    `GET /warm` is what pays for it deliberately, before a question does.

    Not thread safe to construct twice over: the lock is here because the
    application builds the agent lazily and two concurrent first requests would
    otherwise each load a session, which on a function sized for one is the
    whole of the memory. An ONNX Runtime session is itself safe to call from
    several threads, so only the construction is guarded.
    """

    def __init__(self, directory: Path | str | None = None) -> None:
        self._directory = Path(directory) if directory is not None else QUERY_EMBEDDER_DIR
        manifest = self._manifest()
        self._model_name = str(manifest["model"])
        self._pooling = str(manifest["pooling"])
        self._dimensions = int(manifest["dimensions"])
        self._max_length = int(manifest.get("max_length") or MAX_LENGTH)
        if self._pooling != POOLING_CLS:
            raise QueryEmbedderError(
                f"{self._directory / MANIFEST_FILE} says pooling {self._pooling!r}; this "
                f"runs {POOLING_CLS!r} pooling and nothing else"
            )
        self._session: Any | None = None
        self._tokenizer: Any | None = None
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return self._model_name

    @property
    def pooling(self) -> str:
        return self._pooling

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def directory(self) -> Path:
        return self._directory

    def _manifest(self) -> dict[str, Any]:
        """What the directory says it holds, or why it cannot be used.

        Every missing-file case lands here, with the path and the command that
        writes it, because the alternative is ONNX Runtime's own error about a
        protobuf it could not parse.
        """
        for required in (MODEL_FILE, TOKENIZER_FILE, MANIFEST_FILE):
            if not (self._directory / required).is_file():
                raise QueryEmbedderError(
                    f"no query embedder at {self._directory}: {required} is missing. "
                    "Run `uv run python scripts/export_query_embedder.py`, or point "
                    "PRA_QUERY_EMBEDDER_DIR at a directory that has one"
                )
        loaded = json.loads((self._directory / MANIFEST_FILE).read_text(encoding="utf-8"))
        if not isinstance(loaded, dict) or not loaded.get("model"):
            raise QueryEmbedderError(
                f"{self._directory / MANIFEST_FILE} does not name a model; rebuild it with "
                "`uv run python scripts/export_query_embedder.py`"
            )
        return loaded

    def _loaded(self) -> tuple[Any, Any]:
        """The session and the tokenizer, built once."""
        if self._session is None or self._tokenizer is None:
            with self._lock:
                if self._session is None or self._tokenizer is None:
                    import onnxruntime
                    from tokenizers import Tokenizer

                    logger.info(
                        "loading the query embedder",
                        extra={"model": self._model_name, "directory": str(self._directory)},
                    )
                    options = onnxruntime.SessionOptions()
                    options.intra_op_num_threads = THREADS
                    options.inter_op_num_threads = THREADS
                    session = onnxruntime.InferenceSession(
                        str(self._directory / MODEL_FILE),
                        options,
                        providers=["CPUExecutionProvider"],
                    )
                    tokenizer = Tokenizer.from_file(str(self._directory / TOKENIZER_FILE))
                    tokenizer.enable_truncation(max_length=self._max_length)
                    tokenizer.enable_padding()
                    self._session, self._tokenizer = session, tokenizer
        return self._session, self._tokenizer

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """One unit-norm float32 row per text, CLS-pooled out of the last layer.

        An empty batch returns an empty matrix of the right width rather than
        calling the session, because ONNX Runtime refuses a zero-length batch
        and "nothing to embed" is not an error.
        """
        if not texts:
            return np.zeros((0, self._dimensions), dtype=np.float32)
        session, tokenizer = self._loaded()
        encoded = tokenizer.encode_batch(list(texts))
        feeds = {
            "input_ids": np.asarray([item.ids for item in encoded], dtype=np.int64),
            "attention_mask": np.asarray([item.attention_mask for item in encoded], dtype=np.int64),
            "token_type_ids": np.asarray([item.type_ids for item in encoded], dtype=np.int64),
        }
        names = {item.name for item in session.get_inputs()}
        hidden = session.run([ONNX_OUTPUT], {k: v for k, v in feeds.items() if k in names})[0]
        return normalize_rows(np.asarray(hidden[:, 0, :], dtype=np.float32))

    def embed_query(self, text: str) -> np.ndarray:
        """The query with its model's instruction prefix, when its model wants one."""
        row: np.ndarray = self.embed([query_prefix(self._model_name) + text])[0]
        return row


class SentenceTransformerEmbedder:
    """`sentence-transformers` over a small local model. The one indexes are built with.

    The model is loaded on first use rather than in the constructor, so
    building the object costs nothing and a process that ends up not embedding
    anything never pays for torch.

    Nothing that serves a question constructs this any more; `OnnxEmbedder` is
    the query side everywhere. It is still the document side, and it is still
    the definition of what the index's vectors are: the ONNX graph is an export
    of this model and the equivalence test is what says so.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self._model_name = model_name
        self._model: Any | None = None
        self._dimensions = 0

    @property
    def name(self) -> str:
        return self._model_name

    @property
    def pooling(self) -> str:
        """What the loaded model's pooling module reports, not what its name suggests.

        A `SentenceTransformer` is a sequence of modules and one of them is the
        `Pooling`, which holds its own configuration: `cls` for bge, `mean` for
        MiniLM. Reading it means an index records the real answer for whichever
        model built it rather than a guess from a table in this file. Two
        spellings because the attribute was a `get_pooling_mode_str()` method
        before `sentence-transformers` 4 and a `pooling_mode` string after, and
        the floor this project declares is 3.

        It loads the model, which every caller of this class has already done
        or is about to: a build embeds the corpus in the next breath.
        """
        for module in self._loaded():
            mode = getattr(module, "pooling_mode", None)
            if isinstance(mode, str):
                return mode
            reported = getattr(module, "get_pooling_mode_str", None)
            if callable(reported):
                return str(reported())
        return POOLING_NONE

    @property
    def dimensions(self) -> int:
        if not self._dimensions:
            self._dimensions = int(self._loaded().get_sentence_embedding_dimension())
        return self._dimensions

    def _loaded(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("loading the embedding model", extra={"model": self._model_name})
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def embed_query(self, text: str) -> np.ndarray:
        """The query with its model's instruction prefix, when its model wants one."""
        row: np.ndarray = self.embed([query_prefix(self._model_name) + text])[0]
        return row

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self._loaded().encode(
            list(texts),
            batch_size=64,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)
