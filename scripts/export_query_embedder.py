"""Put the serving query embedder's files where `pipeline.query_embedder` looks.

    uv run python scripts/export_query_embedder.py
    uv run python scripts/export_query_embedder.py --out /opt/embedder

Three files: `model.onnx`, `tokenizer.json` and an `embedder.json` naming the
model and its pooling. `Dockerfile.agent` runs this in a build stage and the
runtime image gets the directory and none of this script's dependencies; a
laptop runs it once and `uv run python -m pipeline.card_index query` works.

**Nothing is exported, in the usual sense.** `BAAI/bge-small-en-v1.5` publishes
`onnx/model.onnx` in its own Hugging Face repository, 133,093,490 bytes, fp32
and in one file, so the honest thing is to download it rather than to run
`optimum` over the checkpoint and produce a graph that is supposed to be the
same. A model that ships no ONNX is the case `--export` is for: it converts
with `optimum`, which is a build-time dependency of this script and of nothing
else, and writes the same three files. Either way the result is a directory,
and `pipeline.query_embedder` cannot tell which produced it.

**The pooling is read, not assumed.** `1_Pooling/config.json` in the model's
repository says which mode the sentence-transformers model uses, and that is
what goes into `embedder.json` and what `CardIndex.load` checks against the
index. A model whose pooling this runtime does not implement fails here, with
its name and its mode, rather than at the first question with a plausible
ranking of the wrong cards.

No network at run time anywhere else: this is the one place that reaches for
Hugging Face, it is never on a request path, and the fast test suite does not
call it.
"""

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any, Final

from pipeline.config import QUERY_EMBEDDER_DIR
from pipeline.observability import configure_logging, emit_summary
from pipeline.query_embedder import (
    DEFAULT_MODEL,
    MODEL_FILE,
    POOLING_CLS,
    TOKENIZER_FILE,
    write_manifest,
)

logger = logging.getLogger(__name__)

STAGE: Final = "export_query_embedder"

# Where a published graph sits inside a model repository, and what the two
# halves of an external-data pair are called when there is one. bge-small has
# no external data; a larger model would, and ONNX Runtime finds it by name
# beside the graph, so it is copied with the same name or not at all.
REPO_MODEL_FILE: Final = "onnx/model.onnx"
EXTERNAL_DATA_FILES: Final = ("onnx/model.onnx_data", "onnx/model.onnx.data")
POOLING_CONFIG_FILE: Final = "1_Pooling/config.json"

# How `1_Pooling/config.json` spells each mode, and what this project calls it.
POOLING_FLAGS: Final = {
    "pooling_mode_cls_token": POOLING_CLS,
    "pooling_mode_mean_tokens": "mean",
    "pooling_mode_max_tokens": "max",
    "pooling_mode_mean_sqrt_len_tokens": "mean_sqrt_len_tokens",
}


class ExportError(RuntimeError):
    """The model cannot be turned into the three files the serving side needs."""


def pooling_mode(config: dict[str, Any]) -> str:
    """The one pooling mode a `1_Pooling/config.json` turns on.

    Exactly one, because a model that pools two ways at once concatenates them
    and is a different width than its hidden size, which is not something this
    runtime implements and not something a silent guess should paper over.
    """
    modes = [name for flag, name in POOLING_FLAGS.items() if config.get(flag)]
    if len(modes) != 1:
        raise ExportError(f"{POOLING_CONFIG_FILE} turns on {modes or 'no'} pooling; expected one")
    return modes[0]


def fetch(model: str, out: Path, *, export: bool) -> dict[str, Any]:
    """Write the three files for `model` into `out`. Returns what was written."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    out.mkdir(parents=True, exist_ok=True)
    pooling_path = hf_hub_download(model, POOLING_CONFIG_FILE)
    pooling_config = json.loads(Path(pooling_path).read_text(encoding="utf-8"))
    pooling = pooling_mode(pooling_config)
    dimensions = int(pooling_config["word_embedding_dimension"])
    if pooling != POOLING_CLS:
        raise ExportError(
            f"{model} pools {pooling!r}; pipeline.query_embedder runs {POOLING_CLS!r} pooling "
            "and nothing else, so adding this model means adding its pooling there first"
        )

    shutil.copyfile(hf_hub_download(model, TOKENIZER_FILE), out / TOKENIZER_FILE)
    if export:
        _export_with_optimum(model, out)
    else:
        shutil.copyfile(hf_hub_download(model, REPO_MODEL_FILE), out / MODEL_FILE)
        for sidecar in EXTERNAL_DATA_FILES:
            try:
                downloaded = hf_hub_download(model, sidecar)
            except EntryNotFoundError:
                continue
            shutil.copyfile(downloaded, out / Path(sidecar).name)
    write_manifest(out, model=model, pooling=pooling, dimensions=dimensions)
    return {
        "model": model,
        "pooling": pooling,
        "dimensions": dimensions,
        "out": str(out),
        "model_bytes": (out / MODEL_FILE).stat().st_size,
        "exported": export,
    }


def _export_with_optimum(model: str, out: Path) -> None:
    """Convert a checkpoint that publishes no graph, with `optimum`.

    Imported here because it is not installed anywhere this project deploys
    and barely anywhere it develops: the one model in use ships its own ONNX,
    and the import failing is the right error for a flag nobody has needed.
    """
    from optimum.onnxruntime import ORTModelForFeatureExtraction

    converted = ORTModelForFeatureExtraction.from_pretrained(model, export=True)
    converted.save_pretrained(out / "_optimum")
    for produced in sorted((out / "_optimum").glob("*.onnx*")):
        shutil.copyfile(produced, out / produced.name.replace("model.onnx", MODEL_FILE))
    shutil.rmtree(out / "_optimum")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python scripts/export_query_embedder.py",
        description="Download or export the ONNX graph and tokenizer the serving side runs.",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, metavar="NAME", help=f"(default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=QUERY_EMBEDDER_DIR,
        metavar="PATH",
        help=f"where to write the three files (default: {QUERY_EMBEDDER_DIR})",
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="convert with optimum instead of downloading a published graph",
    )
    args = parser.parse_args(argv)
    configure_logging(STAGE)
    try:
        written = fetch(args.model, args.out, export=args.export)
    except ExportError as failure:
        logger.error("the query embedder could not be written", extra={"error": str(failure)})
        return 1
    emit_summary(
        logger,
        "query embedder written",
        written,
        text=(
            f"wrote {written['model']} ({written['pooling']} pooling, "
            f"{written['dimensions']} dimensions) to {written['out']}"
        ),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
