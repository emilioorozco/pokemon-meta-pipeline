"""SOURCE_DATA_DIR is required, but only when the source is actually used.

Also pins the deprecation contract: importing the legacy package warns."""

import importlib
from pathlib import Path

import pytest

import pipeline.legacy.kaggle
from pipeline.legacy.kaggle import config

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def test_importing_legacy_kaggle_warns() -> None:
    # The package is already imported by the time tests run, so re-execute it.
    with pytest.warns(DeprecationWarning, match="docs/adr/0001-deprecate-kaggle-source.md"):
        importlib.reload(pipeline.legacy.kaggle)


def test_unset_source_is_a_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SOURCE_DATA_DIR", raising=False)
    with pytest.raises(config.SourceNotConfigured, match="SOURCE_DATA_DIR is not set"):
        config.source_data_dir()
    problems = config.validate_source()
    assert len(problems) == 1
    assert problems[0].startswith("SOURCE_DATA_DIR is not set")


def test_blank_source_counts_as_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SOURCE_DATA_DIR", "   ")
    with pytest.raises(config.SourceNotConfigured):
        config.replay_batches()


def test_missing_paths_are_reported(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SOURCE_DATA_DIR", str(tmp_path))
    problems = config.validate_source()
    assert len(problems) == 3
    assert all(str(tmp_path) in p for p in problems)
