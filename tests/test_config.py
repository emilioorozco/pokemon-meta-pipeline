"""SOURCE_DATA_DIR is required, but only when the source is actually used."""
import pytest

from pipeline import config


def test_unset_source_is_a_clear_error(monkeypatch):
    monkeypatch.delenv("SOURCE_DATA_DIR", raising=False)
    with pytest.raises(config.SourceNotConfigured, match="SOURCE_DATA_DIR is not set"):
        config.source_data_dir()
    problems = config.validate_source()
    assert len(problems) == 1
    assert problems[0].startswith("SOURCE_DATA_DIR is not set")


def test_blank_source_counts_as_unset(monkeypatch):
    monkeypatch.setenv("SOURCE_DATA_DIR", "   ")
    with pytest.raises(config.SourceNotConfigured):
        config.replay_batches()


def test_missing_paths_are_reported(monkeypatch, tmp_path):
    monkeypatch.setenv("SOURCE_DATA_DIR", str(tmp_path))
    problems = config.validate_source()
    assert len(problems) == 3
    assert all(str(tmp_path) in p for p in problems)
