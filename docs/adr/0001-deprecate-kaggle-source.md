# 0001. Deprecate the Kaggle corpus source

Date: 2026-09-21

## Status

Accepted.

## Context

The first stage 1 of this pipeline (`ingest` and `enrich`, written 2026-08-31)
read a corpus of about 1,240 AI-versus-AI replays from a Kaggle competition
that has since finished. That corpus is not redistributable, so no one can
reproduce the run from the repository alone, and its metadata endpoint may stop
answering at any time.

The project's purpose has changed. It is now the analytics data platform for
Play Rough Analytics, a web application where players upload real Pokemon
Trading Card Game (TCG) Live battle logs. Its parsed-game blobs in Amazon
Simple Storage Service (S3) are the source (see `docs/discovery.md`), and
every downstream stage in `docs/stages.md` is designed against them.

Keeping two live sources would tell two stories: two ingest paths, two
schemas, and a reader unable to tell which one the warehouse is built on.

## Decision

Deprecate the Kaggle stage without erasing it.

- The code moves to `pipeline/legacy/kaggle/` (history preserved with
  `git mv`) and its source config moves with it, out of `pipeline/config.py`.
- Importing `pipeline.legacy.kaggle` emits a `DeprecationWarning`.
- Its tests still run, under `tests/legacy/`, so the code keeps working.
- The package is excluded from coverage and is not wired into any stage,
  DAG or command-line interface. The live bronze stage is written fresh.
- Delete it only if it blocks something (a dependency upgrade, a rename, a
  tooling change). Until then it stays.

## Consequences

- The history is honest: the repository shows what was built first and why it
  was set aside, instead of a rewritten past.
- The patterns worth keeping stay readable as reference: the idempotent bronze
  writer with hive partitioning, quarantine of games that violate the
  contract, and landing the whole upstream response before parsing it.
- Slight repository weight: two modules, one config helper and four test files
  that no stage uses. Tests still cost a few seconds per CI run.
- A reader must not mistake it for the live path. The `legacy` package name,
  the import-time warning and this record are the guard rails; the README
  points here rather than describing the old source.
