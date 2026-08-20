# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] - 2026-08-20

First release.

### Added

- `plan`, `migrate`, `verify`, `seed`, `demo` and `version` commands; every
  command exits 0 on success and 1 on failure.
- Migration of config (as params), full-fidelity history via `scan_history`
  (as metric series with original steps), summaries (as `final.*` metrics and
  `summary.*` params), tags, notes, groups, job types, and sweep parent/child
  nesting.
- Opt-in byte-costing flags: `--files`, `--artifacts`, `--system-metrics`,
  bounded by `--max-artifact-size`.
- `--artifact-root` to place artifact bytes explicitly, because MLflow derives
  the default from the working directory rather than the tracking URI.
- Server-side idempotency: re-running a migration skips completed runs,
  replaces half-written ones, and `--overwrite` soft-deletes rather than
  destroys.
- A live self-test (`demo`): seeds a real W&B project of hostile runs, migrates
  it, and verifies against a manifest recorded from what was actually logged.
- An enforced read-only guarantee towards W&B: the migration path calls six
  read APIs and nothing else, asserted by AST-level tests.
- [MAPPING.md](MAPPING.md) as the loss contract, with every drop counted and
  written to the `wandb.dropped` tag.

### Fixed

- `--system-metrics` never worked: it called `scan_history(stream="events")`,
  a parameter that has never existed in any released wandb, failing every run
  in the migration. The stream is now read via `history(stream="system")`,
  and a contract-test tier binds the adapter's calls against the installed
  wandb's real signatures so a phantom argument can never ship again.
- System-metric keys arrive from W&B already prefixed `system.`; the migrator
  no longer doubles the prefix to `system.system.*`.
- Live `verify` now accepts the same opt-in flags as `migrate`, so verifying a
  `--system-metrics true` migration no longer reports every correct `system.*`
  series as unexpected.

[Unreleased]: https://github.com/Nailujj/wandb-to-mlflow/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Nailujj/wandb-to-mlflow/releases/tag/v0.1.0
