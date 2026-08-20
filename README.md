# wandb-to-mlflow

**Migrate, export, and back up your Weights & Biases (W&B / wandb) experiments
to MLflow** — metrics, configs, artifacts, media files, and sweeps, into any
MLflow tracking server you control.

[![CI](https://github.com/Nailujj/wandb-to-mlflow/actions/workflows/ci.yml/badge.svg)](https://github.com/Nailujj/wandb-to-mlflow/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/wandb-to-mlflow)](https://pypi.org/project/wandb-to-mlflow/)
[![Python](https://img.shields.io/pypi/pyversions/wandb-to-mlflow)](https://pypi.org/project/wandb-to-mlflow/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

Point it at a W&B project, tell it where your MLflow store is, and it copies the
runs across: config as params, full training history as metric series with their
original step numbers, summaries as `final.*` metrics, tags, groups, sweep
nesting, run files and artifact bytes. It **reads from W&B and never writes to
it**, so you can migrate, check the result, and decide about your subscription
afterwards. Works with self-hosted MLflow, a local SQLite store, or any remote
tracking server — anywhere you want your experiment data to live after leaving
wandb.ai.

**New here? Start with [Quickstart](#quickstart).**
**About to cancel your W&B subscription? Read [What survives](#what-survives-and-what-does-not) first.**

- [Requirements](#requirements)
- [Install](#install)
- [Quickstart](#quickstart)
- [Authenticating with W&B](#authenticating-with-wb)
- [Choosing where MLflow puts things](#choosing-where-mlflow-puts-things)
- [Commands](#commands)
- [What survives, and what does not](#what-survives-and-what-does-not)
- [Your W&B data is never touched](#your-wb-data-is-never-touched)
- [Troubleshooting](#troubleshooting)
- [Known limitations](#known-limitations)
- [It tests itself](#it-tests-itself)
- [As an MLflow Project](#as-an-mlflow-project)
- [UI acceptance checklist](#ui-acceptance-checklist)
- [FAQ](#faq)
- [Development](#development)

---

## Requirements

- **Python 3.10 or newer.**
- **A W&B API key** with read access to the projects you want to migrate. Get one
  from <https://wandb.ai/authorize>.
- **An MLflow tracking store.** SQLite is the easiest and is what this guide
  uses. A remote tracking server works too — anything `MlflowClient` accepts.
  A plain `./mlruns` directory does **not**; see
  [Troubleshooting](#the-filesystem-tracking-backend-is-in-maintenance-mode).
- [`uv`](https://docs.astral.sh/uv/) for the commands below, though `pip` works
  just as well.

## Install

From PyPI:

```bash
pip install wandb-to-mlflow
```

Or from source:

```bash
git clone https://github.com/Nailujj/wandb-to-mlflow.git
cd wandb-to-mlflow
uv venv
uv pip install -e .
```

Then either activate the environment (`source .venv/bin/activate`) so that
`wandb-to-mlflow` is on your `PATH`, or prefix every command with `uv run`. This
guide assumes the former. `python -m wandb_to_mlflow` works identically if you
prefer not to rely on the console script.

Add the test and lint tooling with `uv pip install -e ".[dev]"`.

## Quickstart

Five minutes, start to finish. Nothing here writes to W&B.

**1. Authenticate.** Set the key for the account that owns the project:

```bash
export WANDB_API_KEY=...
```

**2. Confirm you are who you think you are.** This is the single most common
source of confusion — see [Authenticating with W&B](#authenticating-with-wb):

```bash
python -c "import wandb; print(wandb.Api().default_entity)"
```

The entity it prints is the one you pass as `--entity`. It is also the first path
segment of your project's URL: `wandb.ai/<entity>/<project>`.

**3. Choose the MLflow destination.** SQLite in the current directory:

```bash
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db
```

**4. See what would happen.** `plan` writes nothing at all:

```bash
wandb-to-mlflow plan --entity my-team --project my-project
```

```
Target experiment: my-project
Runs to migrate:   6
Params:            94
Metric series:     123
Metric points:     1176

Will NOT be migrated (see MAPPING.md):
  media        10
    media image-file           5
    media table-file           5

Sparse-logging nulls: 224 (NOT data loss -- W&B pads rows for keys that were not logged at that step)

Keys renamed for MLflow:  0
Param values truncated:   0
Sweeps (become parents):  0
Artifacts:                skipped (pass --artifacts true)
Run files:                skipped (pass --files true)
System metrics:           skipped (pass --system-metrics true)

Nothing was written. This was a plan.
```

**5. Migrate.** Include the bytes, and say where they go:

```bash
wandb-to-mlflow migrate --entity my-team --project my-project \
    --files true --artifacts true \
    --artifact-root ./mlflow-artifacts
```

A live progress bar tracks the runs as they land. It draws on stderr and clears
itself when done, so piped output and `--json` stay untouched.

**6. Check it.** Live verification re-derives what the migration should have
produced and compares it against what is actually in MLflow. **Pass the same
opt-in flags you migrated with**, or correctly-migrated data is reported as
unexpected:

```bash
wandb-to-mlflow verify --entity my-team --project my-project \
    --experiment my-project --files true --artifacts true
```

```
No unexpected loss. Every difference is one MAPPING.md documents.
```

**7. Look at it.**

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Re-running `migrate` is free and safe: runs already migrated are skipped, not
duplicated. A migration killed halfway leaves a run that the next attempt
replaces rather than trusts. `--overwrite` re-migrates everything, soft-deleting
what was there rather than destroying it.

## Authenticating with W&B

The tool never asks for credentials itself. It uses whatever the `wandb` library
finds, in this order:

1. `WANDB_API_KEY` in the environment.
2. `~/.netrc`, which is what `wandb login` writes.

**If you have ever run `wandb login` for a different account, `~/.netrc` holds
that account's key** — and every command will silently run as that user. Projects
belonging to your other account then appear not to exist at all:

```
ValueError: Could not find project my-project
```

The first line of output tells you which source was used:

```
wandb: [wandb.Api()] Loaded credentials for https://api.wandb.ai from WANDB_API_KEY
wandb: [wandb.Api()] Loaded credentials for https://api.wandb.ai from /home/you/.netrc
```

`WANDB_API_KEY` takes precedence, so exporting it is the reliable fix. Use
`wandb login` only if you want to overwrite the stored key permanently.

## Choosing where MLflow puts things

Two MLflow behaviours surprise nearly everyone, and neither is this tool's doing.

### Use a SQL backend, not a bare directory

MLflow 3 puts the filesystem tracking store (`./mlruns`) in maintenance mode. It
refuses to open without `MLFLOW_ALLOW_FILE_STORE=true`, and even with that set
its own UI fails against it: endpoints the run table depends on return 500 and
the experiment reads as empty although every run migrated correctly. Point the
migration at SQLite from the start:

```bash
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db
```

### `--tracking-uri` does not decide where artifact bytes go

An experiment's artifact root defaults to `./mlruns/<experiment_id>`, resolved
against the **current working directory** — it is not derived from the tracking
URI. Two tracking databases used from the same directory therefore share one
artifact tree, keyed by an experiment id each assigns independently. The
artifacts interleave, and deleting an experiment or running `mlflow gc` against
one database can remove bytes belonging to the other.

So whenever the tracking store is not the default one in this directory, say
where the bytes go:

```bash
wandb-to-mlflow migrate -e my-team -p my-project \
  --tracking-uri sqlite:///mlflow.db \
  --artifact-root /data/mlflow-artifacts \
  --files true --artifacts true
```

MLflow records the artifact location when the experiment is **created** and
ignores it afterwards, so this cannot relocate an experiment that already exists.
Passing it for one logs a warning rather than pretending otherwise.

## Commands

| Command | What it does | Writes to MLflow? |
|---|---|---|
| `plan` | Reports what a migration would do. | No |
| `migrate` | Copies a W&B project into an MLflow experiment. | Yes |
| `verify` | Checks a migration, against live W&B or a seeded manifest. | No |
| `seed` | Creates a disposable W&B project of hostile runs, for testing. | No (writes to **W&B**) |
| `demo` | seed → migrate → verify in one go. | Yes |
| `version` | Prints the version. | No |

Every command exits **0** on success and **1** on failure, so they compose in
scripts and CI.

### Shared options

`--entity` / `-e` and `--project` / `-p` identify the W&B side; both are options,
not positional arguments. `--experiment` names the MLflow target and defaults to
the W&B project name. `--tracking-uri` overrides `MLFLOW_TRACKING_URI`. `--json`
emits machine-readable output on stdout. `--verbose` / `-v` turns on debug
logging, which goes to stderr so it never contaminates `--json`.

### Options that cost bytes are opt-in

`--artifacts`, `--files` and `--system-metrics` all default to `false`, and take
an explicit value (`--artifacts true`) rather than being bare flags — `MLproject`
entry points substitute parameters positionally and cannot omit a flag
conditionally.

| Flag | What it adds |
|---|---|
| `--files true` | Everything under `run.files()`, as artifacts beneath `wandb_files/`. This is what brings media **files** across. |
| `--artifacts true` | Logged artifact bytes, beneath `artifacts/<name>/`. Includes W&B's automatic per-run history parquet. |
| `--system-metrics true` | CPU/GPU/memory/network series, as `system.*` metrics. Server-sampled by W&B; no exhaustive reader exists. |
| `--max-artifact-size` | Size ceiling per artifact, default `100MB`. Accepts `512`, `20MB`, `1.5GiB`. Anything larger is skipped, counted and listed. |

### `migrate` only

`--overwrite` re-migrates runs that are already there. `--workers N` migrates N
runs concurrently — a large project with many small runs benefits most.
`--artifact-root` is described [above](#--tracking-uri-does-not-decide-where-artifact-bytes-go).

### `verify` only

Two modes. Against **live W&B** (`--entity` and `--project`), it re-plans the
migration to derive expectations. Against a **manifest** (`--manifest`), it uses
ground truth the seeder recorded at seed time.

The manifest mode is the stronger test. Live mode compares the migration against
the same logic that produced it, so it proves the migration matches what the tool
would do today — not that the tool is right. Manifest mode is what the self-test
uses, and it is the only mode that genuinely tests the read side.

Live mode needs the **same opt-in flags the migration ran with**. Verifying a
`--system-metrics true` migration without the flag reports every correct
`system.*` series as an unexpected extra metric.

## What survives, and what does not

Some things **do not survive the migration**. Not "degrade" — are not migrated at
all. If any of these matter to you, export them separately **before** you lose
access:

| Not migrated | What that means |
|---|---|
| **Media and table *panels*** | `wandb.Image`, `wandb.Table`, `wandb.Audio`, `wandb.Video` and plots cannot become MLflow metrics and will not render as charts or panels. **The underlying files do come across** if you pass `--files true` — they land as artifacts under `wandb_files/media/`. So you lose the visualisations and the step-linked association, not the pixels. Counted and reported per type either way. |
| **Reports** | Not migrated. Export them from W&B first. |
| **Workspace panels, custom charts, layouts** | Not migrated. MLflow has no equivalent. |
| **Sweep configuration** | The sweep's search space, method, metric goal and early-terminate rules are not migrated. Only the sweep *id* and the parent/child structure come across. |
| **Model registry, registered models, artifact aliases and versions** | Not migrated. Artifact **bytes** can be copied (`--artifacts true`); their versions, aliases, lineage and metadata cannot. |
| **Reference artifacts** | Artifacts whose bytes live in S3/GCS/HTTP are recorded as URIs. This tool will not reach into your buckets on your behalf. |
| **Artifact lineage graph** | Not migrated. |
| **Run comments, teams, permissions, launch queues, automations** | Not migrated. |

Things that survive but **change shape**:

- `crashed` and `failed` are distinct in W&B; both become MLflow `FAILED`. The
  original string is kept in the `wandb.state` tag.
- Run **end times are approximate**. W&B's public API exposes no true end time,
  so it is derived from the last history timestamp. Which source was used is
  recorded per run in the `wandb.end_time_source` tag.
- Long config values are **truncated** at MLflow's param limit, with the
  affected keys listed in the `wandb.truncated_params` tag.
- Metric keys illegal in MLflow are **renamed**; the originals are kept in the
  `wandb.renamed_keys` tag. (`train/loss` and `héllo` are already legal and are
  left alone.)
- `NaN`, `±inf`, strings, lists and **booleans** are not logged as metrics.
  Booleans especially: `True` is an `int` in Python, and logging it as `1.0`
  would invent data that was never measured.
- **System metrics are sampled.** W&B samples them server-side and offers no
  exhaustive reader, so `--system-metrics true` migrates the sampled series.

### Two things that survive better than you might expect

Pass `--files true --artifacts true` and you also get:

- **The media files themselves**, under `wandb_files/media/`. Not as MLflow
  image panels, but the bytes are there.
- **`artifacts/run-<id>-history_v0/0000.parquet`** — W&B logs a history artifact
  for every run automatically, and it is the complete raw history at full
  fidelity: every media reference, `NaN`, bool and string that could not become
  an MLflow metric, with original step numbers. A few KB per run, and the
  closest thing to a lossless escape hatch this migration has.

One caveat, learned live: W&B finalises some artifacts **asynchronously** after a
run ends, so migrating seconds after a run finishes can miss them. Re-run with
`--overwrite` to pick them up.

**[MAPPING.md](MAPPING.md) is the full contract**, and it is what the test suite
asserts against. Every drop above is counted, surfaced in the CLI report, and
written to the `wandb.dropped` tag on the migrated run.

## Your W&B data is never touched

Migrating **reads** from W&B and writes to MLflow. That is the whole of it. The
migration path calls exactly six things on the W&B API — `runs()`,
`scan_history()`, `history()`, `files()`, `logged_artifacts()` and `download()` —
all of them reads. Nothing in `plan`, `migrate` or `verify` can create, modify or
delete a W&B run, artifact or project.

Keep W&B as long as you like. Migrate, run `verify`, look at the result in
`mlflow ui`, and decide afterwards. If the migration is wrong, the originals are
still there — which is the point.

This is enforced, not merely intended:

- A test parses the source adapter's AST and fails the build if it calls any
  mutating W&B method.
- Another asserts the package contains **exactly one** `delete` call anywhere,
  and that it is in the seeder.
- That one call is guarded *at the point of deletion* — not just in the CLI — and
  refuses any project whose name does not start with `w2m-selftest-`, the prefix
  only this tool's own self-test projects carry. A library caller cannot bypass
  it either.

The only thing `seed --cleanup` and the `demo` output ever offer to delete is the
disposable self-test project the tool created minutes earlier. It will refuse to
touch anything else, and it never deletes anything without being asked.

## Troubleshooting

### `Missing option '--project' / '-p'`

`--entity` and `--project` are options, not positional arguments:

```bash
wandb-to-mlflow plan --entity my-team --project my-project   # not: plan my-project
```

### The filesystem tracking backend is in maintenance mode

```
MlflowException: The filesystem tracking backend (e.g., './mlruns') is in
maintenance mode and will not receive further updates.
```

No tracking URI was set, so MLflow fell back to `./mlruns`. Set one:

```bash
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db
```

`MLFLOW_ALLOW_FILE_STORE=true` also silences it, but opts you into a backend
whose UI does not work — see
[Choosing where MLflow puts things](#choosing-where-mlflow-puts-things).

### `ValueError: Could not find project X`

W&B returns the same error for three different causes, and the message does not
distinguish them:

1. **The project name is wrong.** Check it against the URL.
2. **The entity is wrong.** It is the URL slug, not your display name.
3. **You are authenticated as a different account**, and cannot see it. This is
   the most common and the least obvious — see
   [Authenticating with W&B](#authenticating-with-wb).

Confirm all three at once:

```bash
python -c "import wandb; api=wandb.Api(); print(api.default_entity, [p.name for p in api.projects()])"
```

### `verify` reports metrics "present that should not be"

The migration used an opt-in flag that the verification did not. Pass `verify`
the same `--files` / `--artifacts` / `--system-metrics` values you migrated with.

### `mlflow run .` keeps running the old code after an edit

That is MLflow's environment cache, not your edit. It keys the environment on
`python_env.yaml`, so editing the package does not invalidate it and the stale
wheel stays installed. Clear it with `rm -rf ~/.mlflow/envs/*`, or iterate with
`--env-manager local`.

### `pytest` crashes inside unrelated plugins

Some system installs put their own pytest plugins on `PYTHONPATH` — ROS 2 is a
common culprit, and fails with `ModuleNotFoundError: No module named 'lark'`
before any test runs. Run `env -u PYTHONPATH uv run pytest`, or
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest`.

## Known limitations

Stated plainly, because finding these yourself mid-migration is worse:

- **`plan` does not enumerate files or artifacts.** It reports metrics, params
  and drops only. Passing `--files true --artifacts true` to `plan` changes
  nothing in its output, so it cannot tell you how many bytes a migration will
  download or whether anything exceeds `--max-artifact-size`. You find that out
  during `migrate`.
- **Live `verify` does not check param values.** It checks the param *count*, so
  a wrong value with the right key is not detected. Manifest mode does check
  values.
- **Neither `verify` mode checks intermediate metric values.** It checks point
  counts per series and the `final.*` values. A migration that corrupted values
  mid-series while preserving counts and endpoints would pass.
- **System metrics are sampled**, not exhaustive. W&B offers no scanning reader
  for that stream.

## It tests itself

```bash
wandb-to-mlflow demo --entity my-team
```

This seeds a **real** W&B project with twenty deliberately hostile runs —
seventeen standalone plus a three-child sweep —
20,000-step histories, `NaN`s, booleans, colliding metric keys, 20,000-character
config values, media, a sweep, emoji — migrates it, and verifies the result
against a manifest the seeder wrote from what it actually logged. Exit code 0,
or a precise diff.

The manifest matters: verifying against a second query to W&B would only prove
the tool is self-consistent with itself. Ground truth recorded at seed time is
the only thing that tests the read side.

`demo` prints the cleanup command it wants you to run afterwards. The seeded
project is named `w2m-selftest-<utc-timestamp>`, and `seed --cleanup` refuses to
touch anything not named that way. W&B's public API has no project delete, so the
emptied project shell stays until you remove it from the web UI.

## As an MLflow Project

Every entry point works through `mlflow run`:

```bash
mlflow run . -e demo    -P entity=my-team
mlflow run . -e plan    -P entity=my-team -P project=my-project
mlflow run . -e migrate -P entity=my-team -P project=my-project -P artifacts=true
mlflow run . -e seed    -P entity=my-team
mlflow run . -e verify  -P manifest=manifest.json -P experiment=my-project
```

`mlflow run` opens its own MLflow run for the entry point. The migrator uses
`MlflowClient` exclusively — never the fluent API — so migrated runs never nest
inside it or pollute the target experiment. A test asserts this by running a
migration inside an active ambient run, and another test fails the build on any
fluent-API call anywhere in the package.

## UI acceptance checklist

Automated tests cannot catch a broken step axis. After `demo`, open the UI:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

and check by eye:

- [ ] Runs are sorted by their **original** W&B start times, not by when the
      migration ran. (Sort by "Created" — the spread should span the seeding
      session, and `wandb.end_time_source` should be visible as a tag.)
- [ ] The runs table shows `final.*` columns, and sorting by `final.accuracy`
      orders the sweep children correctly.
- [ ] The `many-steps` run's chart renders without timing out, and its x-axis
      runs to 20,000 (or whatever `--steps` you passed), not to ~500. A ceiling
      near 500 means something fell back to `run.history()`.
- [ ] `sparse-logging` shows `dense` with 100 points and `sparse` with 10, each
      on its own step positions — not resampled onto a shared axis.
- [ ] The three `sweep-child-*` runs are **nested** under one `sweep-<id>`
      parent row, collapsible as a group.
- [ ] `same-name` appears **twice**, as two separate runs.
- [ ] `ünïcode 실험` renders correctly in the run list, and its notes show the
      Cyrillic and CJK text.
- [ ] `bool-trap` has **no** `improved` metric chart. If one exists, booleans are
      being logged as numbers.
- [ ] `nonfinite-metrics` shows `loss` with a single point and no gaps, spikes or
      broken axes from `inf`.
- [ ] `hostile-keys` shows `train/loss` and `héllo` under their **original**
      names, and `x_y_` renamed. Its `wandb.renamed_keys` tag lists the mapping.
- [ ] `media-and-tables` has a `wandb.dropped` tag reading
      `{"media": 3, "media_types": {"image-file": 2, "table-file": 1}}`.
- [ ] `crashed-run` and `failed-run` show status **FAILED**.
- [ ] `with-artifacts` has an `artifacts/small-dataset*/` folder containing
      `data.csv` and `_wandb_artifact.json`, and a `wandb.reference_artifacts`
      tag whose bytes were **not** fetched.

## FAQ

### How do I export my data from Weights & Biases before cancelling?

Run this tool while your W&B subscription (or free-tier access) is still
active — it needs read access to the API. `plan` first to see exactly what
will and will not come across, then `migrate` with `--files true --artifacts
true` for the fullest copy, then `verify`. Export W&B **Reports** separately
from the web UI; they are the one thing with no API to read from
([details](#what-survives-and-what-does-not)).

### Is this a W&B to MLflow converter, or a sync tool?

A one-way, re-runnable copier. It converts W&B runs into MLflow's data model
in a single direction; re-running picks up runs that are new or previously
failed and skips everything already migrated. It is not a live two-way sync,
and it never writes to W&B.

### Can it delete or corrupt my W&B data?

No. The migration path calls six read-only W&B APIs and nothing else, and the
test suite enforces that at the AST level — see
[Your W&B data is never touched](#your-wb-data-is-never-touched).

### Does it work with a self-hosted MLflow server?

Yes — anything `MlflowClient` accepts as a tracking URI: a local SQLite file,
`http(s)://` tracking servers, or a database URI. Pass `--artifact-root` so
artifact bytes land where you expect
([why](#--tracking-uri-does-not-decide-where-artifact-bytes-go)).

### Are sweeps migrated?

The runs and their parent/child structure, yes: each sweep becomes a parent
MLflow run with its children nested beneath it. The sweep's search-space
configuration is not migrated ([full list](#what-survives-and-what-does-not)).

### How long does a migration take?

Metrics-only migration of a typical project is seconds to minutes; `--files`
and `--artifacts` add download time for the bytes. `--workers N` migrates N
runs in parallel, and a progress bar shows where you are.

## Development

```bash
uv pip install -e ".[dev]"

uv run pytest                  # tiers 1 and 2: no network, a few seconds
uv run mypy --strict src/
uv run ruff check .
uv run ruff format --check .

# Tier 3: real W&B and real MLflow. Creates and then deletes a scratch project.
W2M_E2E=1 W2M_E2E_ENTITY=my-team WANDB_API_KEY=... uv run pytest -m e2e
```

The suite is in three tiers:

1. **Pure functions** — coercion, key sanitisation, limits. `coerce.py` is held
   at 100% branch coverage by CI.
2. **Fake sources into a real MLflow store.** No network, but a real backend, so
   assertions read the store back rather than trusting the migrator's own
   bookkeeping.
3. **Real services**, opt-in. The only tier that can catch W&B API drift.

Tiers 1 and 2 never touch the network — an autouse fixture makes socket
connection raise, so a regression that introduces a live call fails loudly rather
than making CI quietly slow. CI runs tiers 1 and 2 on Python 3.10 and 3.12.

Because mocks cannot catch a signature that moved underneath them,
`tests/test_wandb_api_contract.py` binds the adapter's real call arguments
against the **installed** wandb's real signatures. API drift then fails offline,
immediately, without needing the e2e tier to happen to exercise that code path.

- [MAPPING.md](MAPPING.md) — the contract. Written before the code, and what the
  tests assert against.
- [DECISIONS.md](DECISIONS.md) — every judgement call made where the spec was
  silent, and why.

## Licence

Apache-2.0. See [LICENSE](LICENSE).
