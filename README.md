# wandb-to-mlflow

Migrate Weights & Biases runs into any MLflow tracking server.

---

## Read this before you cancel your W&B subscription

Some things **do not survive the migration**. Not "degrade" — are not migrated
at all. If any of these matter to you, export them separately **before** you
lose access:

| Not migrated | What that means |
|---|---|
| **Media and tables** | Every `wandb.Image`, `wandb.Table`, `wandb.Audio`, `wandb.Video`, plot and custom rich type logged to history is dropped. They are counted and reported per type, never silently discarded — but the data does not come across. |
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
- `NaN`, `±inf`, `None`, strings, lists and **booleans** are not logged as
  metrics. Booleans especially: `True` is an `int` in Python, and logging it as
  `1.0` would invent data that was never measured.

### Your W&B data is never touched

Migrating **reads** from W&B and writes to MLflow. That is the whole of it. The
migration path calls exactly five things on the W&B API — `runs()`,
`scan_history()`, `files()`, `logged_artifacts()` and `download()` — all of them
reads. Nothing in `plan`, `migrate` or `verify` can create, modify or delete a
W&B run, artifact or project.

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

**[MAPPING.md](MAPPING.md) is the full contract**, and it is what the test suite
asserts against. Every drop above is counted, surfaced in the CLI report, and
written to the `wandb.dropped` tag on the migrated run.

---

## Install

```bash
uv venv && uv pip install -e ".[dev]"
```

## Use it

```bash
# See what would happen. Writes nothing.
wandb-to-mlflow plan --entity my-team --project my-project

# Do it.
export MLFLOW_TRACKING_URI=http://localhost:5000
wandb-to-mlflow migrate --entity my-team --project my-project

# With artifact bytes and run files, 8 runs at a time.
wandb-to-mlflow migrate --entity my-team --project my-project \
    --artifacts true --files true --max-artifact-size 500MB --workers 8

# Check the result against the live W&B project.
wandb-to-mlflow verify --entity my-team --project my-project --experiment my-project
```

Re-running `migrate` is free and safe: runs already migrated are skipped, not
duplicated. A migration killed halfway leaves a run that the next attempt
replaces rather than trusts. `--overwrite` re-migrates everything (soft-deleting
what was there, so nothing is actually lost).

### Options that cost bytes are opt-in

`--artifacts`, `--files` and `--system-metrics` all default to `false`. They take
an explicit value (`--artifacts true`) rather than being bare flags, because
`MLproject` entry points substitute parameters positionally and cannot omit a
flag conditionally.

---

## It tests itself

```bash
wandb-to-mlflow demo --entity my-team
```

This seeds a **real** W&B project with sixteen deliberately hostile runs —
20,000-step histories, `NaN`s, booleans, colliding metric keys, 20,000-character
config values, media, a sweep, emoji — migrates it, and verifies the result
against a manifest the seeder wrote from what it actually logged. Exit code 0,
or a precise diff.

The manifest matters: verifying against a second query to W&B would only prove
the tool is self-consistent with itself. Ground truth recorded at seed time is
the only thing that tests the read side.

`demo` prints the cleanup command it wants you to run afterwards. The seeded
project is named `w2m-selftest-<utc-timestamp>`, and `seed --cleanup` refuses to
touch anything not named that way.

---

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

---

## UI acceptance checklist

Automated tests cannot catch a broken step axis. After `demo`, open the UI:

```bash
# MLflow 3 requires this to open a ./mlruns file store at all.
MLFLOW_ALLOW_FILE_STORE=true mlflow ui
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
- [ ] `ünïcode 🎉 실험` renders correctly in the run list, and its notes show the
      emoji and Cyrillic.
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

---

## Development

```bash
uv run pytest                  # tiers 1 and 2: no network, a few seconds
uv run mypy --strict src/
uv run ruff check .
W2M_E2E=1 W2M_E2E_ENTITY=my-team uv run pytest -m e2e   # tier 3: real services
```

Tiers 1 and 2 never touch the network — an autouse fixture makes socket
connection raise, so a regression that introduces a live call fails loudly
rather than making CI quietly slow. CI runs tiers 1 and 2 only.

If `pytest` crashes inside unrelated plugins on your machine (ROS 2 puts its own
pytest plugins on `PYTHONPATH`, for one), run `env -u PYTHONPATH uv run pytest`.

**If you change the code and `mlflow run .` keeps running the old version**,
that is MLflow's environment cache, not your edit. It keys the environment on
`python_env.yaml`, so editing the package does not invalidate it and the stale
wheel stays installed. Clear it with `rm -rf ~/.mlflow/envs/*`, or iterate with
`--env-manager local`.

- [MAPPING.md](MAPPING.md) — the contract. Written before the code, and what the
  tests assert against.
- [DECISIONS.md](DECISIONS.md) — every judgement call made where the spec was
  silent, and why.

## Licence

Apache-2.0. See [LICENSE](LICENSE).
