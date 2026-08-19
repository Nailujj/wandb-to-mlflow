# MAPPING.md — the migration contract

This document is written **before** the implementation and is the thing the test
suite asserts against. If the code and this table disagree, the code is wrong.

Legend for **Lossless?**:

- **yes** — the information survives the round trip intact.
- **approximate** — a value is produced, but it is derived/estimated, not copied.
- **lossy** — survives in a degraded form (truncated, flattened, coarsened).
- **no** — not migrated at all. These are the rows that matter most.

---

## 1. Full mapping table

| W&B concept | MLflow destination | Lossless? | Notes |
|---|---|---|---|
| project | experiment (name configurable via `--experiment`, defaults to the W&B project name) | yes | Experiment is created if absent. |
| run | run | yes | One MLflow run per W&B run. |
| `run.id` | tag `wandb.run_id` | yes | This is the idempotency key — **not** the run name. |
| `run.name` | tag `mlflow.runName` | yes | Names are not unique in W&B and are not treated as keys. |
| `run.url` | tag `wandb.url` | yes | Lets a user jump back to the original while W&B still exists. |
| `run.entity` / `run.project` | tags `wandb.entity`, `wandb.project` | yes | |
| `run.created_at` | `run.info.start_time` (epoch ms, UTC) | yes | Backdated explicitly on run creation. |
| last history timestamp (`_timestamp`) | `run.info.end_time` | **approximate** | W&B exposes no true end time via the public API. Falls back to `run.summary._timestamp`, then to `start_time + run.summary._runtime`, then to `start_time`. The chosen source is recorded in tag `wandb.end_time_source`. |
| `run.state` | run status + tag `wandb.state` | **lossy** | `finished`→`FINISHED`, `crashed`→`FAILED`, `failed`→`FAILED`, `killed`→`KILLED`, `running`→`RUNNING`, anything else→`FINISHED` with the raw string preserved in `wandb.state`. `crashed` and `failed` are distinct in W&B and collapse to `FAILED`. |
| `run.config` (nested dicts) | params, dotted keys (`optimizer.lr`) | **lossy** | Flattened; lists are JSON-serialised into a single param. Values over `MAX_PARAM_VAL_LENGTH` are truncated (see §2). Keys starting `_` are dropped. Note that W&B discards empty-dict config values server-side, so they never reach the migrator at all. |
| `run.summary`, numeric values | metrics `final.<key>` logged at step 0 | yes | Gives the runs table a sortable final-value column. **W&B populates summary itself** with the last value of every logged key, so most `final.*` metrics appear whether or not the user ever wrote a summary. Measured against a live run. |
| `run.summary`, non-numeric values | params `summary.<key>` | **lossy** | JSON-serialised, truncated at the param limit. |
| `run.summary`, `_`-prefixed keys | dropped | n/a | W&B internals (`_step`, `_runtime`, `_timestamp`, `_wandb`). |
| `run.history()` scalars | metrics with original `step` and `timestamp` | yes for finite scalars | Read with `scan_history()` so no sampling occurs. Step comes from `_step`, timestamp from `_timestamp`. |
| history `bool` values | dropped, counted | **no** | `bool` is a subclass of `int` in Python; logging it as a metric would silently invent 0/1 data. Counted under `dropped.bool`. |
| history `NaN` / `±inf` | dropped, counted | **no** | Backend support is inconsistent; not gambled on. Counted under `dropped.nonfinite`. **W&B's API returns these as the JSON strings `"NaN"`, `"Infinity"`, `"-Infinity"`, not as floats** (measured, not assumed). Those three exact spellings are recognised so the drop is filed under the right reason — they are rejected either way, so no number is ever invented from a string. A user who genuinely logged the string `"NaN"` is counted as non-finite rather than as a string; both are dropped. Lowercase `"nan"`/`"inf"` stay classified as strings. |
| history `None` | rejected, **not** counted as loss | n/a | W&B pads sparse rows with explicit nulls for keys that were not logged at that step — a run logging an image every 5th epoch of 25 comes back with 20 nulls. There was never a value, so this is absence, not loss. Tracked separately as padding and reported as such; it never appears in `wandb.dropped` and never fails `verify`. |
| history strings | dropped, counted | **no** | Scalar-looking strings are **not** parsed into numbers. Counted under `dropped.str`. |
| history lists | dropped, counted | **no** | Counted under `dropped.list`. |
| history media / tables (dicts with `_type`) | dropped as **metrics**, counted by `_type`; the underlying **files** migrate under `wandb_files/media/` when `--files true` | **partial — read the note** | `wandb.Image`, `wandb.Table`, `wandb.Audio`, `wandb.Video` and plots cannot become MLflow metrics and will not render as panels: that much is genuinely lost. But the **bytes are not**. With `--files true` the actual PNGs/JSON land as MLflow artifacts under `wandb_files/media/...`, verified on a live run (5 logged images → 5 PNGs migrated). You lose the charts and the step-linked association, not the images. Per-type counts land in `wandb.dropped` and in the run report. |
| `wandb-history` artifact (W&B logs one per run automatically) | artifact `artifacts/run-<id>-history_v0/0000.parquet` with `--artifacts true` | yes | **The complete raw history as parquet**, including every value that could not become a metric — media references, `NaN`s, bools, strings — at full fidelity and with original steps. This is the closest thing to a lossless escape hatch the migration has, and it costs a few KB per run. Verified on live runs. |
| metric/param/tag keys illegal in MLflow | sanitised keys | **lossy** | See §3. Original names preserved in tag `wandb.renamed_keys`. |
| `run.tags` | tags `wandb.tag.<t>` = `"true"` | yes | MLflow has no tag-set concept; each W&B tag becomes its own key. |
| `run.notes` | tag `mlflow.note.content` | **lossy if long** | Truncated at `MAX_TAG_VAL_LENGTH`; truncation is flagged in `wandb.truncated_tags`. |
| `run.group` | tag `wandb.group` | yes | |
| `run.job_type` | tag `wandb.job_type` | yes | |
| sweep | synthetic parent MLflow run + `mlflow.parentRunId` on each child | **partial** | The parent run is created by this tool; it has no W&B counterpart beyond the sweep id. Tagged `wandb.sweep_id` and `wandb.is_sweep_parent=true`. |
| sweep config / search space / method | tag `wandb.sweep_id` only | **no** | The search space, method, metric goal and early-terminate config are **not migrated**. |
| sweep best-run pointer | — | **no** | |
| run files (`run.files()`) | artifacts under `wandb_files/` | yes | Opt-in via `--files`. Subject to `--max-artifact-size`. |
| logged artifacts (`run.logged_artifacts()`) | artifacts under `artifacts/<name>/` | **lossy** | Opt-in via `--artifacts`. Note that W&B finalises some artifacts (the per-run history one especially) **asynchronously** after a run ends — migrating seconds after a run finishes can miss them. Re-run with `--overwrite` to pick them up. Bytes are copied; **versions, aliases, lineage, types and metadata are not**. A JSON sidecar `artifacts/<name>/_wandb_artifact.json` records name, version, aliases and digest as inert text. |
| reference artifacts (`is_reference`) | tag/sidecar with the source URI; bytes not fetched | **no** | The tool will not reach into S3/GCS/HTTP on the user's behalf. Recorded in `wandb.reference_artifacts`. |
| artifacts over `--max-artifact-size` | skipped, counted, listed in the report | **no** | Size is checked before download. |
| system metrics (`run.history(stream="events")`) | metrics prefixed `system.` | **lossy** | Opt-in via `--system-metrics`. W&B samples these server-side; the migrated series is the sampled one, not the raw one. |
| Reports | — | **no** | |
| Workspace panels, custom charts, panel layouts | — | **no** | |
| Model registry, registered models, artifact aliases | — | **no** | |
| Artifact lineage graph | — | **no** | |
| Run comments / discussion threads | — | **no** | |
| Team / user / permission structure | — | **no** | |
| Launch jobs, queues, agents | — | **no** | |
| Automations / webhooks / alerts | — | **no** | |

---

## 2. Truncation rules

All limits are read at runtime from the installed MLflow (`limits.py`), never
hardcoded.

- Param values longer than `MAX_PARAM_VAL_LENGTH` are cut to
  `limit - len("…[truncated]")` and get `…[truncated]` appended.
- Tag values longer than `MAX_TAG_VAL_LENGTH` are truncated the same way.
- Every truncated param key is listed in tag `wandb.truncated_params`
  (JSON array, itself truncated to the tag limit).
- Every truncated tag key is listed in tag `wandb.truncated_tags`.

---

## 3. Key sanitisation

The accepted character set was determined by reading the installed MLflow's
`validate_param_and_metric_name`, **not** assumed:

- POSIX: `^[/\w.\- :]*$` — slashes, unicode word characters, periods, dashes,
  colons, spaces.
- Windows: the same **minus the colon**.

Two consequences verified by test rather than assumed:

- `train/loss` survives **unchanged**. `/` is legal.
- `héllo` survives **unchanged**. `\w` is unicode-aware, so accented letters and
  most non-Latin scripts are already legal MLflow keys.
- `x@y!` does not survive; `@` and `!` are replaced.

This tool sanitises to the **portable** set (colon excluded), so a migration
produced on Linux does not break against a tracking server running on Windows.

MLflow additionally rejects keys whose path form is non-canonical
(`path_not_unique`): empty keys, `.`, `..`, `a//b`, `../x`, leading `/`,
trailing `/`.

Rules, applied in order:

1. Every character outside `[/\w.\- ]` is replaced with `_`.
2. The key is split on `/`; any segment that is empty, `.` or `..` is replaced
   with `_` / `_` / `__` respectively, and rejoined. This makes the key
   path-canonical without silently collapsing `a/../b` into `b`.
3. Leading/trailing whitespace is stripped; an empty result becomes `unnamed`.
4. The key is truncated to `MAX_ENTITY_KEY_LENGTH`, then step 2 is reapplied in
   case the cut produced a trailing `/`.
5. If two distinct source keys sanitise to the same target, **every** colliding
   key — including the first — gets `_<6 hex chars of sha1(original)>` appended
   (with the stem shortened as needed to stay under the length limit), and a
   warning is logged.
6. Any key that changed is recorded in tag `wandb.renamed_keys` as a JSON object
   `{"<original>": "<sanitised>"}`, truncated at the tag limit.

Sanitisation is deterministic: the same input **set** always produces the same
output set, independent of iteration order, because collisions are resolved by
hashing the original key rather than by arrival order. This is why the first
colliding key is suffixed too — leaving it bare would make the result depend on
which key was seen first.

## 4. Reserved / conflicting keys

- `final.<k>` is reserved for summary metrics. A history key that already reads
  `final.<something>` is left alone; collisions between a history metric and a
  summary metric of the same name are reported, and the summary value wins at
  step 0 (the history series keeps its own steps).
- Tags under the `mlflow.` namespace are only written where this table says so
  (`mlflow.runName`, `mlflow.note.content`, `mlflow.parentRunId`). W&B data
  never lands in `mlflow.*` otherwise.

---

## 5. Per-run report

Every migrated run carries a machine-readable summary of what was lost, as tag
`wandb.dropped` (JSON), and the same data appears in the CLI report and in
`verify`'s expected-loss comparison:

```json
{"nonfinite": 3, "bool": 2, "str": 1,
 "media": 3, "media_types": {"image-file": 2, "table-file": 1}}
```

Sparse-row nulls are deliberately absent from this tag — see the `None` row
above. They are surfaced in the CLI report, labelled as not being data loss.

Only **history** values appear in this tally. Summary values are never counted
as dropped, because none of them is lost: each becomes either a `final.*` metric
or a `summary.*` param.

**Expected loss** (matches the manifest / this table) never fails `verify`.
**Unexpected loss** (a finite scalar that should have migrated and did not)
always does.
