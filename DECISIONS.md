# DECISIONS.md

A running log of judgement calls made where the build spec was silent or
ambiguous. Newest entries at the bottom.

## M1 — skeleton

**Q: What Python version floor?**
A: 3.10. `mlflow>=2.9` supports it, and `X | None` syntax is used throughout.

**Q: Build backend?**
A: `hatchling`. It is a build-time requirement, not a runtime dependency, so it
does not violate the fixed dependency list.

**Q: `MAPPING.md` says `run.url`, `entity`, `project` become tags, which the
spec's table did not list.**
A: Added. They are free, and someone auditing a migration needs a way back to
the original run while W&B still exists.

**Q: How is `end_time` derived, given W&B's public API exposes no end time?**
A: Fall back chain — last history `_timestamp` → `summary._timestamp` →
`start_time + summary._runtime` → `start_time`. Which one was used is recorded
in the tag `wandb.end_time_source` so the approximation is auditable rather than
invisible.

**Q: What happens on a `final.<k>` / history-key collision?**
A: Summary wins at step 0; the history series keeps its own steps. Reported, not
silently resolved. Recorded in MAPPING.md §4.

**Q: Are `--files`, `--artifacts`, `--system-metrics` on by default?**
A: No. All three are opt-in. A default migration should be fast and cheap; bytes
cost money and time, and the spec marks artifacts as opt-in explicitly.

## M2 — limits and coercion

**Q: MLflow's key regex allows a colon on POSIX but not on Windows. Which set do
we sanitise to?**
A: The portable one — colon excluded. A migration run on Linux against a
tracking server hosted on Windows would otherwise fail server-side on keys the
client considered fine. Cost is one extra rename for a rare character.

**Q: `\w` is unicode-aware, so `héllo` is already a legal MLflow key. Do we
transliterate it anyway?**
A: No. Leaving it unchanged is lossless; transliterating would be gratuitous
loss. Verified by test against the installed MLflow validator.

**Q: MLflow also rejects non-canonical path keys (`a//b`, `../x`, trailing `/`).
The spec did not mention this.**
A: Handled as sanitisation step 2 — per-segment replacement rather than
`posixpath.normpath`, because normpath would collapse `a/../b` to `b` and
silently discard a path component.

**Q: On a collision, does only the second key get the hash suffix?**
A: No — all colliding keys do, including the first. Suffixing only the later one
makes the output depend on iteration order, which contradicts the spec's
"deterministic and collision-checked" requirement.

**Q: MLflow 3.15 reports `MAX_PARAM_VAL_LENGTH == 6000`; older MLflow used 500.**
A: Exactly why `limits.py` reads them at runtime with fallbacks. No test asserts
a literal limit value; tests assert behaviour relative to the resolved limit.

**Q: `mypy --strict` with `python_version = "3.10"` fails inside numpy's own stubs
(`Type statement is only supported in Python 3.12 and greater`).**
A: Dropped the `python_version` pin so mypy targets the interpreter it runs
under. Pinning an older target than the installed stubs were written for breaks
on third-party code, not ours. `requires-python` stays at `>=3.10`.

**Q: `uv run pytest` crashes on this machine inside ROS 2's pytest plugins,
which are on `PYTHONPATH` (`/opt/ros/jazzy/...`).**
A: Environment-specific, not a repo problem — a `-p no:...` workaround in
`pyproject.toml` would ship one developer's machine config to everyone. Local
runs use `env -u PYTHONPATH uv run pytest`; CI, which has no such `PYTHONPATH`,
runs the plain command. Noted in the README's development section.

**Q: How is "no network in unit tests" actually enforced, rather than just
asserted in prose?**
A: An autouse fixture in `tests/conftest.py` patches `socket.connect` to raise
for every test not marked `e2e`. A regression that introduces a live call fails
immediately instead of silently making CI slow and flaky.

## M3 — source protocol and fixtures

**Q: The spec's `SourceRun` protocol has no `entity`/`project`. The migrator
needs both to write the `wandb.entity`/`wandb.project` tags.**
A: Added to the protocol. Also added `system_metrics()` alongside `history()`,
because system metrics come from a different W&B stream (`stream="events"`) and
folding them into `history()` would make the opt-in flag unimplementable.

**Q: How does the migrator honour `--max-artifact-size` without downloading?**
A: `SourceArtifact` exposes `size` and `is_reference` as plain attributes,
populated from the artifact manifest. `is_reference` is derived from whether any
manifest entry carries a `ref`, since W&B has no single flag for it.

**Q: W&B property access raises on partially-populated records.**
A: All adapter attribute reads go through `_safe()`, which logs at debug and
falls back. A migration must not die because one run has no `job_type`.

**Q: `all_runs()` includes the throughput case at 200 steps, not 20,000.**
A: Tier 2 runs on every commit and must stay under a few seconds. The full
20,000-step fixture exists and is exercised by its own test; the seeder logs the
real 20,000 for tier 3.

## M4 — the migrator

**Q: History cannot be sanitised incrementally — a collision is only visible once
every key is known — but streaming avoids buffering. Which wins?**
A: Buffering. Points already written under an un-suffixed key cannot be
retracted, so two distinct W&B series would silently merge into one MLflow
series. Peak memory is O(points in a single run); the spec's worst case
(20,000 steps x 5 metrics) is tens of MB. Correctness over footprint here.

**Q: MLflow 3.x refuses to open the filesystem tracking backend without
`MLFLOW_ALLOW_FILE_STORE=true` ("maintenance mode").**
A: Tier 2 sets it in `conftest.py`, since the file store is the only backend
needing no server. Product code is untouched — but the README tells users the
same flag is needed for `mlflow ui` against `./mlruns` on MLflow 3.

**Q: Does "vendor-neutral" get tested?**
A: Yes. The full fixture set is migrated against a SQLite backend as well as the
file store. The SQL store validates keys and lengths more strictly, so it is
where a sanitisation bug would actually surface.

**Q: Should system metrics contribute to `end_time`?**
A: No. The events stream can outlive the run's own data. Only the history stream
defines when the run stopped. Tested.

**Q: How is "no fluent API" enforced?**
A: A test parses every module's AST and fails on any `mlflow.<fluent>` attribute
access or `from mlflow import <fluent>`. A prose check over the raw source was
tried first and gave a false positive on this module's own docstring.

**Q: The `--overwrite` option exists in `MigrateOptions` but is unused so far.**
A: It belongs to idempotency (M5) and is wired there, not guessed at now.

## M5 — idempotency and resume

**Q: Where does resume state live — a local journal file or the server?**
A: The server. A local journal is one `rm -rf` away from turning a resume into a
duplication, and it cannot see runs a colleague migrated from another machine.
The idempotency key is the `wandb.run_id` tag; `wandb.migration_complete` is
written **last**, so a run carrying the first tag but not the second is
provably half-written and gets replaced rather than trusted.

**Q: How is a half-written run distinguished from a complete one, given MLflow
has no transactions?**
A: Write-ordering. `create_run` → params → metrics → artifacts →
`set_terminated` → completion marker. Any interruption leaves the marker absent.
The cost is one extra tag write per run.

**Q: What if a future mapping change makes already-migrated runs wrong?**
A: `wandb.migration_version` records the mapping version that produced each run.
Bumping `MAPPING_VERSION` makes older runs count as not-reusable, so they are
re-migrated instead of being silently left stale.

**Q: Does `--overwrite` hard-delete?**
A: No, MLflow's soft delete. The run leaves the active view so it cannot appear
as a duplicate, but stays restorable. A tool whose entire purpose is not losing
data should not be the thing that loses it.

**Q: Building the state index — one query per run, or one paginated scan?**
A: One paginated scan of the target experiment. A 4,000-run project would
otherwise pay 4,000 round trips just to discover it has nothing to do. Asserted
by a test that counts `search_runs` calls.

## M6 — verification

**Q: What exactly separates expected from unexpected loss?**
A: `expected_dropped` in the manifest, compared **in both directions**. Dropping
more than the manifest says is unexpected loss. Dropping *less* is worse than it
sounds: it means a value the mapping requires be rejected — a NaN, a bool — got
through and is now fabricated data sitting in someone's metric chart. Both fail;
an exact match is reported as informational.

**Q: Should a metric present in MLflow but absent from the manifest fail?**
A: Yes. Fabricated series are as damaging as missing ones, and this is precisely
where a regression in the `bool`-is-`int` rule would surface. Metric key sets are
compared for equality, not containment.

**Q: Live mode compares against expectations derived from the same source API
the migration read. Doesn't that only prove self-consistency?**
A: Yes, and that limitation is stated in the module docstring. It is still the
only thing available to a user migrating real data, and it does catch write-side
failures (dropped batches, truncated series, wrong statuses). The manifest mode
exists because ground truth recorded at seed time is the only way to test the
read side, and that is what the self-test uses.

**Q: A verify bug found during M6 — planning did not know about sweeps, so every
sweep child verified as having an "unexpected parent".**
A: `RunReport` now carries `wandb_sweep_id`, populated during planning before
any parent run exists. Worth recording as evidence that verify earns its keep:
it caught a real defect in its own first run.

## M7 — the CLI

**Q: `--artifacts` reads naturally as a bare flag, but the spec's `MLproject`
substitutes `--artifacts {artifacts}` with a default of `"false"`, which a flag
cannot accept.**
A: `--artifacts`, `--files` and `--system-metrics` take an explicit value
(`--artifacts true`). One spelling that works from both a shell and an MLflow
entry point beats two spellings that each work in one place. A value that is
neither truthy nor falsy is rejected, never guessed.

**Q: `--max-artifact-size` — raw bytes or human sizes?**
A: Both. `parse_size` accepts `512`, `20MB`, `1.5GiB`. Nobody should have to
type `104857600`.

**Q: `plan` "writes nothing", but MLflow's file store creates a `Default`
experiment the moment a client touches it.**
A: That is MLflow initialising its own store, not the tool writing. The test
asserts the precise thing that matters: after `plan`, the target experiment does
not exist and no run exists anywhere.

**Q: How is "no `print()` in library code" enforced?**
A: An AST test over every module except `cli.py` and `__main__.py`.

## M8 — the seeder

**Q: The spec wants a seeded run in W&B's `crashed` state. `crashed` is what W&B
records when a run's heartbeat stops — it cannot be produced deterministically
through the SDK (it needs a killed process and a multi-minute heartbeat timeout).**
A: Both status runs are seeded with non-zero exit codes, which W&B records as
`failed`, and the manifest records `FAILED` for both. The `crashed → FAILED`
half of the mapping is covered by a tier-2 fixture and a unit test on
`STATE_TO_STATUS` instead. Stated here rather than quietly seeding two identical
runs and calling the case covered.

**Q: The manifest must come from what was actually logged, not from a second W&B
query — but the seeder can only know what it handed to `wandb.log`.**
A: That is exactly what it uses. `build_specs()` is pure data; `expected_for()`
counts over those payloads with plain arithmetic; neither runs the migrator.
Key sanitisation is the one shared piece, because sanitisation *is* the mapping
contract. A test migrates the `SourceRun`s W&B would return for each spec and
asserts the independently-derived manifest verifies clean — so the seeder's
arithmetic and the migrator's behaviour are checked against each other offline,
before a real W&B project is ever created.

**Q: How is a reference artifact seeded without cloud credentials?**
A: A `file://` reference to a local file. It is a genuine reference artifact —
W&B stores the URI, not the bytes — and needs no S3 account to set up.

**Q: `seed --cleanup` cannot delete the project itself.**
A: W&B's public API has no project delete (`Project` exposes no `delete`; `Run`
does). Cleanup deletes every run and artifact, then tells the user plainly that
the empty project shell has to go from the web UI. Guarded so it will only ever
act on a project named with the `w2m-selftest-` prefix this tool creates.

**Q: `demo` defaults to 2,000 throughput steps, not 20,000.**
A: `demo` is the thing a stranger runs first; it should finish. `--steps 20000`
gets the full case, and tier 3 uses the full seeder.

## M9 — parallelism

**Q: What actually races when `--workers > 1`?**
A: Sweep-parent creation. Several children of one sweep migrate at once and
would each create their own parent run, silently splitting a sweep into N
sweeps. The state lock is therefore held **across** the check-and-create, not
just around the dictionary write.

**Q: Does parallelism change the output?**
A: No, and that is tested: reports are collected in submission order, so a
migration's output does not depend on which worker finished first. The serial
and parallel migrations of the full fixture set are asserted identical, and
parallel re-runs are asserted still idempotent.

**Q: Default `--workers`?**
A: 1. Parallel writes to a shared tracking server are a decision the operator
should make knowingly — some servers rate-limit, and the file store is not
designed for it. The option is there for people migrating thousands of runs.

## M10 — MLflow Project

**Q: `python_env.yaml` pins `python: "3.11"` while `requires-python` is `>=3.10`.**
A: `python_env` needs a concrete version MLflow can provision; the floor stays
3.10 and a test asserts the pinned version satisfies it. Verified for real:
`mlflow run . -e verify` under the default env manager built Python 3.11 from
source, created a fresh venv from this file, installed the package, and exited 0.

**Q: Is the ambient-run trap actually avoided in a real `mlflow run`, not just
in the unit test?**
A: Checked directly after a real `mlflow run . -e verify`: the entry-point runs
sit in `Default` with their own two params, zero metrics and zero children,
while the 20 migrated runs plus one sweep parent sit untouched in the target
experiment.

**Q: `tests/test_mlproject.py` imports `yaml`, which is not in the fixed
dependency list.**
A: It arrives with MLflow, which is a runtime dependency, so nothing new is
installed. Not added to `pyproject.toml` — a test asserts the declared
dependency sets are exactly the four runtime and four dev packages the spec
fixes, so an accidental addition fails the build.

**Q: `mlflow run . -e demo` is the headline gate but needs real W&B credentials.**
A: Every other entry point is verified end to end here. `demo` is verified in
two halves: its seed step is faked while its migrate and verify steps run for
real in a tier-2 test, and the whole loop against live services is the tier-3
`test_the_whole_loop`. Stated plainly rather than claimed as run.

## M11 — docs

**Q: What goes above the fold in the README?**
A: The "not migrated" table, before installation instructions. Someone deciding
whether to cancel a W&B subscription needs to know what they are about to lose
before they need to know how to `pip install`.

## Live self-test findings (tier 3, run against a real W&B account)

The e2e tier immediately justified its existence. Four behaviours of real W&B
that no amount of offline testing could have surfaced, each measured with a
targeted probe rather than guessed at:

**1. `wandb.Image` rejects a nested Python list.**
`AttributeError: 'list' object has no attribute 'ndim'`. The seeder now passes a
numpy array. numpy is imported inside the seeder's network path only — it is a
transitive dependency via MLflow, not one this tool declares.

**2. W&B cannot store 4-byte characters in `run.name` or `run.notes`.**
`Error 3988 (HY000): Conversion from collation utf8mb4_unicode_ci into
utf8mb3_general_ci impossible`. Probed field by field: accented Latin, Hangul,
Cyrillic, Greek and CJK are fine everywhere; emoji are fine in **config values
and tags** but rejected in **name and notes**. The seeder's encoding case now
puts emoji where W&B accepts them and keeps three-byte scripts in name/notes.
`tests/fixtures.py` deliberately keeps an emoji-in-name run: the migrator must
handle one even though W&B cannot currently produce one.

**3. W&B returns non-finite numbers as JSON *strings*.**
`float("nan")` comes back from `scan_history` as `"NaN"`, `float("inf")` as
`"Infinity"`. This is a finding about the **tool**, not just the seeder: those
values were being correctly rejected (spec 5.1 forbids parsing scalar-looking
strings) but filed under `dropped.str`, so the report told users they had logged
strings they never logged. `coerce.as_metric` now recognises those three exact
JSON spellings and files them under `nonfinite`. It changes only the
*classification* of a rejection — the value is dropped either way, so no number
is ever invented from a string, and lowercase `"nan"`/`"inf"` stay strings.

**4. W&B auto-populates `run.summary`** with the last logged value of every
history key, media included, whether or not the user ever wrote a summary. The
first live self-test reported a dozen "unexpected" `final.*` metrics that were
all entirely correct — the manifest was wrong, not the migration. Both
`seed.expected_for` and the offline `fake_run_for` bridge now model this.

Two consequential decisions fell out of these:

**Q: Should summary values count toward `wandb.dropped`?**
A: No, and this was changed. Every summary value becomes either a `final.*`
metric or a `summary.*` param — none is lost — so counting them as dropped
overstated the loss. The tally is now history-only, which is the only place data
actually goes missing. MAPPING.md says so explicitly.

**Q: The seeded sweep's children carried config the manifest never predicted.**
A: A W&B sweep injects its own search-space parameters into each child's config.
The seeder now sweeps over exactly the `lr` values the specs declare, so the
search space *is* the children's configs and the two cannot disagree.

**Q: W&B discards empty-dict config values server-side.**
A: Modelled in `expected_for` rather than removed from the seeded config: the
seeder really does log it, and W&B really does drop it. Recorded in MAPPING.md.

**Q: `mlflow run .` kept executing pre-fix code after the source had changed.**
A: MLflow caches the run environment keyed on `python_env.yaml`. Editing the
package does not invalidate that key, so the stale wheel stayed installed and a
bug already fixed kept reproducing. Not a repo defect, but an easy hour to lose,
so it is in the README's development section: `rm -rf ~/.mlflow/envs/*`, or
iterate with `--env-manager local`.

**Q: `demo` printed `mlflow ui --backend-store-uri ./mlruns` even when the store
came from `MLFLOW_TRACKING_URI`.**
A: Fixed to print `client.tracking_uri`, the effective URI. The old message sent
the reader to an empty store — the one moment the tool has someone's attention
is a bad one to be wrong about where the data is.

## Read-only guarantee

**Q: Should the tool ever delete migrated W&B data, or offer to?**
A: No — and this was raised by the user rather than the spec, which is silent on
it. W&B is the user's insurance policy: they keep it precisely so that if the
migration turns out to be wrong, the originals still exist. A tool that migrates
*and* mutates the source destroys the thing that makes it safe to try in the
first place. There is no `--delete-source` flag and there will not be one.

Migration touches exactly five W&B API calls — `runs()`, `scan_history()`,
`files()`, `logged_artifacts()`, `download()` — all reads.

Made enforceable rather than left as a convention:

- An AST test fails the build if `source.py` calls any mutating W&B method.
- Another asserts the whole package contains exactly one `delete` call, in the
  seeder. If that count ever changes, someone added a way to destroy user data.
- The prefix guard moved **into** `seed.cleanup()`, the function that actually
  deletes, having previously lived only in `cli._cleanup`. A guard a library
  caller can bypass is not a guard, and this is the single code path in the
  package capable of destroying anything.

## Planning must not take the migration's shortcuts

**Q: Live `verify` reported every correctly-migrated run as broken — zero params,
zero metrics expected — but only against a project that had already been
migrated.**
A: Planning reuses `Migrator` with `dry_run=True`, and `migrate_run` skips runs
already present in the target, returning early to avoid re-reading their history
from W&B. That shortcut is right for a real migration and wrong for planning:
`plan` and live `verify` both need the full picture of what a run *should*
contain. The early return now happens only when actually migrating; planning
notes `skip_reason` and carries on collecting.

`plan` was equally affected — run against an already-migrated project it
reported 0 params and 0 metric points. It now reports the full picture plus a
line saying how many runs are already in MLflow.

**Q: Why did 259 tests miss this?**
A: They pointed the planner at a different tracking store from the migration, so
it never saw the existing runs and never took the skip path. The regression test
deliberately uses the same client for both. `manifest_from_source` also no
longer defaults to a bare `MlflowClient()` — that silently followed whatever
`MLFLOW_TRACKING_URI` happened to be, which is how the two stores diverged in
the first place; the caller now passes the client the verification uses.

## Sparse nulls are not data loss

**Q: A live migration reported `none=20` under "Values dropped" for a run that
lost nothing.**
A: W&B pads sparse history rows with explicit nulls for keys not logged at that
step — measured: a run logging an image every 5th epoch of 25 comes back with
exactly 20 nulls. There was never a value there, so counting it as loss tells
the user they lost 20 things they never had. This is why the spec says to reject
`None` *silently* while every other rejection is "counted and reported".

`None` is now tracked separately as padding: excluded from `DropReport.total`
and from the `wandb.dropped` tag, so it neither inflates the loss report nor
affects `verify`. The CLI still shows the count, explicitly labelled as not
being data loss, because hiding it entirely would be its own kind of dishonesty.

This also explains an apparent non-determinism: the same project migrated twice
reported different drop counts. The first run happened seconds after the runs
were created, before W&B had materialised their history, so the nulls were not
there yet. Nothing about the migration changed — only how much of the history
W&B had finished writing.

## MLflow 3's UI cannot read a file store

**Q: After a successful migration into `./mlruns`, the MLflow UI showed the
experiment but no runs, and spammed 500s.**
A: MLflow 3 has the filesystem store in maintenance mode. It refuses to open
without `MLFLOW_ALLOW_FILE_STORE=true`, and its UI then calls endpoints the file
store does not implement (`traces/metrics` returns 500 on every poll). The data
was migrated correctly the whole time — the `2.0` run-search endpoint returned
all seven runs with full metrics and params — but the UI was unusable.

The fix is the destination, not the tool: migrate into SQLite. The same
migration into `sqlite:///mlflow.db` renders correctly, and the tier-2 suite
already covered that backend precisely because "vendor-neutral" had to mean more
than "works against the file store". The README now recommends SQLite up front
instead of the file store, which was bad advice on my part.

## Media survives better than the docs claimed

**Q: Does anything of a `wandb.Image` reach MLflow?**
A: More than MAPPING.md originally said, and the correction matters because the
media row is the headline "what do I lose" claim.

Verified on a live run that logged 5 images:

- The **values** cannot become MLflow metrics and will not render as image
  panels. That much really is lost.
- The **files** migrate under `wandb_files/media/images/*.png` when `--files
  true` is passed — all 5 PNGs, byte for byte.
- W&B additionally logs a `wandb-history` artifact for **every** run, which
  `--artifacts true` migrates: a parquet of the complete raw history including
  every value that could not become a metric — media references, `NaN`s, bools,
  strings — with original step numbers, at a few KB per run.

So "media is not migrated" was wrong as stated. What is lost is the
visualisation and the step-linked association, not the bytes. Both documents now
say so, and `migrate` prints the two flags that would keep more when it sees
media dropped and those flags are off. Someone deciding whether to cancel a
subscription is entitled to the accurate version.

**Q: Why did one run's history artifact fail to migrate the first time?**
A: W&B finalises some artifacts asynchronously after a run ends. The run created
last had no `logged_artifacts()` yet when the migration read it seconds later;
`--overwrite` picked it up. Not a tool defect, but a real trap — documented in
both MAPPING.md and the README.
