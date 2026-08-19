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
