"""Idempotency and resume.

State lives **in the target MLflow server**, not in a local file. A local
journal is one `rm -rf` away from turning a resumed migration into a duplicated
one, and it cannot see runs a colleague migrated from another machine. The
server already stores everything needed:

- ``wandb.run_id`` identifies which W&B run an MLflow run came from. It is the
  idempotency key — **never** the run name, which W&B does not keep unique.
- ``wandb.migration_complete`` is written last, after the run is terminated. A
  run carrying the first tag but not the second was interrupted mid-write, and
  is therefore garbage that must be replaced rather than kept.
- ``wandb.migration_version`` records which version of the mapping produced the
  run, so a future mapping change can force a re-migration instead of silently
  leaving stale data in place.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)

RUN_ID_TAG = "wandb.run_id"
COMPLETE_TAG = "wandb.migration_complete"
VERSION_TAG = "wandb.migration_version"
SWEEP_PARENT_TAG = "wandb.sweep_parent_id"

#: Bump when a mapping change makes previously-migrated runs wrong. Runs written
#: by an older version are treated as incomplete and re-migrated.
MAPPING_VERSION = "1"

_PAGE_SIZE = 1000


@dataclass(frozen=True)
class ExistingRun:
    """An MLflow run already present for a given W&B run id."""

    mlflow_run_id: str
    complete: bool
    version: str | None

    @property
    def reusable(self) -> bool:
        """True only for a run that finished writing under the current mapping."""
        return self.complete and self.version == MAPPING_VERSION


class MigrationState:
    """An index of what has already been migrated into one experiment.

    Built with one paginated search rather than a query per run: a project with
    4,000 runs would otherwise pay 4,000 round trips just to discover it has
    nothing to do.
    """

    def __init__(self, client: MlflowClient, experiment_id: str) -> None:
        self.client = client
        self.experiment_id = experiment_id
        self._by_wandb_id: dict[str, ExistingRun] = {}
        self._sweep_parents: dict[str, str] = {}
        self._loaded = False

    # -- loading ---------------------------------------------------------- #

    def load(self) -> None:
        self._by_wandb_id = {}
        self._sweep_parents = {}
        for run in self._search():
            tags = dict(run.data.tags)
            sweep_parent = tags.get(SWEEP_PARENT_TAG)
            if sweep_parent:
                self._sweep_parents.setdefault(sweep_parent, str(run.info.run_id))
                continue
            wandb_id = tags.get(RUN_ID_TAG)
            if not wandb_id:
                continue
            self._by_wandb_id[wandb_id] = ExistingRun(
                mlflow_run_id=str(run.info.run_id),
                complete=tags.get(COMPLETE_TAG) == "true",
                version=tags.get(VERSION_TAG),
            )
        self._loaded = True
        logger.debug(
            "loaded %d existing runs and %d sweep parents from experiment %s",
            len(self._by_wandb_id),
            len(self._sweep_parents),
            self.experiment_id,
        )

    def _search(self) -> list[Any]:
        out: list[Any] = []
        token: str | None = None
        while True:
            page = self.client.search_runs(
                experiment_ids=[self.experiment_id],
                run_view_type=ViewType.ACTIVE_ONLY,
                max_results=_PAGE_SIZE,
                page_token=token,
            )
            out.extend(page)
            token = getattr(page, "token", None)
            if not token:
                return out

    # -- queries ---------------------------------------------------------- #

    def lookup(self, wandb_run_id: str) -> ExistingRun | None:
        return self._by_wandb_id.get(wandb_run_id)

    def sweep_parent(self, sweep_id: str) -> str | None:
        return self._sweep_parents.get(sweep_id)

    @property
    def loaded(self) -> bool:
        return self._loaded

    def __len__(self) -> int:
        return len(self._by_wandb_id)

    # -- mutation --------------------------------------------------------- #

    def remember_sweep_parent(self, sweep_id: str, mlflow_run_id: str) -> None:
        self._sweep_parents[sweep_id] = mlflow_run_id

    def discard(self, wandb_run_id: str, mlflow_run_id: str) -> None:
        """Retire a stale or half-written run.

        Deletion is MLflow's soft delete: the run leaves the active view (so it
        cannot show up as a duplicate) but stays restorable, which is the right
        default for a tool whose whole job is not losing data.
        """
        logger.info("replacing existing MLflow run %s for W&B run %s", mlflow_run_id, wandb_run_id)
        self.client.delete_run(mlflow_run_id)
        self._by_wandb_id.pop(wandb_run_id, None)

    def mark_complete(self, wandb_run_id: str, mlflow_run_id: str) -> None:
        """Write the completion marker. Must be the migrator's last write for a run."""
        self.client.set_tag(mlflow_run_id, VERSION_TAG, MAPPING_VERSION)
        self.client.set_tag(mlflow_run_id, COMPLETE_TAG, "true")
        self._by_wandb_id[wandb_run_id] = ExistingRun(
            mlflow_run_id=mlflow_run_id, complete=True, version=MAPPING_VERSION
        )
