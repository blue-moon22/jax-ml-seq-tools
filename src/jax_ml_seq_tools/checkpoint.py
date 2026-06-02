"""Orbax-based checkpoint utilities."""

from __future__ import annotations

import logging
from functools import wraps
from pathlib import Path

import jax
import orbax.checkpoint as ocp

log = logging.getLogger(__name__)


def get_abstract_state(state):
    """Returns a shape/dtype-only version of the model's state."""
    return jax.tree_util.tree_map(ocp.utils.to_shape_dtype_struct, state)


def restore_checkpoint(mngr: ocp.CheckpointManager, state):
    """Restore a (state, metrics) pair from an Orbax checkpoint manager."""
    restored = mngr.restore(
        0,
        args=ocp.args.Composite(
            state=ocp.args.StandardRestore(get_abstract_state(state)),
            extra_metadata=ocp.args.JsonRestore(),
        ),
    )
    log.debug("Train state and metrics restored from checkpoint")
    state, metrics = restored.state, restored.extra_metadata
    return state, metrics


def restore(store_path, state):
    """Open a checkpoint manager at `store_path` and restore (state, metrics)."""
    with ocp.CheckpointManager(Path(store_path).resolve()) as mngr:
        state, metrics = restore_checkpoint(mngr, state)
    return state, metrics


def save_final_checkpoint(mngr, state, metrics):
    """Save the final training state and metrics to an Orbax checkpoint."""
    mngr.save(
        0,
        args=ocp.args.Composite(
            state=ocp.args.StandardSave(state),
            extra_metadata=ocp.args.JsonSave(metrics),
        ),
    )
    mngr.wait_until_finished()


def restorable(train_fn):
    """Decorator that makes a training function checkpoint-aware."""

    @wraps(train_fn)
    def wrapper(state, store_path: str | None = None, **kwargs):
        if store_path:
            mngr = ocp.CheckpointManager(Path(store_path).resolve())
            try:
                state, metrics = restore_checkpoint(mngr, state)
            except FileNotFoundError:
                log.debug(
                    "Training new model checkpointing state and metrics..."
                )
                state, metrics = train_fn(state, **kwargs)
                save_final_checkpoint(mngr, state, metrics)
        else:
            log.debug("Training new model without checkpointing...")
            state, metrics = train_fn(state, **kwargs)
        return state, metrics

    return wrapper
