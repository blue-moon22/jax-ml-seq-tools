"""Training and evaluation loops."""

from __future__ import annotations

from typing import Any

import jax
import optax
import tensorflow as tf
from tqdm.auto import tqdm

from .checkpoint import restorable
from .metrics import MetricsLogger, compute_metrics
from .models import TrainStateWithBatchNorm


@restorable
def train(
    state: TrainStateWithBatchNorm,
    rng: jax.Array,
    dataset_splits: dict[str, tf.data.Dataset],
    num_steps: int,
    eval_every: int = 100,
) -> tuple[TrainStateWithBatchNorm, Any]:
    """Train a model and log metrics over steps."""
    metrics = MetricsLogger()
    train_batches = dataset_splits["train"].as_numpy_iterator()

    steps = tqdm(range(num_steps))
    for step in steps:
        steps.set_description(f"Step {step + 1}")

        rng, rng_dropout = jax.random.split(rng, 2)
        train_batch = next(train_batches)
        state, batch_metrics = train_step(state, train_batch, rng_dropout)
        metrics.log_step(split="train", **batch_metrics)

        if step % eval_every == 0:
            for eval_batch in dataset_splits["valid"].as_numpy_iterator():
                batch_metrics = eval_step(state, eval_batch)
                metrics.log_step(split="valid", **batch_metrics)
            metrics.flush(step=step)

        steps.set_postfix_str(metrics.latest(["loss"]))

    return state, metrics.export()


@jax.jit
def train_step(state, batch, rng_dropout: jax.Array):
    """Run a training step and update parameters."""

    def calculate_loss(params, batch):
        """Make predictions on batch and compute binary cross-entropy loss."""
        logits, updates = state.apply_fn(
            {"params": params, "batch_stats": state.batch_stats},
            x=batch["sequences"],
            is_training=True,
            rngs={"dropout": rng_dropout},
            mutable=["batch_stats"],
        )

        loss = optax.sigmoid_binary_cross_entropy(
            logits, batch["labels"]
        ).mean()

        return loss, updates

    grad_fn = jax.value_and_grad(calculate_loss, has_aux=True)
    (loss, updates), grads = grad_fn(state.params, batch)
    state = state.apply_gradients(grads=grads)
    state = state.replace(batch_stats=updates["batch_stats"])

    metrics = {"loss": loss}

    return state, metrics


def eval_step(state, batch):
    """Evaluate model on a single batch."""
    logits = state.apply_fn(
        {"params": state.params, "batch_stats": state.batch_stats},
        x=batch["sequences"],
        is_training=False,
        mutable=False,
    )
    loss = optax.sigmoid_binary_cross_entropy(logits, batch["labels"]).mean()
    metrics = {
        "loss": loss.item(),
        **compute_metrics(batch["labels"], logits),
    }
    return metrics
