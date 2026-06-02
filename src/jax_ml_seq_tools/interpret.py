"""Model interpretability utilities."""

from __future__ import annotations

import jax
import jax.numpy as jnp


@jax.jit
def compute_input_gradient(state, sequence):
    """Compute input gradient for a one-hot DNA sequence."""
    if len(sequence.shape) != 2:
        raise ValueError("Input must be a single one-hot encoded DNA sequence.")

    sequence = jnp.asarray(sequence, dtype=jnp.float32)[None, :]

    def predict(sequence):
        return jnp.mean(
            state.apply_fn(
                {
                    "params": state.params,
                    "batch_stats": state.batch_stats,
                },
                sequence,
                is_training=False,
            )
        )

    gradient = jax.grad(lambda x: predict(x))(sequence)
    return jnp.squeeze(gradient)
