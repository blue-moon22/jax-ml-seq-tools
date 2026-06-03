"""Flax model components and TrainState for sequence classification."""

from __future__ import annotations

import flax.linen as nn
import jax
from flax.training.train_state import TrainState


class TrainStateWithBatchNorm(TrainState):
    """TrainState with batch norm stats and RNG key."""

    batch_stats: dict | None
    key: jax.Array

class TransformerBlock(nn.Module):
    """Transformer block with self-attention and MLP."""

    num_heads: int = 8
    dense_units: int = 64
    dropout_rate: float = 0.2

    @nn.compact
    def __call__(self, x, is_training: bool = True):
        residual = x
        x = nn.LayerNorm()(x)
        x = nn.SelfAttention(num_heads=self.num_heads)(x)
        x += residual

        residual = x
        x = nn.LayerNorm()(x)
        x = nn.Dense(self.dense_units)(x)
        x = nn.gelu(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not is_training)
        x = nn.Dense(residual.shape[-1])(x)
        x += residual
        return x

class TransformerModel(nn.Module):
    """Model combining transformer blocks."""

    dense_units: int = 128
    dropout_rate: float = 0.2
    num_transformer_blocks: int = 1
    num_transformer_heads: int = 8
    transformer_dense_units: int = 64

    @nn.compact
    def __call__(self, x, is_training: bool = True):
        for _ in range(self.num_transformer_blocks):
            x = TransformerBlock(
                num_heads=self.num_transformer_heads,
                dense_units=self.transformer_dense_units,
                dropout_rate=self.dropout_rate,
            )(x, is_training)

        return nn.Dense(1)(x)

    def create_train_state(self, rng: jax.Array, dummy_input, tx):
        """Initializes model parameters and returns a train state for training."""
        rng, rng_init, rng_dropout = jax.random.split(rng, 3)
        variables = self.init(rng_init, dummy_input)
        state = TrainStateWithBatchNorm.create(
            apply_fn=self.apply,
            tx=tx,
            params=variables["params"],
            batch_stats=variables.get("batch_stats"),
            key=rng_dropout,
        )
        return state
