"""JAX/Flax tools for DNA sequence classification."""

from __future__ import annotations

from .checkpoint import (
    get_abstract_state,
    restorable,
    restore,
    restore_checkpoint,
    save_final_checkpoint,
)
from .data import (
    convert_to_tfds,
    dna_to_one_hot,
    load_dataset,
    load_dataset_splits,
    one_hot_to_dna,
    parse_fasta,
)
from .interpret import compute_input_gradient
from .logging_config import configure_logging
from .metrics import MetricsLogger, compute_metrics
from .models import (
    ConvBlock,
    ConvTransformerModel,
    MLPBlock,
    TrainStateWithBatchNorm,
    TransformerBlock,
)
from .training import eval_step, train, train_step
from .viz import DEFAULT_SPLIT_COLORS, NAMED_COLORS, plot_binding_site, to_df

__version__ = "0.1.0"

__all__ = [
    "ConvBlock",
    "ConvTransformerModel",
    "DEFAULT_SPLIT_COLORS",
    "MLPBlock",
    "MetricsLogger",
    "NAMED_COLORS",
    "TrainStateWithBatchNorm",
    "TransformerBlock",
    "compute_input_gradient",
    "compute_metrics",
    "configure_logging",
    "convert_to_tfds",
    "dna_to_one_hot",
    "eval_step",
    "get_abstract_state",
    "load_dataset",
    "load_dataset_splits",
    "one_hot_to_dna",
    "parse_fasta",
    "plot_binding_site",
    "restorable",
    "restore",
    "restore_checkpoint",
    "save_final_checkpoint",
    "to_df",
    "train",
    "train_step",
]
