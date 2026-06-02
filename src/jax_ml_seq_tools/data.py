"""Dataset loading and DNA sequence encoding utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import tensorflow as tf


def dna_to_one_hot(dna_sequence: str) -> np.ndarray:
    """Convert DNA into a one-hot encoded format with channel ordering ACGT."""
    base_to_one_hot = {
        "A": (1, 0, 0, 0),
        "C": (0, 1, 0, 0),
        "G": (0, 0, 1, 0),
        "T": (0, 0, 0, 1),
        "N": (1, 1, 1, 1),  # N represents any unknown or ambiguous base.
    }
    one_hot_encoded = np.array([base_to_one_hot[base] for base in dna_sequence])
    return one_hot_encoded


def one_hot_to_dna(one_hot_sequence: np.ndarray) -> str:
    """Convert a one-hot encoded DNA sequence to a DNA string."""
    base_map = {
        (1, 0, 0, 0): "A",
        (0, 1, 0, 0): "C",
        (0, 0, 1, 0): "G",
        (0, 0, 0, 1): "T",
        (1, 1, 1, 1): "N",
    }
    dna_sequence = []
    for base_one_hot in one_hot_sequence:
        dna_sequence.append(base_map[tuple(base_one_hot)])
    return "".join(dna_sequence)


def load_dataset(sequence_db) -> dict[str, np.ndarray]:
    """Load sequences and labels from a CSV into numpy arrays."""
    df = pd.read_csv(sequence_db)
    max_seq_len = max(len(seq) for seq in df["sequence"])
    x_train_list = []
    for seq in df["sequence"]:
        padded_seq = seq.ljust(max_seq_len, "N")
        x_train_list.append(dna_to_one_hot(padded_seq))
    return {
        "labels": df["label"].values[:, None],
        "sequences": np.array(x_train_list),
    }


def convert_to_tfds(
    dataset, batch_size: int | None = None, is_training: bool = False
):
    """Convert DNA sequences and labels to a TensorFlow dataset."""
    ds = tf.data.Dataset.from_tensor_slices(dataset)
    if is_training:
        ds = ds.shuffle(buffer_size=len(dataset["sequences"]))
        ds = ds.repeat()
    batch_size = batch_size or len(dataset["labels"])
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds


def load_dataset_splits(path, batch_size: int | None = None):
    """Load TF dataset splits (train, valid) as TensorFlow datasets."""
    dataset_splits = {}
    for split in ["train", "val"]:
        dataset = load_dataset(sequence_db=f"{path}/{split}.csv")
        key = "valid" if split == "val" else split
        current_batch_size = batch_size if split == "train" else None
        ds = convert_to_tfds(
            dataset, current_batch_size, is_training=(split == "train")
        )
        dataset_splits.update({key: ds})
    return dataset_splits


def parse_fasta(fasta_string: str) -> dict[str, str]:
    """Parse a FASTA-formatted string into a {id: sequence} dictionary."""
    sequences: dict[str, str] = {}
    current_id = None
    current_sequence: list[str] = []

    for line in fasta_string.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">") or line.startswith(";"):
            if current_id and current_sequence:
                sequences[current_id] = "".join(current_sequence)
            current_id = line[1:].split()[0]
            current_sequence = []
        else:
            current_sequence.append(line)

    if current_id and current_sequence:
        sequences[current_id] = "".join(current_sequence)
    return sequences
