"""Sequence preprocessing helpers extracted from the notebook."""

from __future__ import annotations

import numpy as np


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def build_aa_to_index(amino_acids: str = AMINO_ACIDS) -> dict[str, int]:
    return {aa: idx for idx, aa in enumerate(amino_acids)}


def index_to_aa_mapping(amino_acids: str = AMINO_ACIDS) -> dict[int, str]:
    aa_to_index = build_aa_to_index(amino_acids)
    return {idx: aa for aa, idx in aa_to_index.items()}


def one_hot_encode_sequence(
    sequence: str,
    max_length: int,
    amino_acids: str = AMINO_ACIDS,
) -> np.ndarray:
    aa_to_index = build_aa_to_index(amino_acids)
    one_hot = np.zeros((len(sequence), len(amino_acids)), dtype=int)

    for i, aa in enumerate(sequence):
        if aa in aa_to_index:
            one_hot[i, aa_to_index[aa]] = 1

    flattened = one_hot.flatten()
    target_length = max_length * len(amino_acids)
    if len(flattened) < target_length:
        flattened = np.pad(flattened, (0, target_length - len(flattened)), "constant")
    elif len(flattened) > target_length:
        flattened = flattened[:target_length]
    return flattened


def build_one_hot_encoder(
    max_length: int,
    amino_acids: str = AMINO_ACIDS,
):
    def encoder(sequence: str) -> np.ndarray:
        return one_hot_encode_sequence(sequence, max_length=max_length, amino_acids=amino_acids)

    return encoder
