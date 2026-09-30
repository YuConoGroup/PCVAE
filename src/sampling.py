"""Shared sampling utilities for CVAE sequence generation."""

import numpy as np
from preprocessing import AMINO_ACIDS, index_to_aa_mapping


def safe_softmax(x: np.ndarray, temperature: float = 0.8) -> np.ndarray:
    """Numerically stable softmax implementation."""
    x = np.asarray(x, dtype=np.float64)
    x = x / temperature
    x_max = np.max(x)
    if np.isinf(x_max) or np.isnan(x_max):
        return np.ones_like(x) / len(x)
    exp_x = np.exp(x - x_max)
    sum_exp = np.sum(exp_x)
    if sum_exp == 0:
        return np.ones_like(x) / len(x)
    return exp_x / sum_exp


def nucleus_sampling(probs: np.ndarray, p: float = 0.9, temperature: float = 0.8) -> tuple:
    """Nucleus sampling. Returns (sampled_index, original_probability)."""
    probs = np.asarray(probs, dtype=np.float64)

    if np.any(~np.isfinite(probs)):
        probs = np.ones_like(probs) / len(probs)
    else:
        probs = np.clip(probs, 1e-12, 1.0)
        probs = probs / np.sum(probs)

    if temperature > 0.0:
        probs = safe_softmax(np.log(np.maximum(probs, 1e-12)), temperature)
    else:
        idx = np.argmax(probs)
        return idx, probs[idx]

    sorted_idx = np.argsort(probs)[::-1]
    cumulative = np.cumsum(probs[sorted_idx])
    cutoff = cumulative > p
    if np.any(cutoff):
        last_idx = np.where(cutoff)[0][0] + 1
        valid_idx = sorted_idx[:last_idx]
    else:
        valid_idx = sorted_idx

    valid_probs = probs[valid_idx]
    sum_valid = np.sum(valid_probs)
    if sum_valid > 0:
        valid_probs = valid_probs / sum_valid
    else:
        valid_probs = np.ones_like(valid_probs) / len(valid_idx)

    try:
        sampled = np.random.choice(valid_idx, p=valid_probs)
    except (ValueError, TypeError):
        sampled = valid_idx[np.argmax(valid_probs)]

    return sampled, probs[sampled]


def reconstruct_from_lstm_probs(
    probs,
    min_length: int = 10,
    max_length: int = 30,
    temperature: float = 0.8,
    top_p: float = 0.9,
    min_probability: float = 0.001,
    length_alpha: float = 0.05,
    random_terminate_prob: float = 0.03,
    target_length: int = None,
) -> str:
    """Reconstruct amino acid sequence from LSTM output probability matrix."""
    try:
        import torch
        if isinstance(probs, torch.Tensor):
            probs = probs.cpu().detach().numpy()
    except ImportError:
        pass
    if probs.ndim == 3:
        probs = probs[0]

    index_to_aa = index_to_aa_mapping(AMINO_ACIDS)
    sequence = []
    for pos_idx, pos_probs in enumerate(probs):
        if len(sequence) >= max_length:
            break

        dynamic_threshold = min_probability * (1 + length_alpha * pos_idx)

        if target_length is not None and len(sequence) >= target_length:
            dynamic_threshold *= 1.5

        aa_idx, aa_prob = nucleus_sampling(pos_probs, p=top_p, temperature=temperature)

        should_terminate = False
        if len(sequence) >= min_length:
            if aa_prob < dynamic_threshold:
                should_terminate = True
            elif __import__('random').random() < random_terminate_prob:
                should_terminate = True

        if should_terminate:
            break

        sequence.append(index_to_aa[aa_idx])

    return "".join(sequence)
