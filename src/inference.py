"""
Final improved CVAE model inference code - enhanced length diversity + numerical stability hardening
Main optimizations:
1. Introduce dynamic length penalty and random termination mechanism
2. Support target length-guided sampling
3. Optimize early termination logic to avoid all sequences being generated at maximum length
4. Multiple numerical stability safeguards to completely eliminate NaN
"""

from __future__ import annotations

import os
import torch
import numpy as np
import pandas as pd
import joblib
import random
import matplotlib.pyplot as plt
from model import ConditionalVAE
from preprocessing import index_to_aa_mapping, AMINO_ACIDS
from sampling import safe_softmax, nucleus_sampling, reconstruct_from_lstm_probs
from config import SAVE_DIR, GENERATED_DIR


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ------------------------- Helper functions -------------------------
def has_four_consecutive(sequence: str) -> bool:
    if len(sequence) < 4:
        return False
    count = 1
    for i in range(1, len(sequence)):
        if sequence[i] == sequence[i-1]:
            count += 1
            if count >= 4:
                return True
        else:
            count = 1
    return False


def is_valid_sequence(sequence: str, min_length: int = 10, max_length: int = 30) -> bool:
    if not sequence:
        return False
    seq_len = len(sequence)
    if seq_len < min_length or seq_len > max_length:
        return False
    if '-' in sequence:
        return False
    if 'C' in sequence:
        return False
    if has_four_consecutive(sequence):
        return False
    valid_aas = set(AMINO_ACIDS)
    for aa in sequence:
        if aa not in valid_aas:
            return False
    return True


def load_model_and_scaler(checkpoint_path: str, scaler_path: str = None, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = ConditionalVAE(
        label_dims=checkpoint.get('label_dims', [1, 1, 1, 1]),
        input_dim=checkpoint['input_dim'],
        num_hidden=checkpoint.get('num_hidden', 32),
        dropout=checkpoint.get('dropout', 0.2),
        max_len=checkpoint.get('max_len', 30),
        vocab_size=checkpoint.get('vocab_size', len(AMINO_ACIDS))
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    scaler = None
    if scaler_path and os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)

    print(f"Model loaded: {checkpoint_path}")
    print(f"  Max length: {model.max_len}, Vocab: {model.vocab_size}")
    print(f"  Latent dimension: {model.num_hidden}")
    return model, scaler, checkpoint


def generate_single_sequence(
    model: ConditionalVAE,
    conditions: tuple,
    device: torch.device,
    min_length: int = 10,
    max_length: int = 30,
    temperature: float = 1.5,
    top_p: float = 0.9,
    noise_std: float = 0.0,
    max_attempts: int = 100,
    length_alpha: float = 0.05,
    random_terminate_prob: float = 0.03,
    target_length: int = None,
) -> str | None:
    mic, a, t, aip = conditions
    model.eval()

    y1 = torch.tensor([[mic]], dtype=torch.float32).to(device)
    y2 = torch.tensor([[a]], dtype=torch.float32).to(device)
    y3 = torch.tensor([[t]], dtype=torch.float32).to(device)
    y4 = torch.tensor([[aip]], dtype=torch.float32).to(device)

    for _ in range(max_attempts):
        with torch.no_grad():
            probs = model.sample(
                num_samples=1,
                y_1=y1, y_2=y2, y_3=y3, y_4=y4,
                device=device,
                temperature=temperature
            )
            # Check if output contains anomalous values
            if torch.isnan(probs).any() or torch.isinf(probs).any():
                continue

            seq = reconstruct_from_lstm_probs(
                probs[0].cpu().numpy(),
                min_length=min_length,
                max_length=model.max_len,
                temperature=temperature,
                top_p=top_p,
                min_probability=0.001,
                length_alpha=length_alpha,
                random_terminate_prob=random_terminate_prob,
                target_length=target_length,
            )
            if is_valid_sequence(seq, min_length, max_length):
                return seq
    return None


def generate_diverse_sequences(
    model: ConditionalVAE,
    conditions: list,
    num_samples_per_condition: int = 2000,
    device: torch.device = None,
    min_length: int = 10,
    max_length: int = 30,
    temperature: float = 1.5,
    top_p: float = 0.9,
    noise_std: float = 0.0,
    max_attempts_per_seq: int = 100,
    progress_file: str = None,
    length_variation: bool = False,
    length_alpha: float = 0.05,
    random_terminate_prob: float = 0.03,
) -> tuple[list, list]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    existing_sequences = []
    existing_conditions = []
    if progress_file and os.path.exists(progress_file):
        try:
            df_existing = pd.read_csv(progress_file)
            for _, row in df_existing.iterrows():
                seq = row['Sequence']
                cond = {
                    'MIC': row['MIC'],
                    'Label_a': row['Label_a'],
                    'Label_t': row['Label_t'],
                    'AIP': row['AIP']
                }
                if is_valid_sequence(seq, min_length, max_length):
                    existing_sequences.append(seq)
                    existing_conditions.append(cond)
            print(f"Loaded {len(existing_sequences)} valid sequences from progress file")
        except Exception as e:
            print(f"Error loading progress file: {e}")

    cond_counts = {}
    for cond in existing_conditions:
        key = (cond['MIC'], cond['Label_a'], cond['Label_t'], cond['AIP'])
        cond_counts[key] = cond_counts.get(key, 0) + 1

    all_sequences = existing_sequences.copy()
    all_conditions = existing_conditions.copy()

    total_target = len(conditions) * num_samples_per_condition
    if len(all_sequences) >= total_target:
        print(f"Progress file already contains enough sequences, returning first {total_target}")
        return all_sequences[:total_target], all_conditions[:total_target]

    for cond_idx, (mic, a, t, aip) in enumerate(conditions):
        key = (mic, a, t, aip)
        existing = cond_counts.get(key, 0)
        needed = num_samples_per_condition - existing
        if needed <= 0:
            print(f"Condition {cond_idx+1}/{len(conditions)} already completed, skipping")
            continue

        print(f"Condition {cond_idx+1}/{len(conditions)}: MIC={mic:.3f}, a={a}, t={t}, AIP={aip}")
        print(f"  {existing} existing, {needed} more needed")

        generated = 0
        attempts = 0
        while generated < needed:
            attempts += 1
            if attempts > needed * max_attempts_per_seq:
                print(f"  Warning: reached maximum attempts, stopping current condition")
                break

            cur_min = min_length
            cur_max = max_length
            target_len = None
            if length_variation and generated > 0:
                cur_min = random.randint(min_length, min(15, max_length))
                cur_max = random.randint(cur_min + 3, max_length)
                target_len = random.randint(cur_min, cur_max)

            seq = generate_single_sequence(
                model=model,
                conditions=(mic, a, t, aip),
                device=device,
                min_length=cur_min,
                max_length=cur_max,
                temperature=temperature,
                top_p=top_p,
                noise_std=noise_std,
                max_attempts=max_attempts_per_seq,
                length_alpha=length_alpha,
                random_terminate_prob=random_terminate_prob,
                target_length=target_len,
            )

            if seq is None:
                continue
            if seq in all_sequences:
                continue

            all_sequences.append(seq)
            all_conditions.append({'MIC': mic, 'Label_a': a, 'Label_t': t, 'AIP': aip})
            generated += 1

            if generated % 10 == 0:
                save_progress(all_sequences, all_conditions, progress_file)
                print(f"    [{generated}/{needed}] generated")

        print(f"  Condition completed: generated {generated} new sequences")

    if progress_file:
        save_progress(all_sequences, all_conditions, progress_file)

    final_count = len(all_sequences)
    if final_count < total_target:
        print(f"Warning: final generation {final_count}, did not reach target {total_target}")
    else:
        print(f"Successfully generated target count: {final_count}")

    return all_sequences[:total_target], all_conditions[:total_target]


def save_progress(sequences: list, conditions: list, progress_file: str):
    if not progress_file or len(sequences) != len(conditions):
        return
    try:
        df = pd.DataFrame({
            'Sequence': sequences,
            'MIC': [c['MIC'] for c in conditions],
            'Label_a': [c['Label_a'] for c in conditions],
            'Label_t': [c['Label_t'] for c in conditions],
            'AIP': [c['AIP'] for c in conditions]
        })
        df['Sequence_ID'] = [f"Gen_{i+1:04d}" for i in range(len(sequences))]
        df.to_csv(progress_file, index=False)
    except Exception as e:
        print(f"Error saving progress file: {e}")


def calculate_diversity_score(sequences: list) -> float:
    if len(sequences) < 2:
        return 0.0
    return len(set(sequences)) / len(sequences)


def save_final_results(sequences: list, conditions: list, output_dir: str,
                       filename: str = "generated_sequences.csv", round_tag: str = "1"):
    os.makedirs(output_dir, exist_ok=True)

    filtered_seqs = []
    filtered_conds = []
    for seq, cond in zip(sequences, conditions):
        if '-' not in seq:
            filtered_seqs.append(seq)
            filtered_conds.append(cond)

    df = pd.DataFrame(filtered_conds)
    df['Sequence'] = filtered_seqs
    df['Length'] = [len(s) for s in filtered_seqs]
    df['Round'] = round_tag
    df['Sequence_ID'] = [f"R{round_tag}_Gen_{i+1:04d}" for i in range(len(filtered_seqs))]

    columns = ['Sequence_ID', 'Sequence', 'Length', 'Round', 'MIC', 'Label_a', 'Label_t', 'AIP']
    df = df[columns]

    output_path = os.path.join(output_dir, filename)
    df.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")
    return df


def create_condition_combinations():
    conditions = [
        (0.1, 1.0, 0.0, 1.0),
        (0.1, 1.0, 0.0, 0.0),
        (0.1, 1.0, 1.0, 0.0),
        (0.1, 1.0, 1.0, 1.0),
    ]
    return conditions


def main():
    SEED = 666
    set_all_seeds(SEED)

    ROUND = os.environ.get("GEN_ROUND", "1")
    NUM_SAMPLES_PER_CONDITION = int(os.environ.get("GEN_NUM_PER_COND", "2000"))

    CHECKPOINT_PATH = os.environ.get("CKPT_PATH", os.path.join(SAVE_DIR, "best_model.pth"))
    SCALER_PATH = os.environ.get("SCALER_PATH", os.path.join(SAVE_DIR, "mic_scaler_seed_666.pkl"))
    OUTPUT_DIR = GENERATED_DIR
    PROGRESS_FILE = os.path.join(OUTPUT_DIR, f"progress_round{ROUND}.csv")

    MIN_SEQUENCE_LENGTH = 10
    MAX_SEQUENCE_LENGTH = 30
    MAX_ATTEMPTS_PER_SEQ = 500

    TEMPERATURE = float(os.environ.get("GEN_TEMP", "1.5"))
    TOP_P = 0.9
    NOISE_STD = 0.0
    LENGTH_VARIATION = True
    LENGTH_ALPHA = 0.08
    RANDOM_TERMINATE_PROB = 0.02

    print("=== LSTM-CVAE Enhanced Length Diversity Inference (Numerically Stable Version) ===")
    print(f"Model path: {CHECKPOINT_PATH}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Length range: {MIN_SEQUENCE_LENGTH}-{MAX_SEQUENCE_LENGTH}")
    print(f"Temperature: {TEMPERATURE}, Top-p: {TOP_P}")
    print(f"Length penalty coefficient: {LENGTH_ALPHA}, Random termination probability: {RANDOM_TERMINATE_PROB}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("1. Loading model...")
    model, scaler, checkpoint = load_model_and_scaler(CHECKPOINT_PATH, SCALER_PATH, device)

    if MAX_SEQUENCE_LENGTH > model.max_len:
        print(f"Warning: MAX_SEQUENCE_LENGTH truncated to model max length {model.max_len}")
        MAX_SEQUENCE_LENGTH = model.max_len

    print("2. Creating generation conditions...")
    conditions = create_condition_combinations()
    print(f"Total {len(conditions)} condition combinations")

    print(f"3. Generating sequences ({NUM_SAMPLES_PER_CONDITION} per condition)...")
    sequences, conditions_list = generate_diverse_sequences(
        model=model,
        conditions=conditions,
        num_samples_per_condition=NUM_SAMPLES_PER_CONDITION,
        device=device,
        min_length=MIN_SEQUENCE_LENGTH,
        max_length=MAX_SEQUENCE_LENGTH,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        noise_std=NOISE_STD,
        max_attempts_per_seq=MAX_ATTEMPTS_PER_SEQ,
        progress_file=PROGRESS_FILE,
        length_variation=LENGTH_VARIATION,
        length_alpha=LENGTH_ALPHA,
        random_terminate_prob=RANDOM_TERMINATE_PROB,
    )

    if scaler is not None:
        print("4. Inverse-normalizing MIC values...")
        mic_vals = np.array([c['MIC'] for c in conditions_list]).reshape(-1, 1)
        mic_denorm = scaler.inverse_transform(mic_vals).flatten()
        for i, mic in enumerate(mic_denorm):
            conditions_list[i]['MIC'] = mic

    print("5. Saving results...")
    df = save_final_results(
        sequences=sequences,
        conditions=conditions_list,
        output_dir=OUTPUT_DIR,
        filename=f"generated_sequences_round{ROUND}.csv",
        round_tag=ROUND,
    )

    lengths = df['Length'].values
    diversity = calculate_diversity_score(sequences)
    print("" + "="*50)
    print("Generation complete!")
    print(f"Total sequences: {len(sequences)}")
    print(f"Unique sequences: {len(set(sequences))}")
    print(f"Sequence diversity: {diversity:.4f}")
    print(f"Length distribution: min={np.min(lengths)}, max={np.max(lengths)}, mean={np.mean(lengths):.2f}")
    print(f"Length frequency: {dict(zip(*np.unique(lengths, return_counts=True)))}")
    print(f"Output directory: {OUTPUT_DIR}")

    plt.figure(figsize=(8, 4))
    plt.hist(lengths, bins=range(MIN_SEQUENCE_LENGTH, MAX_SEQUENCE_LENGTH+2), edgecolor='black')
    plt.xlabel('Sequence Length')
    plt.ylabel('Count')
    plt.title('Generated Sequence Length Distribution')
    plt.savefig(os.path.join(OUTPUT_DIR, f'length_distribution_round{ROUND}.png'), dpi=150)
    plt.show()

    return df


if __name__ == "__main__":
    result_df = main()
