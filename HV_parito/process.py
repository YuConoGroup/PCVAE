import pandas as pd
import numpy as np


def normalize_minmax(series):
    """Min-max normalize a 1D array to [0, 1], ignoring NaN."""
    arr = series.values.astype(float)
    min_val = np.nanmin(arr)
    max_val = np.nanmax(arr)
    if max_val - min_val == 0:
        return np.zeros_like(arr)  # constant input -> all zeros
    return (arr - min_val) / (max_val - min_val)


def _pick_col(df, candidates):
    """Return the first existing column name from ``candidates``, else None."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def load_and_normalize(mic_file, tox_file, aip_file,
                       id_col='sequence', output_file='normalized_with_seq.csv',
                       keep_raw=True):
    """Read three prediction CSVs, normalize the objective columns and keep the
    sequence identifier.

    Parameters
    ----------
    mic_file : MIC prediction CSV, must contain a 'predicted_MIC' column.
    tox_file : toxicity prediction CSV, must contain a
        'Prediction_Probability' column.
    aip_file : AIP prediction CSV, must contain a 'probability' column.
    id_col : column used as the identifier (default 'sequence'; the loader will
        try [id_col, 'Sequence_ID', 'Sequence'] in order and fall back to the
        row index if none exist).
    output_file : output CSV filename.
    keep_raw : whether to keep the raw prediction columns in the result.

    Returns
    -------
    result_df : DataFrame with the ID/sequence and the three normalized values
        (plus the raw values), also written to ``output_file``.
    """
    # 1. Read the files.
    df_mic = pd.read_csv(mic_file)
    df_tox = pd.read_csv(tox_file)
    df_aip = pd.read_csv(aip_file)

    assert len(df_mic) == len(df_tox) == len(df_aip), \
        f"row count mismatch: mic={len(df_mic)}, tox={len(df_tox)}, aip={len(df_aip)}"

    # 2. Extract raw values (the three files are assumed to share row order).
    mic_raw = df_mic['predicted_MIC'].values.astype(float)
    tox_raw = df_tox['Prediction_Probability'].values.astype(float)
    aip_raw = df_aip['probability'].values.astype(float)

    # 3. Negate AIP to turn it into a minimization objective.
    aip_neg = -aip_raw

    # 4. Min-max normalize each objective independently.
    mic_norm = normalize_minmax(pd.Series(mic_raw))
    tox_norm = normalize_minmax(pd.Series(tox_raw))
    aip_norm = normalize_minmax(pd.Series(aip_neg))

    # 5. Sequence identifier and sequence itself (case-insensitive fallback).
    id_name = _pick_col(df_mic, [id_col, 'Sequence_ID', 'Sequence'])
    seq_ids = df_mic[id_name].values if id_name else np.arange(len(df_mic))

    seq_name = _pick_col(df_mic, ['Sequence', 'sequence'])
    seqs = df_mic[seq_name].values if seq_name else np.full(len(df_mic), '', dtype=object)

    # 6. Build the result DataFrame and save it (normalized + sequence + raw).
    result_df = pd.DataFrame({
        'ID': seq_ids,
        'Sequence': seqs,
        'MIC_norm': mic_norm,
        'TOXIN_norm': tox_norm,
        'AIP_norm': aip_norm,
    })
    if keep_raw:
        result_df['predicted_MIC'] = mic_raw
        result_df['toxin_prob'] = tox_raw
        result_df['AIP_prob'] = aip_raw

    result_df.to_csv(output_file, index=False)
    print(f"Normalized result saved to: {output_file}")
    return result_df


# ========= Example usage =========
if __name__ == "__main__":
    # Replace with the actual file paths.
    mic_csv = "mic_predictions.csv"
    tox_csv = "tox_predictions.csv"
    aip_csv = "aip_predictions.csv"

    result = load_and_normalize(mic_csv, tox_csv, aip_csv,
                                id_col='sequence',
                                output_file='normalized_with_seq.csv')

    print("\nShape of the normalized 3-objective array:", result[['MIC_norm', 'TOXIN_norm', 'AIP_norm']].shape)
    print("Statistics per objective:")
    print("MIC_norm: min={:.4f}, max={:.4f}, mean={:.4f}".format(
        result['MIC_norm'].min(), result['MIC_norm'].max(), result['MIC_norm'].mean()))
    print("TOXIN_norm: min={:.4f}, max={:.4f}, mean={:.4f}".format(
        result['TOXIN_norm'].min(), result['TOXIN_norm'].max(), result['TOXIN_norm'].mean()))
    print("AIP_norm: min={:.4f}, max={:.4f}, mean={:.4f}".format(
        result['AIP_norm'].min(), result['AIP_norm'].max(), result['AIP_norm'].mean()))
