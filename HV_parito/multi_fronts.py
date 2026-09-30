"""Compute the first N Pareto fronts (layered non-dominated sorting) with sequences.

Uses:
  - process.load_and_normalize   (normalized objectives + sequences)
  - HV_calculate.nondominated_sort_layers (vectorized layered sorting)

Outputs to HV_parito/result/:
  - pareto_layer_{k}_round{TAG}.csv   each front separately (k = 1..N, non-empty only)
  - all_fronts_top{N}_round{TAG}.csv  merged file with 'Front' column
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(os.path.dirname(BASE), "src"))  # PCVAE/src/ for the shared config

from process import load_and_normalize          # noqa: E402
from HV_calculate import nondominated_sort_layers  # noqa: E402
from config import CLASSIFIER_OUTPUT_DIR        # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round-tag", default="1")
    ap.add_argument("--mic", default=None)
    ap.add_argument("--tox", default=None)
    ap.add_argument("--aip", default=None)
    ap.add_argument("--fronts", type=int, default=10)
    ap.add_argument("--out-dir", default=None,
                    help="output directory (default: result/; when set, filenames omit the round suffix)")
    args = ap.parse_args()

    tag = args.round_tag
    cls = CLASSIFIER_OUTPUT_DIR
    mic = args.mic or f"{cls}/mic_predictions_round{tag}.csv"
    tox = args.tox or f"{cls}/tox_predictions_round{tag}.csv"
    aip = args.aip or f"{cls}/aip_round{tag}/AIP_prediction_result.csv"

    if args.out_dir:
        result_dir = args.out_dir
        tag_suffix = ""
    else:
        result_dir = os.path.join(BASE, "result")
        tag_suffix = f"_round{args.round_tag}"
    os.makedirs(result_dir, exist_ok=True)

    df = load_and_normalize(mic, tox, aip,
                            output_file=os.path.join(BASE, f"normalized_with_seq_round{tag}.csv"))
    obj_cols = ["MIC_norm", "TOXIN_norm", "AIP_norm"]
    obj = df[obj_cols].to_numpy(dtype=float)

    valid = ~np.isnan(obj).any(axis=1)
    if (~valid).any():
        print(f"Dropping {(~valid).sum()} rows containing NaN")
        df = df.loc[valid].reset_index(drop=True)
        obj = obj[valid]

    layers = nondominated_sort_layers(obj, max_layers=args.fronts)
    print(f"Extracted {len(layers)} fronts "
          f"(requested {args.fronts}; sizes: {[len(l) for l in layers]})")

    out_cols = ["Front", "Rank_in_Front", "ID", "Sequence",
                "predicted_MIC", "toxin_prob", "AIP_prob",
                "MIC_norm", "TOXIN_norm", "AIP_norm", "dist_to_ideal"]

    merged_parts = []
    for k, idx in enumerate(layers, start=1):
        sub = df.iloc[idx].copy()
        sub.insert(0, "Rank_in_Front",
                   np.arange(1, len(idx) + 1))
        sub.insert(0, "Front", k)
        sub["dist_to_ideal"] = np.linalg.norm(obj[idx], axis=1)

        layer_path = os.path.join(result_dir, f"pareto_layer_{k}{tag_suffix}.csv")
        sub.to_csv(layer_path, index=False)
        print(f"Front {k:2d}: {len(idx):5d} solutions -> {os.path.basename(layer_path)}")
        merged_parts.append(sub[out_cols])

    merged = pd.concat(merged_parts, ignore_index=True)
    merged_path = os.path.join(result_dir, f"all_fronts_top{len(layers)}{tag_suffix}.csv")
    merged.to_csv(merged_path, index=False)
    print("=" * 60)
    print(f"Merged {len(merged)} solutions across {len(layers)} fronts")
    print(f"Saved: {merged_path}")


if __name__ == "__main__":
    main()
