"""Normalize MIC/toxin/AIP prediction results and compute hypervolume (HV).

Steps:
1. process.load_and_normalize  -> normalized 3-objective points
2. save normalized points to HV_parito/input/
3. extract Pareto front + compute HV -> HV_parito/result/
"""
import argparse
import os
import sys

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from process import load_and_normalize  # noqa: E402
from HV_calculate import HyperVolume  # noqa: E402


def fast_pareto_front(points: np.ndarray) -> np.ndarray:
    """Vectorized non-dominated (minimization) front extraction, O(n^2) but numpy-batched."""
    n = len(points)
    leq = np.all(points[:, None, :] <= points[None, :, :], axis=2)          # (n, n)
    strict = np.all(points[:, None, :] < points[None, :, :], axis=2)        # (n, n)
    dominates = leq & (strict | ~np.eye(n, dtype=bool))
    dominated = dominates.any(axis=0)
    return points[~dominated]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mic", required=True, help="MIC predictions csv (column predicted_MIC)")
    ap.add_argument("--tox", required=True, help="toxin predictions csv (column Prediction_Probability)")
    ap.add_argument("--aip", required=True, help="AIP predictions csv (column probability)")
    ap.add_argument("--round-tag", default=os.environ.get("GEN_ROUND", "1"))
    args = ap.parse_args()

    input_dir = os.path.join(BASE, "input")
    result_dir = os.path.join(BASE, "result")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    print("Normalizing prediction results...")
    result_df = load_and_normalize(args.mic, args.tox, args.aip)
    if hasattr(result_df, "columns"):
        obj_cols = ["MIC_norm", "TOXIN_norm", "AIP_norm"]
        missing = [c for c in obj_cols if c not in result_df.columns]
        if missing:
            raise ValueError(f"normalized dataframe missing columns: {missing}")
        points = result_df[obj_cols].to_numpy(dtype=float)
    else:
        points = np.asarray(result_df, dtype=float)
    print(f"Raw points shape: {points.shape}")

    mask = ~np.isnan(points).any(axis=1)
    if (~mask).any():
        print(f"Dropping {(~mask).sum()} rows containing NaN")
        points = points[mask]

    norm_path = os.path.join(input_dir, f"normalized_points_round{args.round_tag}.csv")
    np.savetxt(norm_path, points, delimiter=",",
               header="MIC_norm,TOXIN_norm,AIP_norm", comments="")
    print(f"Normalized points saved: {norm_path}")

    print("Extracting Pareto front...")
    pareto = fast_pareto_front(points)
    print(f"Pareto front size: {len(pareto)}")

    ref_point = [1.0, 1.0, 1.0]
    hv_calculator = HyperVolume(pareto, ref_point)
    hv_value = hv_calculator.compute(use_pareto_front=False)

    pareto_path = os.path.join(result_dir, f"pareto_front_round{args.round_tag}.csv")
    np.savetxt(pareto_path, pareto, delimiter=",",
               header="MIC_norm,TOXIN_norm,AIP_norm", comments="")

    result_path = os.path.join(result_dir, f"hv_result_round{args.round_tag}.txt")
    with open(result_path, "w") as f:
        f.write(f"round: {args.round_tag}\n")
        f.write(f"num_samples: {len(points)}\n")
        f.write(f"pareto_front_size: {len(pareto)}\n")
        f.write(f"reference_point: {ref_point}\n")
        f.write(f"hypervolume: {hv_value:.6f}\n")

    print("=" * 50)
    print(f"HV = {hv_value:.6f}")
    print(f"Pareto front saved: {pareto_path}")
    print(f"Result saved: {result_path}")


if __name__ == "__main__":
    main()
