"""Compute per-round hypervolume (HV) using a SINGLE global normalization across all rounds.

Difference vs run_hv.py: instead of per-round independent min-max normalization
(which makes HV non-comparable across rounds), this script pools ALL five rounds'
MIC / toxin / AIP prediction results together, computes one global min-max per
objective, and normalizes every round with the SAME scaler. HV is then comparable.

Sources (excludes any files under output/classifier/old/):
  round 1 : output/classifier/aip_round1/{mic,tox}_predictions_round1.csv + AIP_prediction_result.csv
  round 2 : output/classifier/aip_round2/{...}
  round 3 : output/classifier/{mic,tox}_predictions_round3.csv + aip_round3/AIP_prediction_result.csv
  round 4 : output/classifier/aip_round4/{...}
  round 5 : output/classifier/{mic,tox}_predictions_round5.csv + aip_round5/AIP_prediction_result.csv

Outputs saved to PCVAE/output/:
  hv_global_summary.txt   : per-round global-normalized Pareto size + HV
  global_norm_params.txt  : the global min/max used per objective
  normalized_round{1..5}.csv : each round's globally-normalized points
"""
import os
import sys
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(os.path.dirname(BASE), "src"))  # PCVAE/src/ for the shared config

from HV_calculate import HyperVolume, nondominated_sort_layers  # noqa: E402
from config import CLASSIFIER_OUTPUT_DIR, OUTPUT_DIR            # noqa: E402

CLS = CLASSIFIER_OUTPUT_DIR
OUT_DIR = OUTPUT_DIR

# round -> (mic_csv, tox_csv, aip_csv)
ROUND_FILES = {
    1: (f"{CLS}/aip_round1/mic_predictions_round1.csv",
        f"{CLS}/aip_round1/tox_predictions_round1.csv",
        f"{CLS}/aip_round1/AIP_prediction_result.csv"),
    2: (f"{CLS}/aip_round2/mic_predictions_round2.csv",
        f"{CLS}/aip_round2/tox_predictions_round2.csv",
        f"{CLS}/aip_round2/AIP_prediction_result.csv"),
    3: (f"{CLS}/mic_predictions_round3.csv",
        f"{CLS}/tox_predictions_round3.csv",
        f"{CLS}/aip_round3/AIP_prediction_result.csv"),
    4: (f"{CLS}/aip_round4/mic_predictions_round4.csv",
        f"{CLS}/aip_round4/tox_predictions_round4.csv",
        f"{CLS}/aip_round4/AIP_prediction_result.csv"),
    5: (f"{CLS}/mic_predictions_round5.csv",
        f"{CLS}/tox_predictions_round5.csv",
        f"{CLS}/aip_round5/AIP_prediction_result.csv"),
}

REF_POINT = [1.0, 1.0, 1.0]


def load_round(round_idx):
    mic_f, tox_f, aip_f = ROUND_FILES[round_idx]
    mic = pd.read_csv(mic_f)["predicted_MIC"].astype(float).to_numpy()
    tox = pd.read_csv(tox_f)["Prediction_Probability"].astype(float).to_numpy()
    aip = pd.read_csv(aip_f)["probability"].astype(float).to_numpy()
    n = len(mic)
    if not (len(tox) == len(aip) == n):
        raise ValueError(f"round {round_idx}: length mismatch mic={n} tox={len(tox)} aip={len(aip)}")
    return {"mic": mic, "tox": tox, "aip": aip, "aip_neg": -aip}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1. Load all rounds
    data = {r: load_round(r) for r in sorted(ROUND_FILES)}

    # 2. Global min/max per objective across ALL rounds pooled together
    all_mic = np.concatenate([data[r]["mic"] for r in sorted(data)])
    all_tox = np.concatenate([data[r]["tox"] for r in sorted(data)])
    all_aip_neg = np.concatenate([data[r]["aip_neg"] for r in sorted(data)])

    g_min = {"MIC": all_mic.min(), "TOXIN": all_tox.min(), "AIP": all_aip_neg.min()}
    g_max = {"MIC": all_mic.max(), "TOXIN": all_tox.max(), "AIP": all_aip_neg.max()}
    print("Global normalization parameters (per objective, pooled over 5 rounds):")
    for k in ["MIC", "TOXIN", "AIP"]:
        print(f"  {k:6s} min={g_min[k]:.6f}  max={g_max[k]:.6f}  (range={g_max[k]-g_min[k]:.6f})")

    with open(os.path.join(OUT_DIR, "global_norm_params.txt"), "w") as f:
        f.write("Global min-max normalization parameters (pooled across rounds 1-5)\n")
        f.write("NOTE: AIP is negated (minimization direction): aip_neg = -probability\n")
        f.write(f"reference_point: {REF_POINT}\n")
        for k in ["MIC", "TOXIN", "AIP"]:
            f.write(f"{k}: min={g_min[k]:.6f}, max={g_max[k]:.6f}, range={g_max[k]-g_min[k]:.6f}\n")

    def normalize(raw, key):
        rng = g_max[key] - g_min[key]
        return 0.0 * raw if rng < 1e-12 else (raw - g_min[key]) / rng

    # 3. Normalize each round & compute HV
    lines = []
    lines.append("Per-round Hypervolume using GLOBAL normalization (comparable across rounds)")
    lines.append("Reference point: [1,1,1] | lower is better for all 3 objectives")
    results = {}
    print("\n" + "=" * 70)
    print(f"{'round':>6} {'n':>7} {'pareto':>8} {'HV':>10}")
    print("=" * 70)
    for r in sorted(data):
        mic_n = normalize(data[r]["mic"], "MIC")
        tox_n = normalize(data[r]["tox"], "TOXIN")
        aip_n = normalize(data[r]["aip_neg"], "AIP")
        pts = np.column_stack([mic_n, tox_n, aip_n])
        pts = pts[~np.isnan(pts).any(axis=1)]

        out_csv = os.path.join(OUT_DIR, f"normalized_round{r}.csv")
        pd.DataFrame(pts, columns=["MIC_norm", "TOXIN_norm", "AIP_norm"]).to_csv(out_csv, index=False)

        # Vectorized first-front extraction (fast on ~20k points), then HV on the front
        layers = nondominated_sort_layers(pts, max_layers=1)
        front = pts[layers[0]] if layers else pts[:0]
        hv_calc = HyperVolume(front, REF_POINT)
        hv = hv_calc.compute(use_pareto_front=False)
        results[r] = {"n": len(pts), "pareto": len(front), "hv": hv}
        print(f"{r:>6} {len(pts):>7} {len(front):>8} {hv:.6f}")
        lines.append(f"round {r}: n={len(pts)}, pareto_front_size={len(front)}, HV={hv:.6f}")

    # 4. Save summary
    summary_path = os.path.join(OUT_DIR, "hv_global_summary.txt")
    with open(summary_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write(" PCVAE per-round Hypervolume (global normalization across rounds 1-5)\n")
        f.write("=" * 70 + "\n")
        f.write(f"reference_point: {REF_POINT}\n")
        f.write(f"objectives: MIC (lower-better), TOXIN (lower-better), AIP (higher-better -> negated)\n")
        f.write(f"normalization: global min-max pooled over all rounds (see global_norm_params.txt)\n")
        f.write("=" * 70 + "\n")
        for line in lines:
            f.write(line + "\n")
        f.write("=" * 70 + "\n")
    print("\n" + "=" * 70)
    print(f"Saved summary: {summary_path}")
    print(f"Saved per-round normalized points: {OUT_DIR}/normalized_round{{1..5}}.csv")
    print("=" * 70)


if __name__ == "__main__":
    main()
