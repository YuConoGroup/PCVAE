"""AMP classifier local inference.

Pipeline (must match training exactly):
  ESM2-t6 last-4-layer mean over ALL padded positions (max_len=100)
  -> pca_model.pkl -> scaler_a.pkl -> ensemble of 5 CV-fold models

Usage:
  python amp_inference_local.py --csv <input.csv> --out <output.csv>
"""
import argparse
import os

import numpy as np
import pandas as pd
import pickle
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoConfig, AutoModel

BASE = os.path.dirname(os.path.abspath(__file__))
FW = os.path.join(BASE, "feature_and_weight")
MAX_LEN = 100
BATCH = 128


def extract_features(sequences, tokenizer, model, device):
    feats = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(sequences), BATCH):
            batch = sequences[i:i + BATCH]
            inputs = tokenizer(
                list(batch),
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=MAX_LEN,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            hs = outputs.hidden_states              # tuple: [13, B, L, 320]
            feat = torch.stack(list(hs[-4:])).mean(0).mean(dim=1)  # [B, 320]
            feats.append(feat.cpu().numpy())
    return np.vstack(feats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    df = pd.read_csv(args.csv)
    seq_col = "Sequence" if "Sequence" in df.columns else df.columns[1]
    id_col = "Sequence_ID" if "Sequence_ID" in df.columns else None
    sequences = df[seq_col].astype(str).tolist()
    print(f"{len(sequences)} sequences from {args.csv}")

    local_esm = os.path.join(BASE, "esm2_t6_8M_UR50D")
    tokenizer = AutoTokenizer.from_pretrained(local_esm, do_lower_case=False)
    config = AutoConfig.from_pretrained(local_esm, output_hidden_states=True)
    esm = AutoModel.from_pretrained(local_esm, config=config).to(device).eval()

    print("Extracting ESM2 features...")
    features = extract_features(sequences, tokenizer, esm, device)
    print("Raw features:", features.shape)

    with open(os.path.join(FW, "pca_model.pkl"), "rb") as f:
        pca = pickle.load(f)
    features = pca.transform(features)
    print(f"After PCA: {features.shape}")

    with open(os.path.join(FW, "scaler_a.pkl"), "rb") as f:
        scaler = pickle.load(f)
    features = scaler.transform(features)

    tensor = torch.tensor(features, dtype=torch.float32)
    loader = DataLoader(TensorDataset(tensor), batch_size=BATCH, shuffle=False)

    models = []
    for fold in range(1, 6):
        m = __import__("model").stage_1model(features.shape[1], features.shape[1])
        sd = torch.load(os.path.join(FW, f"stage-1model_fold_{fold}_best.pth"),
                        map_location=device, weights_only=True)
        m.load_state_dict(sd)
        m.to(device).eval()
        models.append(m)
    print(f"Loaded {len(models)} fold models")

    all_probs = []
    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            probs = torch.softmax(models[0](batch_x), dim=1)[:, 1]
            for m in models[1:]:
                probs = probs + torch.softmax(m(batch_x), dim=1)[:, 1]
            all_probs.append((probs / len(models)).cpu().numpy())
    avg_probs = np.concatenate(all_probs)
    labels = (avg_probs >= 0.5).astype(int)

    out = pd.DataFrame({
        "Sequence_ID": df[id_col] if id_col else range(1, len(df) + 1),
        "Sequence": sequences,
        "Length": [len(s) for s in sequences],
        "AMP_prob": np.round(avg_probs, 4),
        "Predicted_Label": labels,
    })
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Saved: {args.out}")
    print(f"AMP ratio: {labels.mean():.4f} | prob mean: {avg_probs.mean():.4f} "
          f"| min: {avg_probs.min():.4f} | max: {avg_probs.max():.4f}")


if __name__ == "__main__":
    main()
