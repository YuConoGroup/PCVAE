# tox_classifier — Toxicity Classifier

Binary classification: predicts whether a peptide sequence is toxic. Uses ESM-2 t12_35M embeddings + CNN-BiLSTM-CBAM classifier (same architecture as amp_classifier).

## Files

| File | Role |
|------|------|
| `model.py` | Model architectures: `ProteinDataset`, `ChannelAttention`, `SpatialAttention`, `CBAM`, `stage_1model` (CNN+BiLSTM+CBAM → 2-class) |
| `extract_features.py` | Training-time feature extraction: loads ESM-2 t12_35M, extracts last-4-layer hidden states, applies PCA (95% variance), saves features + PCA model |
| `extract_features_inference.py` | Inference-time feature extraction: loads pre-trained PCA model, extracts features for new CSV of sequences |
| `train.py` | 5-fold CV training (100 epochs/fold), ROC/confusion matrix/loss curve plots, best-model by AUC |
| `inference.py` | Ensemble inference: loads all 5 fold models, applies standardization, outputs probabilities + confidence levels |

## Input Data

| File | Columns | Description |
|------|---------|-------------|
| `data/ToxinPred3.0 all.csv` | `sequence`, `toxicity`, `rand` | ~11,037 sequences with toxicity labels (0/1) |

## Output

- `features/features.pkl` — PCA-reduced embeddings + labels
- `features/pca_model.pkl` — fitted PCA model
- `scaler.pkl` — StandardScaler
- `save/save_model/stage-1model_fold_{N}_best.pth` — per-fold checkpoints
- `save/kfold_results.csv`, `save/mean_metrics.csv` — CV metrics
- `save/picture/` — ROC, confusion matrix, loss curves
- Inference: `predictions_a.csv` with probability/confidence columns

## Usage

```bash
# Step 1: Extract features from training data
python extract_features.py

# Step 2: Train 5-fold classifier
python train.py

# Step 3: Extract features for new sequences (inference mode)
python extract_features_inference.py

# Step 4: Run ensemble inference
python inference.py
```

## Configuration

Paths are resolved by `config.py` from environment variables (defaults relative
to this directory): `TOX_CLASSIFIER_DIR` (module root), `TOX_ESM2_DIR` (ESM-2
model directory), `TOX_DATA_DIR` (input CSVs), `TOX_FEATURES_DIR` (PCA-reduced
features) and `TOX_RESULT_DIR` (inference outputs).

## Dependencies

- `torch`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `tqdm`, `joblib`, `pickle`
- `transformers` (ESM-2), HuggingFace model: `esm2_t12_35M_UR50D`
- Local: `model.py`
