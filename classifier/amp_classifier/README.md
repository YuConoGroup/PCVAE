# amp_classifier — Antimicrobial Peptide Classifier

Binary classification: predicts whether a peptide sequence has antimicrobial activity. Uses ESM-2 protein language model embeddings + CNN-BiLSTM-CBAM classifier.

## Files

| File | Role |
|------|------|
| `model.py` | Model architectures: `ProteinDataset`, `ChannelAttention`, `SpatialAttention`, `CBAM`, `stage_1model` (CNN+BiLSTM+CBAM → 2-class) |
| `extract_features.py` | Feature extraction: loads ESM-2 t6_8M, extracts last-4-layer hidden states, applies PCA (95% variance), saves as `features.pkl` |
| `train.py` | 5-fold CV training with tqdm progress bars, ROC/confusion matrix/loss curve plots, best-model tracking by AUC |
| `ensemble_predict.py` | Ensemble prediction: loads all 5 CV-fold models, averages predictions for new sequences |
| `inference.py` | Full inference pipeline: ESM-2 extraction → PCA → standardization → classification → timing report + visualizations |

## Input Data

| File | Columns | Description |
|------|---------|-------------|
| `data/amp_balanced_processing.csv` | `aa_seq`, `AMP`, `length`, `rand` | ~22,113 peptide sequences with AMP labels (0/1) |

## Output

- `features.pkl` — PCA-reduced ESM-2 embeddings
- `scaler.pkl` — StandardScaler parameters
- `stage-1model_fold_{N}_best.pth` — per-fold model checkpoints
- `save/kfold_results.csv`, `save/mean_metrics.csv` — CV metrics
- `save/picture/` — ROC, confusion matrix, loss curve plots
- Predicted CSVs with `Predicted_Label_a` column

## Usage

```bash
# Step 1: Extract ESM-2 features (requires HuggingFace model)
python extract_features.py

# Step 2: Train classifier with 5-fold CV
python train.py

# Step 3: Inference on new sequences
python inference.py
```

## Configuration

Paths are resolved by `config.py` from environment variables (defaults relative
to this directory): `AMP_CLASSIFIER_DIR` (module root), `AMP_ESM2_DIR` (ESM-2
model directory), `AMP_DATA_DIR` (input CSVs), `AMP_WEIGHTS_DIR` (PCA / scaler /
fold checkpoints) and `AMP_RESULT_DIR` (inference outputs).

## Dependencies

- `torch`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `tqdm`, `pickle`
- `transformers` (ESM-2), HuggingFace model: `esm2_t6_8M_UR50D`
- Local: `model.py`
