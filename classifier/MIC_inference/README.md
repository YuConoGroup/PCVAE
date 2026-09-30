# mic_prediction — MIC Value Regression

Predicts the Minimum Inhibitory Concentration (MIC) of peptides against *E. coli* using ProtBERT (protein language model) + MLP regression head.

## Files

| File | Role |
|------|------|
| `model.py` | `REG` model: ProtBERT backbone + 3-layer regressor (1024→512→128→1) with LeakyReLU + Dropout |
| `seq_dataloader.py` | `Seq_Dataset` class, tokenizer config, train/test DataLoader factories, `freeze()` utility |
| `train.py` | Training: MSE loss, AdamW optimizer, gradient clipping, tracks MSE/R²/PCC metrics, saves best model + plots |
| `inference.py` | Loads trained model, predicts MIC values for peptide sequences from CSV |
| `reproduce.py` | Reproduction: loads saved model, evaluates on test set, reports MSE/R²/PCC/Kendall-Tau |

## Input Data

| File | Columns | Description |
|------|---------|-------------|
| `data/EC_Hydrmy_mic_train.csv` | `sequence`, `SEQUENCE_space`, `EC_pMIC`, `labels`, `rand` | ~3,773 training sequences with MIC values |
| `data/EC_Hydrmy_mic_test.csv` | `sequence`, `SEQUENCE_space`, `EC_pMIC`, `labels`, `rand` | ~419 test sequences |

The model uses `SEQUENCE_space` (space-separated amino acids) as input and `EC_pMIC` as the regression target.

## Output

- `result/best_model-EC-{seed}.pth` — trained model checkpoint
- `result/mse_loss_result-EC-{seed}.csv` — per-epoch metrics
- `result/training_plot-EC-{seed}.png` — MSE/R²/PCC trend plots
- Inference: CSV with added `predicted_MIC` column

## Configuration

Paths are resolved by `config.py` from environment variables (defaults relative
to this directory): `MIC_INFERENCE_DIR` (module root), `MIC_PROTBERT_DIR`
(ProtBERT model directory), `MIC_DATA_DIR` (input CSVs), `MIC_RESULT_DIR`
(outputs) and `MIC_MODEL_PATH` (trained checkpoint).

## Usage

```bash
# Train
python train.py

# Predict MIC for new sequences
python inference.py --model <best_model-EC-0.pth> --input <sequences.csv> --output <out.csv>

# Reproduce test-set results
python reproduce.py
```

## Dependencies

- `torch`, `numpy`, `pandas`, `scikit-learn`, `scipy`, `matplotlib`, `transformers` (ProtBERT)
- Local: `model.py`, `seq_dataloader.py`
- Pretrained model: `prot_bert` (directory set by `MIC_PROTBERT_DIR` in `config.py`)
