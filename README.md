# PCVAE — Conditional Variational Autoencoder

Core generative model for antimicrobial peptide (AMP) design. Generates novel
peptide sequences conditioned on four molecular properties (normalized MIC,
antimicrobial activity, toxicity, anti-inflammatory activity) and provides the
multi-objective selection utilities (Pareto front, hypervolume).

## Layout

```
PCVAE/
├── src/                 # generator package
├── classifier/          # property predictors used for scoring
├── HV_parito/           # Pareto front / non-dominated sorting / hypervolume
└── README.md
```

## src/ — generator

| File | Role |
|------|------|
| `preprocessing.py` | Amino-acid one-hot encoding, index mapping (20 standard AAs, `max_len=30`) |
| `model.py` | `ConditionalVAE`: MLP encoder → 32-D latent space → 2-layer LSTM decoder, four property heads; supports temperature / nucleus sampling |
| `sampling.py` | Nucleus / temperature sampling and reconstruction from LSTM probability outputs |
| `train.py` | Loss (`cvae_total_loss`) and training loop (`train_cvae`): reconstruction CE + KLD + MSE + 3×BCE, KL annealing, teacher-forcing decay, gradient clipping, checkpointing, training curves |
| `pipeline.py` | Training entry point: load pre-split datasets → encode → train → evaluate → generate |
| `inference.py` | Standalone generation with enhanced length diversity (dynamic threshold, random termination) |
| `config.py` | Path configuration (environment-variable overridable) |

### Configuration

Paths are resolved by `src/config.py` from environment variables (defaults
relative to `PCVAE/`, the parent of `src/`):

- `PCVAE_ROOT` — project root (default: parent of `src/`)
- `PCVAE_DATA_DIR` — training / validation / test CSVs (`train.csv`, `valid.csv`, `test.csv`)
- `PCVAE_OUTPUT_DIR` — property-predictor output directory
- `PCVAE_GENERATED_DIR` — generated sequences (default directory name `genarated/`)

### Input data

CSV files with columns: `Sequence`, `predicted_MIC`, `Predicted_Label_a`,
`Predicted_Label_t`, `predicted_AIP`.

### Output

- Model checkpoints: `<PCVAE_SAVE_DIR>/<round>/best_model*.pth`
- Generated sequences: `<PCVAE_GENERATED_DIR>/generated_sequences_round*.csv`
- Training curves: `<PCVAE_SAVE_DIR>/<round>/png/training_curve.png`
- MIC normalizer: `<PCVAE_SAVE_DIR>/<round>/mic_scaler_seed_<seed>.pkl`

### Usage

```bash
# Train + evaluate + generate
python src/pipeline.py

# Generate from a trained checkpoint
CKPT_PATH=<best_model.pth> SCALER_PATH=<mic_scaler_seed_666.pkl> python src/inference.py
```

## classifier/ — property predictors

| Directory | Model |
|-----------|-------|
| `amp_classifier/` | antimicrobial activity (ESM-2 + CNN/BiLSTM/CBAM, 5-fold ensemble) |
| `tox_classifier/` | toxicity (ESM-2 + CNN/BiLSTM/CBAM, 5-fold ensemble) |
| `MIC_inference/` | MIC regression (ProtBERT + MLP head) |

Each directory has its own `config.py` and `README.md`; see those for details.

## HV_parito/ — multi-objective analysis

| File | Role |
|------|------|
| `run_hv.py` | Per-round normalization, Pareto front and hypervolume |
| `hv_global.py` | Globally-normalized, cross-round hypervolume |
| `HV_calculate.py` | `Individual`, non-dominated sorting, `HyperVolume` |
| `multi_fronts.py` | Layered non-dominated sorting (first N fronts) |
| `process.py` | MIC / toxicity / AIP normalization with sequences |
| `HV3D2.py` | 3D Pareto front and 2D projection figures |

## Dependencies

`torch`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `joblib`
