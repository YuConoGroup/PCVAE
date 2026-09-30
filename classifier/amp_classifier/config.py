"""Central path configuration for the AMP (antimicrobial) classifier.

Defaults are relative to this file so the module runs from a plain checkout.
Override with environment variables to use external data / pretrained models.

Environment variables
---------------------
AMP_CLASSIFIER_DIR  Module root (default: this directory).
AMP_ESM2_DIR        Local ESM-2 t6_8M model directory.
AMP_DATA_DIR        Input CSV directory.
AMP_WEIGHTS_DIR     PCA / scaler / fold-checkpoint directory.
AMP_RESULT_DIR      Inference output directory.
"""

import os
from pathlib import Path

BASE_DIR = Path(os.environ.get("AMP_CLASSIFIER_DIR", Path(__file__).resolve().parent))

ESM2_DIR = Path(os.environ.get("AMP_ESM2_DIR", BASE_DIR / "esm2_t6_8M_UR50D"))
DATA_DIR = Path(os.environ.get("AMP_DATA_DIR", BASE_DIR / "data"))
WEIGHTS_DIR = Path(os.environ.get("AMP_WEIGHTS_DIR", BASE_DIR / "feature_and_weight"))
RESULT_DIR = Path(os.environ.get("AMP_RESULT_DIR", BASE_DIR / "result"))

SAVE_DIR = BASE_DIR / "save"
MODEL_DIR = SAVE_DIR / "save_model"
PICTURE_DIR = SAVE_DIR / "picture"

# Cached artefacts produced by extract_features.py / train.py.
FEATURES_PKL = Path(os.environ.get("AMP_FEATURES_PKL", BASE_DIR / "features.pkl"))
SCALER_PKL = Path(os.environ.get("AMP_SCALER_PKL", WEIGHTS_DIR / "scaler.pkl"))
PCA_PKL = Path(os.environ.get("AMP_PCA_PKL", WEIGHTS_DIR / "pca_model.pkl"))
