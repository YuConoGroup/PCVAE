"""Central path configuration for the MIC regression model (ProtBERT + MLP).

Defaults are relative to this file so the module runs from a plain checkout.
Override with environment variables to use external data / pretrained models.

Environment variables
---------------------
MIC_INFERENCE_DIR  Module root (default: this directory).
MIC_PROTBERT_DIR   Local ProtBERT model directory.
MIC_DATA_DIR       Input CSV directory.
MIC_RESULT_DIR     Output directory.
MIC_MODEL_PATH     Trained regressor checkpoint (best_model-EC-0.pth).
"""

import os
from pathlib import Path

BASE_DIR = Path(os.environ.get("MIC_INFERENCE_DIR", Path(__file__).resolve().parent))

PROTBERT_DIR = Path(os.environ.get("MIC_PROTBERT_DIR", BASE_DIR / "prot_bert"))
DATA_DIR = Path(os.environ.get("MIC_DATA_DIR", BASE_DIR / "data"))
RESULT_DIR = Path(os.environ.get("MIC_RESULT_DIR", BASE_DIR / "result"))

MODEL_PATH = Path(os.environ.get("MIC_MODEL_PATH", BASE_DIR / "best_model-EC-0.pth"))

# Artefacts used by reproduce.py.
FINETUNE_MODEL_PKL = Path(os.environ.get("MIC_FINETUNE_MODEL_PKL",
                                         BASE_DIR / "prot_bert_finetune_reproduce.pkl"))
TEST_LOADER_PKL = Path(os.environ.get("MIC_TEST_LOADER_PKL",
                                      BASE_DIR / "test_loader_reproduce.pkl"))
