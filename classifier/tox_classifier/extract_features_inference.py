# stage_1_inference.py
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
import pandas as pd
import numpy as np
import pickle
import joblib
from tqdm import tqdm
import os

from config import ESM2_DIR, DATA_DIR, FEATURES_DIR, PCA_PKL

def process_features(
    mode='inference',
    input_file=None,
    output_dir="./features",
    pca_model_path=None,
    max_len=100
):
    """
    Feature processing function for inference mode (compatible with original interface)
    """
    if mode != 'inference':
        print("Only inference mode is supported in this script")
        return None, None, None
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and tokenizer
    local_model_path = str(ESM2_DIR)
    tokenizer = AutoTokenizer.from_pretrained(local_model_path, do_lower_case=False)
    config = AutoConfig.from_pretrained(local_model_path, output_hidden_states=True)
    model = AutoModel.from_pretrained(local_model_path, config=config).to(device)
    
    # Feature extraction function (same as during training)
    def extract_features(sequence, tokenizer, model, device, max_len=100):
        inputs = tokenizer(sequence, return_tensors='pt', padding='max_length', truncation=True, max_length=max_len)
        inputs = {key: val.to(device) for key, val in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)

        hidden_states = outputs.hidden_states
        feature = torch.stack(hidden_states[-4:]).mean(0).squeeze(0).mean(dim=0).cpu().numpy()
        return feature
    
    # Load inference data
    print("Loading inference data...")
    df = pd.read_csv(input_file, encoding='utf-8-sig')
    df.columns = df.columns.str.strip()
    
    # Extract features
    print("Extracting features...")
    tqdm.pandas(desc="Extracting Features")
    features = df['Sequence'].progress_apply(lambda seq: extract_features(seq, tokenizer, model, device, max_len))
    features = np.vstack(features)
    print(f"Raw features shape: {features.shape}")
    
    # Load PCA model and transform
    print(f"Loading PCA model from {pca_model_path}...")
    pca = joblib.load(pca_model_path)
    features_reduced = pca.transform(features)
    print(f"Reduced features shape: {features_reduced.shape}")
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "inference_features.pkl")
    with open(output_path, 'wb') as f:
        pickle.dump(features_reduced, f)
    
    return features_reduced, None, features_reduced.shape[1]

if __name__ == "__main__":
    # Inference mode
    import sys
    input_file = sys.argv[1] if len(sys.argv) > 1 else str(DATA_DIR / "generated_sequences_round1.csv")
    output_dir = sys.argv[2] if len(sys.argv) > 2 else str(FEATURES_DIR)
    print("="*60)
    print("Inference mode: Processing inference data")
    print("="*60)

    # Ensure using the PCA model saved during training
    inference_features, _, inference_dim = process_features(
        mode='inference',
        input_file=input_file,
        output_dir=output_dir,
        pca_model_path=str(PCA_PKL),
        max_len=100
    )
    

    if inference_features is not None:
        print(f"Inference feature extraction complete! Feature dimension: {inference_dim}")
        print("Ensure training and inference feature dimensions are consistent!")
    else:
        print("Inference feature extraction failed!")
