import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.decomposition import PCA
from tqdm import tqdm

from config import ESM2_DIR, DATA_DIR, FEATURES_DIR, FEATURES_PKL, PCA_PKL

def extract_features(sequence, tokenizer, model, device, max_len=100):
    inputs = tokenizer(sequence, return_tensors='pt', padding='max_length', truncation=True, max_length=max_len)
    inputs = {key: val.to(device) for key, val in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    hidden_states = outputs.hidden_states
    feature = torch.stack(hidden_states[-4:]).mean(0).squeeze(0).mean(dim=0).cpu().numpy()
    return feature

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    local_model_path = str(ESM2_DIR)
    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(local_model_path, do_lower_case=False)

    print("Loading Model configuration...")
    config = AutoConfig.from_pretrained(local_model_path, output_hidden_states=True)

    print("Loading Model...")
    model = AutoModel.from_pretrained(local_model_path, config=config).to(device)
    print("Model loaded successfully.")

    print("Loading dataset...")
    df = pd.read_csv(DATA_DIR / "ToxinPred3.0 all.csv", encoding='utf-8-sig')
    print("Columns:", df.columns.tolist())
    df.columns = df.columns.str.strip()
    print("Cleaned Columns:", df.columns.tolist())
    print("Data Preview:")
    print(df.head())

    tqdm.pandas(desc="Extracting Features")

    print("Starting feature extraction...")
    features = df['Sequence'].progress_apply(lambda seq: extract_features(seq, tokenizer, model, device))

    features = np.vstack(features)
    print("Features shape:", features.shape)

    print("Applying PCA...")
    pca = PCA(n_components=0.95)
    features_reduced = pca.fit_transform(features)
    print(f"Reduced features shape: {features_reduced.shape}, n_components={pca.n_components_}")

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    pca_model_path = PCA_PKL
    joblib.dump(pca, pca_model_path)
    print(f"PCA model saved to: {pca_model_path}")

    labels = df['toxicity'].astype(int).values

    print("Saving features and labels...")
    with open(FEATURES_PKL, 'wb') as f:
        pickle.dump((features_reduced, labels), f)
