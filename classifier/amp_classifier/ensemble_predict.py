import torch
import numpy as np
import pickle
import os
import pandas as pd
from model import stage_1model
from config import FEATURES_PKL, SCALER_PKL, WEIGHTS_DIR, DATA_DIR, RESULT_DIR


def ensemble_predict(models, features_tensor):
    all_probs = []
    with torch.no_grad():
        for model in models:
            outputs = model(features_tensor)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            all_probs.append(probs.cpu().numpy())
    avg_probs = np.mean(all_probs, axis=0)
    pred_labels = (avg_probs >= 0.5).astype(int)
    return avg_probs, pred_labels


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    with open(FEATURES_PKL, 'rb') as f:
        features, labels = pickle.load(f)

    with open(SCALER_PKL, 'rb') as f:
        scaler = pickle.load(f)

    features_scaled = scaler.transform(features)
    features_tensor = torch.tensor(features_scaled, dtype=torch.float32).to(device)

    models = []
    for fold in range(1, 6):
        model_path = os.path.join(WEIGHTS_DIR, f'stage-1model_fold_{fold}.pth')
        model = stage_1model(features.shape[1], features.shape[1]).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()
        models.append(model)

    avg_probs, pred_labels = ensemble_predict(models, features_tensor)

    df = pd.read_csv(DATA_DIR / 'if_amp.csv')

    results_df = pd.DataFrame({
        'Sequence': df['aa_seq'],
        'True_Label': labels,
        'Avg_Probability': avg_probs,
        'Predicted_Label': pred_labels
    })

    results_df.to_csv(RESULT_DIR / 'ensemble_predictions.csv', index=False)
    print(f"Ensemble prediction results saved to {RESULT_DIR / 'ensemble_predictions.csv'}")
    print("Prediction results for the first 10 samples:")
    print(results_df.head(10))
