import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import os
import pickle
import time
import matplotlib.pyplot as plt
from model import stage_1model
from config import ESM2_DIR, DATA_DIR, WEIGHTS_DIR, RESULT_DIR, PCA_PKL


class AntimicrobialPeptideDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'features': self.features[idx],
            'labels': self.labels[idx]
        }


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    total_start_time = time.time()

    data_load_start = time.time()
    data_file = DATA_DIR / "amp_copy_predictions.csv"
    data = pd.read_csv(data_file)
    sequences = data.iloc[:, 0].tolist()
    labels = data.iloc[:, 1].values.astype(int)
    data_load_time = time.time() - data_load_start
    print(f"Data loading time: {data_load_time:.4f} seconds")

    esm_load_start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(str(ESM2_DIR))
    pretrained_model = AutoModel.from_pretrained(str(ESM2_DIR))
    pretrained_model.to(device)
    pretrained_model.eval()
    esm_load_time = time.time() - esm_load_start
    print(f"ESM-2 model loading time: {esm_load_time:.4f} seconds")

    feat_extract_start = time.time()
    max_length = 100
    features = []
    batch_times = []

    with torch.no_grad():
        for i, seq in enumerate(sequences):
            batch_start = time.time()

            inputs = tokenizer(
                seq,
                add_special_tokens=True,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            ).to(device)

            outputs = pretrained_model(**inputs)
            sequence_representation = outputs.last_hidden_state.mean(dim=1).squeeze()
            features.append(sequence_representation.cpu().numpy())

            batch_time = time.time() - batch_start
            batch_times.append(batch_time)

            if (i + 1) % 100 == 0:
                avg_time = np.mean(batch_times[-100:])
                print(f"Processed {i+1}/{len(sequences)} sequences | Avg batch time: {avg_time:.4f}s")

    features = np.array(features)
    feat_extract_time = time.time() - feat_extract_start
    print(f"Raw feature shape: {features.shape}")
    print(f"Feature extraction time: {feat_extract_time:.4f} seconds")

    plt.figure(figsize=(10, 6))
    plt.plot(batch_times, alpha=0.7)
    plt.xlabel('Sequence Index')
    plt.ylabel('Time (seconds)')
    plt.title('Feature Extraction Time per Sequence')
    plt.grid(True)
    plt.savefig(RESULT_DIR / "feature_extraction_times1.png")
    print("Saved feature extraction time plot")

    pca_start = time.time()
    pca_path = PCA_PKL

    if not os.path.exists(pca_path):
        print("Training new PCA model...")
        from sklearn.decomposition import PCA
        pca = PCA(n_components=0.95)
        pca.fit(features)
        with open(pca_path, 'wb') as f:
            pickle.dump(pca, f)
        print(f"Saved new PCA model at {pca_path}")

    with open(pca_path, 'rb') as f:
        pca = pickle.load(f)
    features = pca.transform(features)
    pca_time = time.time() - pca_start
    print(f"PCA transformation time: {pca_time:.4f} seconds")

    scaler_start = time.time()
    scaler_path = WEIGHTS_DIR / "scaler_a.pkl"
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        features = scaler.transform(features)
        print("Features standardized using saved scaler")
    else:
        print("Warning: Scaler not found. Using raw features.")
    scaler_time = time.time() - scaler_start
    print(f"Feature scaling time: {scaler_time:.4f} seconds")

    dataset_start = time.time()
    dataset = AntimicrobialPeptideDataset(features, labels)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False)
    dataset_time = time.time() - dataset_start
    print(f"Dataset preparation time: {dataset_time:.4f} seconds")

    cls_load_start = time.time()
    model_path = WEIGHTS_DIR / "stage-1model_fold_3_best_a.pth"

    model = stage_1model(
        max_time_steps=features.shape[1],
        input_size=features.shape[1]
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    cls_load_time = time.time() - cls_load_start
    print(f"Model loaded from {model_path}")
    print(f"Classifier model loading time: {cls_load_time:.4f} seconds")

    print(f"{'='*50}")
    print(f"Total samples: {len(dataset)}")
    print(f"Feature dimensions: {features.shape[1]}")
    print(f"Batch size: {data_loader.batch_size}")
    print(f"Total batches: {len(data_loader)}")
    print(f"{'='*50}")

    inf_start_time = time.time()
    y_true, y_pred, y_prob = [], [], []
    batch_inf_times = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            batch_start = time.time()

            features_batch = batch['features'].to(device)
            labels_batch = batch['labels'].to(device)

            outputs = model(features_batch)
            probs = F.softmax(outputs, dim=1)[:, 1]
            preds = (probs > 0.5).long()

            y_true.extend(labels_batch.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())

            batch_time = time.time() - batch_start
            batch_inf_times.append(batch_time)

            if (batch_idx + 1) % 10 == 0 or batch_idx == 0 or (batch_idx + 1) == len(data_loader):
                print(f"Batch {batch_idx+1}/{len(data_loader)} | Time: {batch_time:.4f}s")

    inf_time = time.time() - inf_start_time
    print(f"Total inference time: {inf_time:.4f} seconds")
    print(f"Average batch inference time: {np.mean(batch_inf_times):.4f} seconds")

    plt.figure(figsize=(10, 6))
    plt.plot(batch_inf_times, 'o-', alpha=0.7)
    plt.xlabel('Batch Index')
    plt.ylabel('Time (seconds)')
    plt.title('Inference Time per Batch')
    plt.grid(True)
    plt.savefig(RESULT_DIR / "inference_times1.png")
    print("Saved inference time plot")

    eval_start = time.time()
    accuracy = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    eval_time = time.time() - eval_start
    print(f"Evaluation time: {eval_time:.4f} seconds")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC: {auc:.4f}")

    save_start = time.time()
    results = pd.DataFrame({
        'Sequence': sequences,
        'True_Label': y_true,
        'Predicted_Label_a': y_pred,
        'Probability': y_prob
    })
    results.to_csv(RESULT_DIR / "predictions_a.csv", index=False)
    save_time = time.time() - save_start
    print(f"Results saving time: {save_time:.4f} seconds")
    print("Predictions saved to predictions_a.csv")

    total_time = time.time() - total_start_time

    time_report = f"""
{'='*50}
TOTAL PROCESSING TIME REPORT
{'='*50}
Data loading:              {data_load_time:.4f} seconds
ESM-2 model loading:       {esm_load_time:.4f} seconds
Feature extraction:        {feat_extract_time:.4f} seconds
PCA transformation:        {pca_time:.4f} seconds
Feature scaling:           {scaler_time:.4f} seconds
Dataset preparation:       {dataset_time:.4f} seconds
Classifier model loading:  {cls_load_time:.4f} seconds
Inference:                 {inf_time:.4f} seconds
Evaluation:                {eval_time:.4f} seconds
Results saving:            {save_time:.4f} seconds
{'-'*50}
TOTAL TIME:                {total_time:.4f} seconds
{'='*50}
"""

    print(time_report)

    with open(RESULT_DIR / "time_report1.txt", "w") as f:
        f.write(time_report)

    chart_labels = [
        'Data Loading',
        'Model Loading',
        'Feature Extraction',
        'PCA',
        'Scaling',
        'Dataset Prep',
        'Inference',
        'Evaluation',
        'Saving Results'
    ]

    sizes = [
        data_load_time,
        esm_load_time,
        feat_extract_time,
        pca_time,
        scaler_time,
        dataset_time,
        inf_time,
        eval_time,
        save_time
    ]

    other_time = total_time - sum(sizes)
    if other_time > 0:
        chart_labels.append('Other')
        sizes.append(other_time)

    plt.figure(figsize=(12, 8))
    plt.pie(sizes, labels=chart_labels, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title('Time Distribution of Processing Steps')
    plt.savefig(RESULT_DIR / "time_distribution1.png")
    print("Saved time distribution chart")
