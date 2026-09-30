import numpy as np
import random
import torch
import os
import pickle
from sklearn.model_selection import KFold
import pandas as pd
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix, matthews_corrcoef, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from model import stage_1model, ProteinDataset
from config import SAVE_DIR, MODEL_DIR, PICTURE_DIR, FEATURES_PKL, SCALER_PKL


def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    loop = tqdm(train_loader, desc="Training", leave=False)
    for batch in loop:
        features_batch = batch['features'].to(device)
        labels_batch = batch['labels'].to(device)
        optimizer.zero_grad()
        outputs = model(features_batch)
        loss = criterion(outputs, labels_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        loop.set_postfix(loss=loss.item())
    return running_loss / len(train_loader)


def evaluate_model(model, test_loader, device):
    model.eval()
    all_labels = []
    all_probs = []
    all_preds = []

    with torch.no_grad():
        loop = tqdm(test_loader, desc="Evaluating", leave=False)
        for batch in loop:
            features_batch = batch['features'].to(device)
            labels_batch = batch['labels'].to(device)
            outputs = model(features_batch)
            probs = nn.Softmax(dim=1)(outputs)[:, 1]
            preds = (probs >= 0.5).long()
            all_labels.extend(labels_batch.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    mcc = matthews_corrcoef(all_labels, all_preds)
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    auc_roc = roc_auc_score(all_labels, all_probs)
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    metrics = {
        'Accuracy': accuracy,
        'Sensitivity': sensitivity,
        'Precision': precision,
        'Specificity': specificity,
        'MCC': mcc,
        'F1 Score': f1,
        'AUC-ROC': auc_roc,
        'Recall': recall,
    }

    # Plot ROC Curve
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_roc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.scatter(fpr[optimal_idx], tpr[optimal_idx], marker='o', color='red', label=f'Optimal Threshold ({optimal_threshold:.2f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend()
    plt.grid(True)
    plt.savefig(PICTURE_DIR / f'roc_fold_{fold+1}.png')
    plt.close()

    # Plot Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - Fold {fold+1}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(PICTURE_DIR / f'confusion_matrix_fold_{fold+1}.png')
    plt.close()

    return metrics, all_probs, all_preds, all_labels


if __name__ == "__main__":
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PICTURE_DIR.mkdir(parents=True, exist_ok=True)

    random.seed(1)
    np.random.seed(17)
    torch.manual_seed(153)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    file_path = FEATURES_PKL
    print("Loading features and labels...")
    with open(file_path, 'rb') as f:
        features, labels = pickle.load(f)

    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    with open(SCALER_PKL, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"StandardScaler saved to {SCALER_PKL}")

    max_time_steps = features.shape[1]
    input_size = features.shape[1]
    print(f"Max time steps: {max_time_steps}, Input size: {input_size}")

    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    fold_results = []
    all_fold_probs = []
    all_fold_labels = []
    all_fold_preds = []

    NUM_EPOCHS = 15

    for fold, (train_idx, test_idx) in enumerate(kf.split(features)):
        print(f'\n{"="*50}\nStarting Fold {fold+1}\n{"="*50}')
        X_train, X_test = features[train_idx], features[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]
        train_dataset = ProteinDataset(X_train, y_train)
        test_dataset = ProteinDataset(X_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        model = stage_1model(max_time_steps, input_size).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

        best_auc = 0.0
        best_epoch = 0
        train_losses = []

        for epoch in range(NUM_EPOCHS):
            epoch_loss = train_model(model, train_loader, criterion, optimizer, device)
            train_losses.append(epoch_loss)

            scheduler.step(epoch_loss)

            print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {epoch_loss:.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')

            fold_metrics, _, _, _ = evaluate_model(model, test_loader, device)
            current_auc = fold_metrics['AUC-ROC']
            print(f"Validation AUC: {current_auc:.4f}")

            if current_auc > best_auc:
                best_auc = current_auc
                best_epoch = epoch
                best_model_path = MODEL_DIR / f'stage-1model_fold_{fold+1}_best.pth'
                torch.save(model.state_dict(), best_model_path)
                print(f"New best model saved at epoch {epoch+1} with AUC: {current_auc:.4f}")

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, NUM_EPOCHS+1), train_losses, 'b-o', linewidth=2)
        plt.axvline(x=best_epoch+1, color='r', linestyle='--', label=f'Best Epoch: {best_epoch+1}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Training Loss Curve for Fold {fold+1}')
        plt.legend()
        plt.grid(True)
        plt.savefig(PICTURE_DIR / f'train_loss_fold_{fold+1}.png')
        plt.close()

        model.load_state_dict(torch.load(best_model_path, weights_only=True))
        fold_metrics, fold_probs, fold_preds, fold_labels = evaluate_model(model, test_loader, device)
        fold_results.append(fold_metrics)

        all_fold_probs.extend(fold_probs)
        all_fold_labels.extend(fold_labels)
        all_fold_preds.extend(fold_preds)

        print(f'Fold {fold+1} Results:')
        for metric, value in fold_metrics.items():
            print(f"{metric}: {value:.4f}")

    all_metrics = pd.DataFrame(fold_results)
    mean_metrics = all_metrics.mean()
    std_metrics = all_metrics.std()

    print("Mean Metrics Across Folds:")
    print(mean_metrics)

    all_metrics.to_csv(SAVE_DIR / 'kfold_results.csv', index=False)
    mean_metrics.to_csv(SAVE_DIR / 'mean_metrics.csv', index=True)

    predictions_df = pd.DataFrame({
        'True Label': all_fold_labels,
        'Predicted Probability': all_fold_probs,
        'Predicted Label': all_fold_preds
    })
    predictions_df.to_csv(SAVE_DIR / 'all_predictions.csv', index=False)
