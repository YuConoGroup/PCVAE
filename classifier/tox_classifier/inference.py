import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pickle
from torch.utils.data import Dataset, DataLoader
import os

# Import model definition
from model import stage_1model
from config import MODEL_DIR, SCALER_PKL, FEATURES_DIR, RESULT_DIR

class ProteinInferenceDataset(Dataset):
    """Dataset class for inference (no labels required)"""
    def __init__(self, features):
        self.features = features
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return {
            'features': torch.tensor(self.features[idx], dtype=torch.float32)
        }

def load_model(fold_num, max_time_steps, input_size, device):
    """Load model for a specific fold"""
    model = stage_1model(max_time_steps, input_size).to(device)
    model_path = os.path.join(MODEL_DIR, f'stage-1model_fold_{fold_num}_best.pth')
    
    if not os.path.exists(model_path):
        print(f"Warning: Model file not found at {model_path}")
        return None
    
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    return model

def inference_single_model(model, test_loader, device):
    """Run inference with a single model"""
    all_probs = []
    all_logits = []  # Store raw outputs (before softmax)
    
    with torch.no_grad():
        for batch in test_loader:
            features_batch = batch['features'].to(device)
            outputs = model(features_batch)
            
            # Get softmax probabilities
            probs = nn.Softmax(dim=1)(outputs)[:, 1]  # Get probability of positive class
            all_probs.extend(probs.cpu().numpy())
            all_logits.extend(outputs.cpu().numpy())  # Store raw logits
    
    return np.array(all_probs), np.array(all_logits)

def ensemble_inference(models, test_loader, device):
    """Run inference using ensemble learning (multiple models)"""
    all_model_probs = []
    all_model_logits = []
    
    for i, model in enumerate(models):
        if model is not None:
            probs, logits = inference_single_model(model, test_loader, device)
            all_model_probs.append(probs)
            all_model_logits.append(logits)
            print(f"  Model {i+1}: Prediction complete")
    
    if not all_model_probs:
        print("Error: No models available for inference")
        return None, None, None
    
    # Average prediction probabilities across all models
    avg_probs = np.mean(all_model_probs, axis=0)
    # Average logits across all models (optional, for subsequent analysis)
    avg_logits = np.mean(all_model_logits, axis=0)
    # Binary classification using threshold 0.5
    avg_preds = (avg_probs >= 0.5).astype(int)
    
    return avg_probs, avg_preds, avg_logits

def load_features(features_path):
    """Load feature file"""
    print(f"Loading feature file: {features_path}")
    
    if not os.path.exists(features_path):
        print(f"Error: Feature file not found at {features_path}")
        return None
    
    try:
        with open(features_path, 'rb') as f:
            features = pickle.load(f)
        
        # Handle different data formats
        if isinstance(features, tuple):
            # If it's a (features, labels) tuple, only take features
            features = features[0]
            print("Detected tuple format, using only the features portion")
        elif isinstance(features, dict):
            # If it's a dictionary, try to get the 'features' key
            if 'features' in features:
                features = features['features']
            elif 'X' in features:
                features = features['X']
            else:
                print("No feature data found in dictionary, attempting to use dictionary values directly")
                # Take the first value as features
                features = list(features.values())[0]
        
        print(f"Feature shape: {features.shape if hasattr(features, 'shape') else 'unknown'}")
        print(f"Number of features: {len(features) if hasattr(features, '__len__') else 'unknown'}")
        
        return features
    
    except Exception as e:
        print(f"Error loading feature file: {e}")
        return None

def apply_scaling(features, scaler_path):
    """Apply standardization"""
    if not os.path.exists(scaler_path):
        print(f"Warning: Scaler file not found at {scaler_path}")
        print("Will use raw features (not standardized)")
        return features
    
    try:
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        print("Applying standardization...")
        # Ensure correct feature shape
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        features_scaled = scaler.transform(features)
        print("Standardization complete")
        return features_scaled
    
    except Exception as e:
        print(f"Error applying standardization: {e}")
        return features

def save_predictions(predictions, output_path):
    """Save prediction results"""
    results_df = pd.DataFrame(predictions)
    
    # Save to CSV
    results_df.to_csv(output_path, index=True, index_label='Sample_ID')
    print(f"Prediction results saved to {output_path}")
    
    return results_df

def predict_unlabeled_features(features_path=None,
                              scaler_path=None,
                              output_path=None):
    """
    Predict unlabeled feature files
    
    Parameters:
    - features_path: Feature file path (default: features/inference_features.pkl).
    - scaler_path: Scaler file path (default: the module's scaler.pkl).
    - output_path: Output results file path (default: generated/preTOX_results.csv).
    """
    features_path = str(features_path) if features_path is not None else str(FEATURES_DIR / 'inference_features.pkl')
    scaler_path = str(scaler_path) if scaler_path is not None else str(SCALER_PKL)
    output_path = str(output_path) if output_path is not None else str(RESULT_DIR / 'preTOX_results.csv')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print("=" * 60)
    print("Starting inference on unlabeled data")
    print("=" * 60)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load features
    features = load_features(features_path)
    if features is None:
        print("Unable to load features, exiting inference")
        return None
    
    # 2. Apply standardization
    features_scaled = apply_scaling(features, scaler_path)
    
    # 3. Get input parameters
    if len(features_scaled.shape) == 1:
        # If only one sample
        max_time_steps = len(features_scaled)
        input_size = len(features_scaled)
        features_scaled = features_scaled.reshape(1, -1)
    else:
        max_time_steps = features_scaled.shape[1]
        input_size = features_scaled.shape[1]
    
    print(f"Processed feature shape: {features_scaled.shape}")
    print(f"Max time steps: {max_time_steps}")
    print(f"Input dimension: {input_size}")
    
    # 4. Create dataset and data loader
    dataset = ProteinInferenceDataset(features_scaled)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # 5. Load models from all folds
    print("Loading models...")
    models = []
    successful_models = 0
    
    for fold_num in range(1, 6):  # Assume 5-fold models
        model = load_model(fold_num, max_time_steps, input_size, device)
        if model is not None:
            models.append(model)
            successful_models += 1
            print(f"OK: Loaded fold {fold_num} model")
        else:
            print(f"OK: Fold {fold_num} model loading failed")
    
    if successful_models == 0:
        print("Error: No models loaded successfully")
        return None
    
    print(f"Successfully loaded {successful_models} models")
    
    # 6. Run ensemble inference
    print("Starting ensemble inference...")
    probabilities, predictions, logits = ensemble_inference(models, dataloader, device)
    
    if probabilities is None:
        print("Inference failed")
        return None
    
    # 7. Prepare results
    print("Preparing prediction results...")
    num_samples = len(probabilities)
    
    # Create results dictionary
    results = {
        'Prediction_Probability': probabilities,
        'Predicted_Label': predictions,
        'Confidence_Level': []
    }
    
    # Add confidence levels
    for prob in probabilities:
        if prob >= 0.8 or prob <= 0.2:
            results['Confidence_Level'].append('Very High')
        elif prob >= 0.7 or prob <= 0.3:
            results['Confidence_Level'].append('High')
        elif prob >= 0.6 or prob <= 0.4:
            results['Confidence_Level'].append('Medium')
        else:
            results['Confidence_Level'].append('Low')
    
    # Add class description
    results['Class_Description'] = ['Positive' if pred == 1 else 'Negative' for pred in predictions]
    
    # 8. Save results
    results_df = save_predictions(results, output_path)
    
    # 9. Print statistics
    print("" + "=" * 60)
    print("Inference complete! Statistics:")
    print("=" * 60)
    print(f"Total samples: {num_samples}")
    print(f"Predicted positive (Positive): {(predictions == 1).sum()} ({((predictions == 1).sum()/num_samples)*100:.1f}%)")
    print(f"Predicted negative (Negative): {(predictions == 0).sum()} ({((predictions == 0).sum()/num_samples)*100:.1f}%)")
    print(f"Average prediction probability: {probabilities.mean():.4f}")
    print(f"Prediction probability std: {probabilities.std():.4f}")
    
    # Confidence statistics
    confidence_counts = pd.Series(results['Confidence_Level']).value_counts()
    print("Confidence distribution:")
    for level, count in confidence_counts.items():
        percentage = (count / num_samples) * 100
        print(f"  {level}: {count} ({percentage:.1f}%)")
    
    # Show first few results
    print("Prediction results for first 10 samples:")
    preview_df = results_df.head(10).copy()
    preview_df['Prediction_Probability'] = preview_df['Prediction_Probability'].apply(lambda x: f"{x:.4f}")
    print(preview_df[['Prediction_Probability', 'Predicted_Label', 'Confidence_Level', 'Class_Description']].to_string())
    
    # Save detailed results (including raw features and logits)
    detailed_results = {
        'features': features_scaled,
        'probabilities': probabilities,
        'predictions': predictions,
        'logits': logits,
        'confidence_levels': results['Confidence_Level']
    }
    
    detailed_output_path = output_path.replace('.csv', '_detailed.pkl')
    with open(detailed_output_path, 'wb') as f:
        pickle.dump(detailed_results, f)
    
    print(f"Detailed results saved to: {detailed_output_path}")
    
    return results_df

def main():
    """Main function - run inference directly"""
    import sys
    round_tag = os.environ.get("GEN_ROUND", "1")
    # Set input and output paths
    features_path = sys.argv[1] if len(sys.argv) > 1 else str(FEATURES_DIR / 'inference_features.pkl')  # Input feature file
    scaler_path = str(SCALER_PKL)  # Scaler saved during training
    output_path = sys.argv[2] if len(sys.argv) > 2 else str(RESULT_DIR / f'tox_predictions_round{round_tag}.csv')  # Output results file
    
    # Run inference
    results = predict_unlabeled_features(features_path, scaler_path, output_path)
    
    if results is not None:
        print("Inference pipeline complete!")
        return results
    else:
        print("Inference pipeline failed!")
        return None

if __name__ == "__main__":
    # Run inference directly
    main()
