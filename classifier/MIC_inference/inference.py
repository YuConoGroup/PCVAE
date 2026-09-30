import torch
import pandas as pd
from transformers import BertModel, BertTokenizer
import torch.nn as nn
import os
import argparse

from config import PROTBERT_DIR, DATA_DIR, RESULT_DIR, MODEL_PATH

# Define model structure
class REG(nn.Module):
    def __init__(self): 
        super(REG, self).__init__()
        self.bert = BertModel.from_pretrained(
            str(PROTBERT_DIR),
            output_attentions=True
        )
        self.regressor = nn.Sequential(
            nn.LayerNorm(self.bert.config.hidden_size),
            nn.Linear(self.bert.config.hidden_size, 512),
            nn.LeakyReLU(inplace=False),
            nn.Dropout(p=0.2),
            nn.Linear(512, 128),
            nn.LeakyReLU(inplace=False),
            nn.Dropout(p=0.2),
            nn.Linear(128, 1)
        )

    def forward(self, input_ids, attention_mask):
        output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        return self.regressor(output.pooler_output), self.bert

# Load trained model (with device selection)
def load_trained_model(model_path, device):
    model = REG()
    if device.type == 'cuda':
        state_dict = torch.load(model_path, weights_only=True)
    else:
        state_dict = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print(f"Successfully loaded model: {model_path}")
    print(f"Model running on: {'GPU' if device.type == 'cuda' else 'CPU'} device")
    return model

# Prediction function (with device support)
def predict_sequences(model, tokenizer, sequences, device):
    predictions = []
    print(f"{'='*50}Starting prediction for {len(sequences)} sequences...")
    print(f"Device status: {'GPU accelerated' if device.type == 'cuda' else 'CPU running'}")
    
    for i, seq in enumerate(sequences):
        inputs = tokenizer(
            " ".join(seq),
            padding='max_length',
            truncation=True,
            max_length=100,
            return_tensors="pt"
        )
        
        
        # Move input data to current device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        print(f"Sequence {i+1}/{len(sequences)}: '{seq[:10]}...'")
        print(f"Tokenizer output:")
        print(f"input_ids shape: {inputs['input_ids'].shape}")
        print(f"attention_mask shape: {inputs['attention_mask'].shape}")
        
        with torch.no_grad():
            pred, _ = model(
                inputs["input_ids"],
                inputs["attention_mask"]
            )
            
            print(f"Regressor output shape: {pred.shape}")
            mic_value = pred.item()
            predictions.append(mic_value)
            print(f"Predicted MIC value: {mic_value:.4f}")
    
    return predictions

if __name__ == "__main__":
    # Model and file paths
    ap = argparse.ArgumentParser(description="Predict MIC for peptide sequences (ProtBERT + MLP)")
    ap.add_argument("--model", default=str(MODEL_PATH), help="trained regressor checkpoint (.pth)")
    ap.add_argument("--input", default=str(DATA_DIR / "sequences_with_properties.csv"),
                    help="input CSV with a 'Sequence' column")
    ap.add_argument("--output", default=str(RESULT_DIR / "PreMIC.csv"),
                    help="output CSV path")
    ap.add_argument("--device", default=None, help="torch device, e.g. cuda:0 or cpu (default: auto)")
    args = ap.parse_args()

    model_path = args.model
    csv_path = args.input
    output_path = args.output

    # 1. Detect and select device (prefer CUDA)
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("" + "="*50)
    print(f"Device detection result:")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Available GPU count: {torch.cuda.device_count()}")
        print(f"Current GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Warning: No CUDA device detected, will use CPU for prediction")
    print("="*50 + "")
    
    # 2. Load model and tokenizer
    model = load_trained_model(model_path, device)
    tokenizer = BertTokenizer.from_pretrained(str(PROTBERT_DIR))
    
    # 3. Read CSV file
    if not os.path.exists(csv_path):
        print(f"Error: File not found {csv_path}")
        exit()
        
    df = pd.read_csv(csv_path)
    sequences = df['Sequence'].tolist()
    
    # 4. Batch prediction
    mic_predictions = predict_sequences(model, tokenizer, sequences, device)
    
    # 5. Add to DataFrame and save results
    df['predicted_MIC'] = mic_predictions
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"{'='*50}")
    print(f"Prediction complete! Results saved to: {output_path}")
    print(f"Total sequences: {len(sequences)}")
    print(f"Device type: {'GPU' if device.type == 'cuda' else 'CPU'}")
    print("Prediction results preview:")
    print(df[['Sequence', 'predicted_MIC']].head())
    print("="*50)
