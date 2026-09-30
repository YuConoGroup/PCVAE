"""
Complete CVAE training pipeline - adapted for LSTM decoder version of CVAE
Modification notes:
1. Training parameters adapted for the new train_cvae (kl_anneal_epochs, teacher_forcing_ratio, tf_decay)
2. Test evaluation uses the new loss function (cvae_total_loss)
3. Model loading uses the new load_model (no model_class parameter needed)
4. Generation examples remain unchanged, LSTM output is already a probability distribution
"""

import pandas as pd
import numpy as np
import torch
import joblib
import os
import shutil
from sklearn.preprocessing import MinMaxScaler
from preprocessing import build_one_hot_encoder, index_to_aa_mapping, AMINO_ACIDS
from train import (
    train_cvae,
    cvae_total_loss,
    load_model,
    set_seed,
)
from sampling import nucleus_sampling, reconstruct_from_lstm_probs
from config import DATA_DIR, SAVE_DIR


# ---------- Helper functions ----------


def set_random_seed(seed=666):
    print(f"Setting random seed: {seed}")
    set_seed(seed)
    return seed


def load_pre_split_datasets():
    print("Loading pre-split datasets...")
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    valid_df = pd.read_csv(DATA_DIR / "valid.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    print(f"Training set: {train_df.shape}, Validation set: {valid_df.shape}, Test set: {test_df.shape}")
    return train_df, valid_df, test_df


def encode_sequences_and_extract_labels(df, encoder, is_training=False, mic_scaler=None):
    X = np.array([encoder(seq) for seq in df['Sequence']])
    y_mic = df['predicted_MIC'].values.astype(float)
    y_a = df['Predicted_Label_a'].values.astype(float)
    y_t = df['Predicted_Label_t'].values.astype(float)
    y_aip = df['predicted_AIP'].values.astype(float)

    if is_training:
        mic_scaler = MinMaxScaler()
        y_mic_scaled = mic_scaler.fit_transform(y_mic.reshape(-1, 1)).flatten()
    elif mic_scaler is not None:
        y_mic_scaled = mic_scaler.transform(y_mic.reshape(-1, 1)).flatten()
    else:
        y_mic_scaled = y_mic
    return X, y_mic_scaled, y_a, y_t, y_aip, mic_scaler


def main(seed=666):
    seed = set_random_seed(seed)

    base_save_dir = SAVE_DIR
    result_save_dir = os.path.join(base_save_dir, "png")
    os.makedirs(base_save_dir, exist_ok=True)
    os.makedirs(result_save_dir, exist_ok=True)
    print(f"Model save directory: {base_save_dir}")
    print(f"Result save directory: {result_save_dir}")

    # 1. Load data
    train_df, valid_df, test_df = load_pre_split_datasets()

    # 2. Sequence encoder
    encoder = build_one_hot_encoder(max_length=30)

    # 3. Process training set
    X_train, y_mic_train_scaled, y_a_train, y_t_train, y_aip_train, mic_scaler = encode_sequences_and_extract_labels(
        train_df, encoder, is_training=True
    )
    print(f"Training set: {X_train.shape}")

    # 4. Process validation set
    X_val, y_mic_val_scaled, y_a_val, y_t_val, y_aip_val, _ = encode_sequences_and_extract_labels(
        valid_df, encoder, is_training=False, mic_scaler=mic_scaler
    )
    print(f"Validation set: {X_val.shape}")

    # 5. Process test set
    X_test, y_mic_test_scaled, y_a_test, y_t_test, y_aip_test, _ = encode_sequences_and_extract_labels(
        test_df, encoder, is_training=False, mic_scaler=mic_scaler
    )
    print(f"Test set: {X_test.shape}")

    # 6. Prepare validation set label combination (new train_cvae requires y_val as [N, 4])
    y_val_combined = np.column_stack([y_mic_val_scaled, y_a_val, y_t_val, y_aip_val])

    # 7. Loss weight settings (beta will be overridden by annealing, but base value still needed)
    loss_weights = {
        'alpha': 1.0,        # Sequence reconstruction weight
        'beta': 0.5,         # KLD base weight (actually annealed)
        'gamma_mic': 10.0,   # MIC loss weight (recommended to increase appropriately)
        'gamma_a': 1.0,
        'gamma_t': 1.0,
        'gamma_aip': 1.0,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 8. Training parameters - adapted for new train_cvae
    train_params = {
        'X_train': X_train,
        'y_1_train': y_mic_train_scaled,
        'y_2_train': y_a_train,
        'y_3_train': y_t_train,
        'y_4_train': y_aip_train,
        'mic_scaler': mic_scaler,
        'learning_rate': 1e-4,
        'num_epochs': 200,
        'batch_size': 512,
        'device': device,
        'max_len': 30,
        'vocab_size': len(AMINO_ACIDS),
        'input_dim': 30 * len(AMINO_ACIDS),   # max_len * vocab_size = 600
        'save_dir': base_save_dir,
        'save_every': 10,
        'X_val': X_val,
        'y_val': y_val_combined,
        'loss_weights': loss_weights,
        'kl_anneal_epochs': 10,           # Gradually increase beta over the first 20 epochs
        'teacher_forcing_ratio': 1.0,     # Initial teacher forcing ratio
        'tf_decay': 0.98,                 # Decay factor per epoch
    }

    print("Starting CVAE training (LSTM decoder version)...")
    # Directly call the new train_cvae (no wrapper needed)
    model, mic_scaler, train_losses, val_losses, train_comps, val_comps = train_cvae(**train_params)

    # Training curves are automatically saved to save_dir by train_cvae, copy to result_dir
    src_curve = os.path.join(base_save_dir, "training_curve.png")
    if os.path.exists(src_curve):
        shutil.copy(src_curve, os.path.join(result_save_dir, "training_curve.png"))
        print(f"Training curve copied to: {result_save_dir}")

    # 9. Load best model (new load_model does not need model_class parameter)
    print("Loading best model...")
    best_model, loaded_scaler, checkpoint = load_model(
        checkpoint_path=os.path.join(base_save_dir, "best_model.pth"),
        device=device
    )
    
    # 12. Save normalizer
    scaler_path = os.path.join(base_save_dir, f"mic_scaler_seed_{seed}.pkl")
    joblib.dump(mic_scaler, scaler_path)
    print(f"Normalizer saved: {scaler_path}")

    # 10. Test set evaluation
    print("Evaluating on test set...")
    best_model.eval()
    with torch.no_grad():
        X_test_tensor = torch.from_numpy(X_test).float().to(device)
        y_mic_test_tensor = torch.from_numpy(y_mic_test_scaled).float().to(device).unsqueeze(1)
        y_a_test_tensor = torch.from_numpy(y_a_test.astype(float)).float().to(device).unsqueeze(1)
        y_t_test_tensor = torch.from_numpy(y_t_test.astype(float)).float().to(device).unsqueeze(1)
        y_aip_test_tensor = torch.from_numpy(y_aip_test.astype(float)).float().to(device).unsqueeze(1)

        # Forward pass (no teacher forcing during testing)
        _, decoded_logits, mu, log_var, _, mic_pred, a_pred, t_pred, aip_pred = best_model(
            X_test_tensor,
            y_mic_test_tensor,
            y_a_test_tensor,
            y_t_test_tensor,
            y_aip_test_tensor,
            teacher_forcing_ratio=0.0
        )

        # Use the new loss function to compute test loss
        test_loss, test_loss_dict = cvae_total_loss(
            decoded_logits, X_test_tensor, mu, log_var,
            mic_pred, y_mic_test_tensor,
            a_pred, y_a_test_tensor,
            t_pred, y_t_test_tensor,
            aip_pred, y_aip_test_tensor,
            max_len=30,
            vocab_size=len(AMINO_ACIDS),
            alpha=loss_weights['alpha'],
            beta=0.5,           # KLD weight is 1.0 during testing
            gamma_mic=loss_weights['gamma_mic'],
            gamma_a=loss_weights['gamma_a'],
            gamma_t=loss_weights['gamma_t'],
            gamma_aip=loss_weights['gamma_aip']
        )
        test_loss = test_loss.item() / len(X_test)
        print(f"Test set total loss: {test_loss:.4f}")
        print(f"  Reconstruction loss: {test_loss_dict['recon'] / len(X_test):.2f}")
        print(f"  KLD loss: {test_loss_dict['kld'] / len(X_test):.2f}")
        print(f"  MIC loss: {test_loss_dict['mic'] / len(X_test):.4f}")

    # 11. Generate examples - using LSTM-specific reconstruction function
    print("Generating new sequence examples...")
    num_samples = 5
    # Set desired condition values (normalized MIC recommended in [0,1] range)
    condition_mic = torch.tensor([[0.3]] * num_samples).float().to(device)
    condition_a = torch.tensor([[1.0]] * num_samples).float().to(device)
    condition_t = torch.tensor([[0.0]] * num_samples).float().to(device)
    condition_aip = torch.tensor([[1.0]] * num_samples).float().to(device)

    with torch.no_grad():
        # sample returns [num_samples, max_len, vocab_size] probability distribution
        generated_probs = best_model.sample(
            num_samples=num_samples,
            y_1=condition_mic,
            y_2=condition_a,
            y_3=condition_t,
            y_4=condition_aip,
            device=device,
            temperature=1       # Lower temperature increases determinism
        )

    generated_sequences = []
    for i in range(num_samples):
        seq = reconstruct_from_lstm_probs(
            generated_probs[i],
            min_length=10,
            max_length=30,
            temperature=1,
            top_p=0.95,
            min_probability=0.001
        )
        generated_sequences.append(seq)
        print(f"  Sequence {i+1}: {seq}")

    # Save generated sequences
    generated_df = pd.DataFrame({
        'Sequence_ID': [f'Generated_{seed}_{i+1}' for i in range(num_samples)],
        'Sequence': generated_sequences,
        'Condition_MIC': [0.3] * num_samples,
        'Condition_a': [1.0] * num_samples,
        'Condition_t': [0.0] * num_samples,
        'Condition_AIP': [1.0] * num_samples
    })
    generated_df.to_csv(os.path.join(result_save_dir, f"generated_sequences_seed_{seed}.csv"), index=False)


    # 13. Save training summary
    summary = {
        'seed': seed,
        'input_dim': 30 * len(AMINO_ACIDS),
        'max_len': 30,
        'vocab_size': len(AMINO_ACIDS),
        'kl_anneal_epochs': train_params['kl_anneal_epochs'],
        'teacher_forcing_ratio': train_params['teacher_forcing_ratio'],
        'tf_decay': train_params['tf_decay'],
        'num_epochs': 200,
        'batch_size': 512,
        'learning_rate': 1e-4,
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'test_samples': len(X_test),
        'final_train_loss': train_losses[-1] if train_losses else None,
        'final_val_loss': val_losses[-1] if val_losses else None,
        'best_epoch': checkpoint.get('epoch', 'unknown'),
        'best_val_loss': checkpoint.get('val_loss', checkpoint.get('train_loss', 'unknown')),
        'loss_weights': loss_weights,
        'test_loss': test_loss,
    }
    with open(os.path.join(result_save_dir, f"training_summary_seed_{seed}.txt"), 'w') as f:
        for k, v in summary.items():
            f.write(f"{k}: {v}")

    print("Training complete!")


if __name__ == "__main__":
    main(seed=666)
