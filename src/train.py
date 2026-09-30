"""
Improved training script - adapted for LSTM decoder CVAE
Main improvements:
- Use CrossEntropyLoss for sequence reconstruction loss
- Introduce Scheduled Sampling to mitigate exposure bias
- Add KL annealing to prevent posterior collapse
- Gradient clipping to stabilize LSTM training
- Automatically save best model and training curves
"""

from __future__ import annotations

import os
import random
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Ensure importing the LSTM version of ConditionalVAE from the correct module
from model import ConditionalVAE
from preprocessing import AMINO_ACIDS, index_to_aa_mapping


def set_seed(seed: int):
    """Fix random seed"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def cvae_total_loss(
    decoded_logits,        # [batch, max_len, vocab_size] unnormalized logits
    x,                     # [batch, input_dim] flattened one-hot vectors
    mu, logvar,
    mic_pred, mic_target,
    a_pred, a_target,
    t_pred, t_target,
    aip_pred, aip_target,
    max_len: int = 30,
    vocab_size: int = 20,
    alpha: float = 1.0,    # Sequence reconstruction weight
    beta: float = 0.5,     # KLD weight (can be annealed)
    gamma_mic: float = 10.0,
    gamma_a: float = 1.0,
    gamma_t: float = 1.0,
    gamma_aip: float = 1.0,
    joint_reward_gamma: float = 0.0,  # 0 = disable per-sample joint-goodness reweighting
) -> tuple[torch.Tensor, dict]:
    """
    Compute CVAE total loss.
    - Sequence reconstruction: cross-entropy loss (more suitable for classification tasks than BCE)
    - KLD: standard Gaussian KL divergence
    - Four property prediction losses

    When joint_reward_gamma > 0, each sample is weighted by
        w_i = 1 + joint_reward_gamma * goodness_i
    where goodness_i in [0,1] scores how jointly "good" a sample is
    (low MIC, a=1, t=0, AIP=1). This boosts learning on jointly-good sequences.
    """
    batch_size = x.size(0)

    # 1. Sequence reconstruction loss (CrossEntropy)
    # Reshape flattened one-hot x to shape [batch, max_len, vocab_size]
    x_reshaped = x.view(batch_size, max_len, vocab_size)
    # Get true amino acid indices [batch, max_len]
    target_indices = x_reshaped.argmax(dim=-1)

    # Cross-entropy requires logits shape [batch * max_len, vocab_size], target shape [batch * max_len]
    ce_loss = F.cross_entropy(
        decoded_logits.reshape(-1, vocab_size),
        target_indices.reshape(-1),
        reduction='none'
    )  # [batch, max_len]

    # 2. KLD loss
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # 3. MIC loss (MSE, continuous values)
    mic_loss = F.mse_loss(mic_pred, mic_target, reduction='none')  # [batch, 1]

    # 4. Three binary classification label losses (BCE)
    a_loss = F.binary_cross_entropy(a_pred, a_target, reduction='none')  # [batch, 1]
    t_loss = F.binary_cross_entropy(t_pred, t_target, reduction='none')  # [batch, 1]
    aip_loss = F.binary_cross_entropy(aip_pred, aip_target, reduction='none')  # [batch, 1]

    if joint_reward_gamma > 0 and batch_size > 0:
        # Joint goodness of each sample (targets: mic normalized lower-better, a=1, t=0, aip=1)
        mic_t = mic_target.clamp(0.0, 1.0)
        goodness = (1.0 - mic_t + a_target + (1.0 - t_target) + aip_target) / 4.0  # [batch,1]
        w = 1.0 + joint_reward_gamma * goodness  # [batch,1]

        ce_w = ce_loss.view(batch_size, max_len).sum(dim=1).unsqueeze(1)  # [batch,1]
        total_loss = (alpha * (ce_w * w).sum() +
                      beta * kld_loss +
                      gamma_mic * (mic_loss * w).sum() +
                      gamma_a * (a_loss * w).sum() +
                      gamma_t * (t_loss * w).sum() +
                      gamma_aip * (aip_loss * w).sum())
    else:
        total_loss = (alpha * ce_loss.sum() +
                      beta * kld_loss +
                      gamma_mic * mic_loss.sum() +
                      gamma_a * a_loss.sum() +
                      gamma_t * t_loss.sum() +
                      gamma_aip * aip_loss.sum())

    ce_loss_sum = ce_loss.sum().item()
    mic_loss_sum = mic_loss.sum().item()
    a_loss_sum = a_loss.sum().item()
    t_loss_sum = t_loss.sum().item()
    aip_loss_sum = aip_loss.sum().item()

    loss_dict = {
        'total': total_loss.item(),
        'recon': ce_loss_sum,
        'kld': kld_loss.item(),
        'mic': mic_loss_sum,
        'a': a_loss_sum,
        't': t_loss_sum,
        'aip': aip_loss_sum
    }
    return total_loss, loss_dict


def adapt_state_dict_for_input_dim(state_dict: dict, new_input_dim: int) -> dict:
    """Adapt a checkpoint trained with a different max_len (input_dim).

    Only encoder.0.weight depends on input_dim (= max_len * vocab_size).
    One-hot layout is residue-major and sequences are left-aligned, so
    truncating columns to the first new_input_dim values is lossless for
    the kept residues (dropped columns correspond to removed tail positions).
    """
    key = "encoder.0.weight"
    if key in state_dict and state_dict[key].shape[1] != new_input_dim:
        old = state_dict[key]
        sd = {k: v.clone() for k, v in state_dict.items()}
        sd[key] = old[:, :new_input_dim].clone()
        print(f"  Adapted {key}: {tuple(old.shape)} -> {tuple(sd[key].shape)} "
              f"(input_dim {old.shape[1]} -> {new_input_dim})")
        return sd
    return state_dict


def train_cvae(
    X_train: np.ndarray,            # [N, input_dim] flattened one-hot
    y_1_train: np.ndarray,          # predicted_MIC (continuous values)
    y_2_train: np.ndarray,          # Predicted_Label_a (0/1)
    y_3_train: np.ndarray,          # Predicted_Label_t (0/1)
    y_4_train: np.ndarray,          # predicted_AIP (0/1)
    mic_scaler: MinMaxScaler = None,
    learning_rate: float = 1e-4,
    num_epochs: int = 100,
    batch_size: int = 512,
    device: torch.device = None,
    max_len: int = 30,
    vocab_size: int = 20,
    input_dim: int = 400,           # max_len * vocab_size
    save_dir: str = "./weights/",
    save_every: int = 10,
    X_val: np.ndarray = None,
    y_val: np.ndarray = None,       # Validation set labels, shape [N, 4] corresponding to MIC, A, T, AIP
    loss_weights: dict = None,
    kl_anneal_epochs: int = 10,     # KL annealing epochs
    teacher_forcing_ratio: float = 1.0,
    tf_decay: float = 0.98,         # Decay coefficient per epoch
    init_checkpoint: str = None,    # Warm start: load model weights from a previous best model
    early_stop_patience: int = None,# Stop if val loss does not improve for N epochs (None = disabled)
    joint_reward_gamma: float = 0.0,# Per-sample joint-goodness reweighting strength (0 = disabled)
) -> tuple:
    """
    Train CVAE model with LSTM decoder.
    
    Parameters:
        X_train, y_*_train: Training data
        mic_scaler: MIC normalizer (if None, will auto-fit)
        save_dir: Directory for saving models and plots
        X_val, y_val: Optional validation set
        loss_weights: Custom loss weight dictionary
        kl_anneal_epochs: Number of epochs for KL weight to linearly increase from 0 to beta
        teacher_forcing_ratio: Initial teacher forcing ratio
        tf_decay: teacher_forcing_ratio *= tf_decay after each epoch
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "checkpoints"), exist_ok=True)

    # ---------- Data preprocessing ----------
    # MIC normalization
    if mic_scaler is None:
        mic_scaler = MinMaxScaler()
        y_1_train_norm = mic_scaler.fit_transform(y_1_train.reshape(-1, 1)).flatten()
    else:
        y_1_train_norm = mic_scaler.transform(y_1_train.reshape(-1, 1)).flatten()

    # Convert to tensors
    X_train_tensor = torch.from_numpy(X_train).float().to(device)
    y1_train_tensor = torch.from_numpy(y_1_train_norm).float().unsqueeze(1).to(device)
    y2_train_tensor = torch.from_numpy(y_2_train.astype(float)).float().unsqueeze(1).to(device)
    y3_train_tensor = torch.from_numpy(y_3_train.astype(float)).float().unsqueeze(1).to(device)
    y4_train_tensor = torch.from_numpy(y_4_train.astype(float)).float().unsqueeze(1).to(device)

    train_dataset = TensorDataset(X_train_tensor, y1_train_tensor, y2_train_tensor, y3_train_tensor, y4_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Validation set processing
    has_validation = X_val is not None and y_val is not None
    if has_validation:
        y_val_mic_norm = mic_scaler.transform(y_val[:, 0:1]).flatten()
        X_val_tensor = torch.from_numpy(X_val).float().to(device)
        y1_val_tensor = torch.from_numpy(y_val_mic_norm).float().unsqueeze(1).to(device)
        y2_val_tensor = torch.from_numpy(y_val[:, 1].astype(float)).float().unsqueeze(1).to(device)
        y3_val_tensor = torch.from_numpy(y_val[:, 2].astype(float)).float().unsqueeze(1).to(device)
        y4_val_tensor = torch.from_numpy(y_val[:, 3].astype(float)).float().unsqueeze(1).to(device)

    # ---------- Loss weights ----------
    if loss_weights is None:
        loss_weights = {
            'alpha': 1.0,       # Reconstruction
            'beta': 0.5,        # KLD (will be overridden by annealing)
            'gamma_mic': 10.0,
            'gamma_a': 1.0,
            'gamma_t': 1.0,
            'gamma_aip': 1.0,
        }

    # ---------- Model initialization ----------
    model = ConditionalVAE(
        label_dims=[1, 1, 1, 1],
        input_dim=input_dim,
        num_hidden=32,          # Recommended 32~64
        dropout=0.2,
        max_len=max_len,
        vocab_size=vocab_size
    ).to(device)

    # ---------- Warm start from a previous checkpoint ----------
    if init_checkpoint:
        if not os.path.exists(init_checkpoint):
            raise FileNotFoundError(f"init_checkpoint not found: {init_checkpoint}")
        ckpt_init = torch.load(init_checkpoint, map_location=device, weights_only=False)
        state = adapt_state_dict_for_input_dim(ckpt_init["model_state_dict"], input_dim)
        model.load_state_dict(state)
        print(f"Warm-started model weights from: {init_checkpoint} "
              f"(checkpoint epoch={ckpt_init.get('epoch', 'N/A')}, "
              f"input_dim {ckpt_init.get('input_dim', 'N/A')} -> {input_dim})")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Record training process
    train_losses = []
    val_losses = [] if has_validation else None
    train_components = {'recon': [], 'kld': [], 'mic': [], 'a': [], 't': [], 'aip': []}
    val_components = {'recon': [], 'kld': [], 'mic': [], 'a': [], 't': [], 'aip': []} if has_validation else None
    best_val_loss = float('inf')
    best_epoch = 0
    es_counter = 0

    current_tf_ratio = teacher_forcing_ratio

    # ---------- Training loop ----------
    for epoch in range(num_epochs):
        model.train()
        epoch_total_loss = 0.0
        epoch_comp = {k: 0.0 for k in train_components.keys()}

        # KL annealing coefficient
        beta_annealed = min(1.0, epoch / kl_anneal_epochs) * loss_weights['beta']

        for batch in train_loader:
            x_batch, y1_batch, y2_batch, y3_batch, y4_batch = [b.to(device) for b in batch]

            # Forward pass (pass in teacher_forcing_ratio)
            _, decoded_logits, mu, log_var, _, mic_pred, a_pred, t_pred, aip_pred = model(
                x_batch, y1_batch, y2_batch, y3_batch, y4_batch,
                teacher_forcing_ratio=current_tf_ratio
            )

            loss, loss_dict = cvae_total_loss(
                decoded_logits, x_batch, mu, log_var,
                mic_pred, y1_batch,
                a_pred, y2_batch,
                t_pred, y3_batch,
                aip_pred, y4_batch,
                max_len=max_len,
                vocab_size=vocab_size,
                alpha=loss_weights['alpha'],
                beta=beta_annealed,
                gamma_mic=loss_weights['gamma_mic'],
                gamma_a=loss_weights['gamma_a'],
                gamma_t=loss_weights['gamma_t'],
                gamma_aip=loss_weights['gamma_aip'],
                joint_reward_gamma=joint_reward_gamma,
            )

            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping to prevent LSTM gradient explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_total_loss += loss.item()
            for k in epoch_comp:
                epoch_comp[k] += loss_dict[k]

        # Compute average loss (normalized by number of samples)
        avg_train_loss = epoch_total_loss / len(train_dataset)
        train_losses.append(avg_train_loss)
        for k in train_components:
            train_components[k].append(epoch_comp[k] / len(train_dataset))

        # ---------- Validation ----------
        if has_validation:
            model.eval()
            with torch.no_grad():
                # No teacher forcing during validation (teacher_forcing_ratio=0.0)
                _, val_logits, mu_val, logvar_val, _, mic_val, a_val, t_val, aip_val = model(
                    X_val_tensor, y1_val_tensor, y2_val_tensor, y3_val_tensor, y4_val_tensor,
                    teacher_forcing_ratio=0.0
                )
                val_loss, val_dict = cvae_total_loss(
                    val_logits, X_val_tensor, mu_val, logvar_val,
                    mic_val, y1_val_tensor,
                    a_val, y2_val_tensor,
                    t_val, y3_val_tensor,
                    aip_val, y4_val_tensor,
                    max_len=max_len,
                    vocab_size=vocab_size,
                    alpha=loss_weights['alpha'],
                    beta=loss_weights['beta'],   # Use full beta during validation
                    gamma_mic=loss_weights['gamma_mic'],
                    gamma_a=loss_weights['gamma_a'],
                    gamma_t=loss_weights['gamma_t'],
                    gamma_aip=loss_weights['gamma_aip'],
                    joint_reward_gamma=joint_reward_gamma,
                )
                avg_val_loss = val_loss.item() / len(X_val)
                val_losses.append(avg_val_loss)
                for k in val_components:
                    val_components[k].append(val_dict[k] / len(X_val))

                # Save best model (based on validation loss)
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_epoch = epoch
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': avg_val_loss,
                        'mic_scaler': mic_scaler,
                        'input_dim': input_dim,
                        'max_len': max_len,
                        'vocab_size': vocab_size,
                        'label_dims': [1, 1, 1, 1],
                    }, os.path.join(save_dir, "best_model.pth"))
                    print(f"  >>> Best model saved (val_loss={avg_val_loss:.4f})")
                else:
                    es_counter += 1
        else:
            # Use training loss as best metric when no validation set
            if avg_train_loss < best_val_loss:
                best_val_loss = avg_train_loss
                best_epoch = epoch
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': avg_train_loss,
                    'mic_scaler': mic_scaler,
                    'input_dim': input_dim,
                    'max_len': max_len,
                    'vocab_size': vocab_size,
                    'label_dims': [1, 1, 1, 1],
                }, os.path.join(save_dir, "best_model.pth"))
                print(f"  >>> Best model saved (train_loss={avg_train_loss:.4f})")

        # Periodically save checkpoints
        if (epoch + 1) % save_every == 0 or epoch == num_epochs - 1:
            checkpoint_path = os.path.join(save_dir, "checkpoints", f"epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss if has_validation else None,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_components': train_components,
                'val_components': val_components,
                'mic_scaler': mic_scaler,
                'teacher_forcing_ratio': current_tf_ratio,
            }, checkpoint_path)
            print(f"  Checkpoint saved: {checkpoint_path}")

        # Print progress
        log_msg = f"Epoch {epoch+1:3d}/{num_epochs} | train_loss={avg_train_loss:.4f}"
        if has_validation:
            log_msg += f" | val_loss={avg_val_loss:.4f}"
        log_msg += f" | TF ratio={current_tf_ratio:.2f}"
        print(log_msg)
        print(f"    Recon: {train_components['recon'][-1]:.2f} | KLD: {train_components['kld'][-1]:.2f} | "
              f"MIC: {train_components['mic'][-1]:.2f} | A: {train_components['a'][-1]:.2f} | "
              f"T: {train_components['t'][-1]:.2f} | AIP: {train_components['aip'][-1]:.2f}")

        # Decay teacher forcing ratio
        current_tf_ratio *= tf_decay

        # ---------- Early stopping ----------
        if has_validation and early_stop_patience is not None:
            if es_counter >= early_stop_patience:
                print(f"Early stopping triggered at epoch {epoch+1}: "
                      f"no val_loss improvement for {early_stop_patience} epochs "
                      f"(best={best_val_loss:.4f} @ epoch {best_epoch+1})")
                break

    # ---------- Plot training curves ----------
    plot_training_curve(
        train_losses, val_losses, save_dir, best_epoch,
        train_components, val_components
    )

    # Save final model
    final_model_path = os.path.join(save_dir, "final_model.pth")
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_components': train_components,
        'val_components': val_components,
        'mic_scaler': mic_scaler,
        'input_dim': input_dim,
        'max_len': max_len,
        'vocab_size': vocab_size,
        'label_dims': [1, 1, 1, 1],
    }, final_model_path)
    print(f"Final model saved: {final_model_path}")
    print(f"Best epoch: {best_epoch+1} with loss {best_val_loss:.4f}")

    return model, mic_scaler, train_losses, val_losses, train_components, val_components


def plot_training_curve(
    train_losses: list,
    val_losses: list | None,
    save_dir: str,
    best_epoch: int | None = None,
    train_components: dict | None = None,
    val_components: dict | None = None
):
    """Plot detailed training curves (total loss + individual components)"""
    epochs = range(1, len(train_losses) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Total loss
    ax1 = axes[0, 0]
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    if val_losses is not None:
        ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    if best_epoch is not None:
        ax1.axvline(x=best_epoch + 1, color='g', linestyle='--', alpha=0.7, label=f'Best ({best_epoch+1})')
    ax1.set_title('Total Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Reconstruction loss + KLD
    ax2 = axes[0, 1]
    if train_components is not None:
        ax2.plot(epochs, train_components['recon'], 'b-', label='Train Recon', alpha=0.7)
        ax2.plot(epochs, train_components['kld'], 'b--', label='Train KLD', alpha=0.7)
    if val_components is not None:
        ax2.plot(epochs, val_components['recon'], 'r-', label='Val Recon', alpha=0.7)
        ax2.plot(epochs, val_components['kld'], 'r--', label='Val KLD', alpha=0.7)
    ax2.set_title('Reconstruction & KLD Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # MIC loss
    ax3 = axes[1, 0]
    if train_components is not None:
        ax3.plot(epochs, train_components['mic'], 'b-', label='Train MIC')
    if val_components is not None:
        ax3.plot(epochs, val_components['mic'], 'r-', label='Val MIC')
    ax3.set_title('MIC Prediction Loss (MSE)')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Three binary classification label losses
    ax4 = axes[1, 1]
    if train_components is not None:
        ax4.plot(epochs, train_components['a'], 'b-', label='Train A', alpha=0.7)
        ax4.plot(epochs, train_components['t'], 'b--', label='Train T', alpha=0.7)
        ax4.plot(epochs, train_components['aip'], 'b:', label='Train AIP', alpha=0.7)
    if val_components is not None:
        ax4.plot(epochs, val_components['a'], 'r-', label='Val A', alpha=0.7)
        ax4.plot(epochs, val_components['t'], 'r--', label='Val T', alpha=0.7)
        ax4.plot(epochs, val_components['aip'], 'r:', label='Val AIP', alpha=0.7)
    ax4.set_title('Binary Label Losses (BCE)')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(save_dir, "training_curve.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Training curve saved to {plot_path}")
    plt.show()


def load_model(checkpoint_path: str, device: torch.device = None):
    """Load a trained model"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = ConditionalVAE(
        label_dims=checkpoint.get('label_dims', [1, 1, 1, 1]),
        input_dim=checkpoint['input_dim'],
        num_hidden=32,                 # Must match training, or read from checkpoint (recommended to save)
        max_len=checkpoint.get('max_len', 30),
        vocab_size=checkpoint.get('vocab_size', 20)
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    mic_scaler = checkpoint.get('mic_scaler', None)
    print(f"Model loaded from {checkpoint_path}")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Loss: {checkpoint.get('val_loss', checkpoint.get('train_loss', 'N/A')):.4f}")
    return model, mic_scaler, checkpoint
