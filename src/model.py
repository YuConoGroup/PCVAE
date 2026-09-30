"""
Improved CVAE model - using LSTM decoder to enhance sequence quality
Main changes:
- Decoder replaced with two-layer LSTM, autoregressive sequence generation
- Teacher forcing during training
- Autoregressive sampling during inference
- Encoder and property prediction heads remain unchanged
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from preprocessing import AMINO_ACIDS  # Ensure correct import of amino acid list


class ConditionalVAE(nn.Module):
    def __init__(
        self,
        label_dims,                 # Dimension of each label, e.g. [1,1,1,1]
        input_dim: int = 600,       # Total input dimension = max_len * vocab_size
        num_hidden: int = 32,       # Latent variable dimension
        dropout: float = 0.2,
        max_len: int = 30,          # Maximum sequence length
        vocab_size: int = len(AMINO_ACIDS)  # Usually 20
    ):
        super().__init__()
        assert len(label_dims) == 4, "Requires 4 label dimensions"
        
        self.input_dim = input_dim
        self.num_hidden = num_hidden
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.label_dims = label_dims
        
        # ---------- Encoder (same as original) ----------
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, self.num_hidden),
            nn.ReLU(),
        )
        
        # VAE mean and variance layers
        self.mu = nn.Linear(self.num_hidden, self.num_hidden)
        self.log_var = nn.Linear(self.num_hidden, self.num_hidden)
        
        # ---------- Label fusion weights ----------
        self.weight_1 = nn.Parameter(torch.ones(1))
        self.weight_2 = nn.Parameter(torch.ones(1))
        self.weight_3 = nn.Parameter(torch.ones(1))
        self.weight_4 = nn.Parameter(torch.ones(1))

        # Label projectors (consistent with num_hidden)
        self.label_projector_1 = nn.Sequential(
            nn.Linear(label_dims[0], self.num_hidden),
            nn.ReLU(),
            nn.Linear(self.num_hidden, self.num_hidden),
            nn.ReLU()
        )
        self.label_projector_2 = nn.Sequential(
            nn.Linear(label_dims[1], self.num_hidden),
            nn.ReLU(),
            nn.Linear(self.num_hidden, self.num_hidden),
            nn.ReLU()
        )
        self.label_projector_3 = nn.Sequential(
            nn.Linear(label_dims[2], self.num_hidden),
            nn.ReLU(),
            nn.Linear(self.num_hidden, self.num_hidden),
            nn.ReLU()
        )
        self.label_projector_4 = nn.Sequential(
            nn.Linear(label_dims[3], self.num_hidden),
            nn.ReLU(),
            nn.Linear(self.num_hidden, self.num_hidden),
            nn.ReLU()
        )

        # ---------- LSTM decoder ----------
        self.lstm_hidden_size = 512
        # Map conditional latent vector to LSTM initial state (h0, c0)
        self.init_h = nn.Linear(self.num_hidden, self.lstm_hidden_size)
        self.init_c = nn.Linear(self.num_hidden, self.lstm_hidden_size)
        
        # Two-layer LSTM
        self.lstm = nn.LSTM(
            input_size=self.vocab_size,          # Amino acid probability distribution from previous time step (or true one-hot)
            hidden_size=self.lstm_hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=False
        )
        
        # Output projection to vocabulary size
        self.output_proj = nn.Linear(self.lstm_hidden_size, self.vocab_size)
        
        # ---------- Property prediction heads (same as original) ----------
        self.mic_predictor = nn.Sequential(
            nn.Linear(self.num_hidden, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
        self.a_predictor = nn.Sequential(
            nn.Linear(self.num_hidden, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        self.t_predictor = nn.Sequential(
            nn.Linear(self.num_hidden, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        self.aip_predictor = nn.Sequential(
            nn.Linear(self.num_hidden, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def condition_on_label_and_features(self, z, y_1, y_2, y_3, y_4):
        projected_label_1 = self.label_projector_1(y_1.float())
        projected_label_2 = self.label_projector_2(y_2.float())
        projected_label_3 = self.label_projector_3(y_3.float())
        projected_label_4 = self.label_projector_4(y_4.float())

        hid_fea = (
            z
            + self.weight_1 * projected_label_1
            + self.weight_2 * projected_label_2
            + self.weight_3 * projected_label_3
            + self.weight_4 * projected_label_4
        )
        return hid_fea

    def decode_sequence(self, z_cond, target_seq=None, teacher_forcing_ratio=0.0, temperature=1.0, return_logits=True):
        """
        Autoregressive sequence decoding
        Args:
            z_cond: [batch, num_hidden]
            target_seq: [batch, max_len, vocab_size] true sequence one-hot (provided during training)
            teacher_forcing_ratio: teacher forcing probability
            temperature: sampling temperature (only effective when return_logits=False)
            return_logits: if True, return logits; otherwise return softmax probabilities
        Returns:
            outputs: [batch, max_len, vocab_size] content determined by return_logits
        """
        batch_size = z_cond.size(0)
        device = z_cond.device

        h0 = self.init_h(z_cond).unsqueeze(0).repeat(2, 1, 1)   # [2, batch, lstm_hidden]
        c0 = self.init_c(z_cond).unsqueeze(0).repeat(2, 1, 1)
        hidden = (h0, c0)

        input_step = torch.zeros(batch_size, 1, self.vocab_size).to(device)
        outputs = []

        for t in range(self.max_len):
            out, hidden = self.lstm(input_step, hidden)
            logits = self.output_proj(out.squeeze(1))            # [batch, vocab_size]

            if return_logits:
                outputs.append(logits.unsqueeze(1))
                # Compute probability distribution (for next step input, preserving gradients)
                probs = F.softmax(logits, dim=-1)      # Note: no temperature division here
                if target_seq is not None and random.random() < teacher_forcing_ratio:
                    input_step = target_seq[:, t:t+1, :]
                else:
                    input_step = probs.unsqueeze(1)     # Use probability distribution, gradients can flow
            else:
                probs = F.softmax(logits / temperature, dim=-1)   # [batch, vocab_size]
                outputs.append(probs.unsqueeze(1))                # Collect probabilities
                # Next step input: use probability distribution during inference (smooth input)
                if target_seq is not None and random.random() < teacher_forcing_ratio:
                    input_step = target_seq[:, t:t+1, :]
                else:
                    if target_seq is not None:
                        # Scheduled sampling during training: use one-hot (consistent with training)
                        with torch.no_grad():
                            pred_indices = probs.argmax(dim=-1)
                            pred_onehot = F.one_hot(pred_indices, num_classes=self.vocab_size).float()
                            input_step = pred_onehot.unsqueeze(1)
                    else:
                        # Pure inference: use probability distribution as next step input
                        input_step = probs.unsqueeze(1)

        return torch.cat(outputs, dim=1)


    def forward(self, x, y_1, y_2, y_3, y_4, teacher_forcing_ratio=0.5, temperature=1.0, return_logits=True):
        encoded = self.encoder(x)
        mu = self.mu(encoded)
        log_var = self.log_var(encoded)
        z = self.reparameterize(mu, log_var)

        hid_fea = self.condition_on_label_and_features(z, y_1, y_2, y_3, y_4)

        batch_size = x.size(0)
        target_seq = x.view(batch_size, self.max_len, self.vocab_size) if teacher_forcing_ratio > 0 else None

        # Return logits during training, probabilities during inference (controlled by return_logits)
        # Automatically switch based on self.training, or you can explicitly specify when calling
        decoded = self.decode_sequence(
            hid_fea,
            target_seq=target_seq,
            teacher_forcing_ratio=teacher_forcing_ratio,
            temperature=temperature,
            return_logits=return_logits   # Training mode returns logits, evaluation mode returns probabilities
        )

        mic_pred = self.mic_predictor(hid_fea)
        a_pred = self.a_predictor(hid_fea)
        t_pred = self.t_predictor(hid_fea)
        aip_pred = self.aip_predictor(hid_fea)

        return encoded, decoded, mu, log_var, hid_fea, mic_pred, a_pred, t_pred, aip_pred

    def sample(self, num_samples, y_1, y_2, y_3, y_4, device=None, temperature=1.0):
        """
        Generate new samples
        Returns: [num_samples, max_len, vocab_size] probability distribution, can be further sampled/reconstructed
        """
        if device is None:
            device = next(self.parameters()).device
        with torch.no_grad():
            # Expand label dimensions to match number of samples
            if y_1.size(0) != num_samples:
                y_1 = y_1.repeat(num_samples, 1)
                y_2 = y_2.repeat(num_samples, 1)
                y_3 = y_3.repeat(num_samples, 1)
                y_4 = y_4.repeat(num_samples, 1)
            
            z = torch.randn(num_samples, self.num_hidden).to(device)
            hid_fea = self.condition_on_label_and_features(z, y_1, y_2, y_3, y_4)
            # Autoregressive generation (no target sequence)
            # return_logits=False -> return softmax probabilities as documented
            samples = self.decode_sequence(hid_fea, target_seq=None, temperature=temperature, return_logits=False)
        return samples
