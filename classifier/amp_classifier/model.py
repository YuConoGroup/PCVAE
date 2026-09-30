import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
class ProteinDataset(Dataset):
    def __init__(self, features, labels):

        # Initialization function for creating class instances
        self.features = features  # Assign input features to instance variable features
        self.labels = labels  # Assign input labels to instance variable labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        # Get data by index idx
        return {
            'features': torch.tensor(self.features[idx], dtype=torch.float32),  # Convert features to torch tensor with float32 dtype
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)  # Convert labels to torch tensor with long dtype
        }
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        # Initialize ChannelAttention class, in_planes is input channels, ratio is compression ratio
        super(ChannelAttention, self).__init__()
        # Call parent class ChannelAttention initialization method
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        # Define adaptive average pooling layer with pool size 1
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        # Define adaptive max pooling layer with pool size 1
        self.fc = nn.Sequential(
            nn.Conv1d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv1d(in_planes // ratio, in_planes, 1, bias=False)
        )
        # Define fully connected layers: first layer compresses input channels to input_channels//16, second layer restores to input channels
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        # Apply average pooling to input x and pass through fully connected layer fc
        avg_out = self.fc(self.avg_pool(x))
        # Apply max pooling to input x and pass through fully connected layer fc
        max_out = self.fc(self.max_pool(x))
        # Add average pooling and max pooling results
        out = avg_out + max_out
        # Apply sigmoid activation to the summed result
        return self.sigmoid(out)
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv1d(2, 1, kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_planes, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
    def forward(self, x):
        out = x * self.channel_attention(x)
        out = out * self.spatial_attention(out)
        return out
class stage_1model(nn.Module):
    def __init__(self, max_time_steps, input_size):
        super(stage_1model, self).__init__()
        self.conv1d_1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv1d_2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.1)
        self.bilstm1 = nn.LSTM(input_size=128, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True)
        self.cbam = CBAM(in_planes=256)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear((max_time_steps // 4) * 256, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 2)
    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.conv1d_1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv1d_2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = x.permute(0, 2, 1)  # Change back to (batch_size, max_time_steps, new_input_size)
        x = self.dropout(x)
        x, _ = self.bilstm1(x)
        x = self.dropout(x)
        x = x.permute(0, 2, 1)  # Change to (batch_size, new_input_size, max_time_steps) for CBAM
        x = self.cbam(x)
        x = x.permute(0, 2, 1)  # Change back to (batch_size, max_time_steps, new_input_size)
        x = self.flatten(x)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = self.fc4(x)
        return x


class stage_2model(nn.Module):
    def __init__(self, input_size, num_classes):
        super(stage_2model, self).__init__()

        self.conv1d_1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv1d_2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.3)


        self.bilstm = nn.LSTM(input_size=64, hidden_size=64, num_layers=1, batch_first=True, bidirectional=True)
        self.cbam = CBAM(in_planes=128)

        self.flatten = nn.Flatten()


        self.fc1 = nn.Linear(input_size * 128, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.conv1d_1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv1d_2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = x.permute(0, 2, 1)
        x = self.dropout(x)
        x, _ = self.bilstm(x)
        x = self.dropout(x)
        x = x.permute(0, 2, 1)
        x = self.cbam(x)
        x = x.permute(0, 2, 1)
        x = self.flatten(x)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x
