import torch
import torch.nn as nn

# Double conv block for UNet
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU()
        )
    def forward(self, x):
        return self.conv(x)

# Simple UNet
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, 64)
        self.pool = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(64, 128)
        self.up = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64)
        self.final = nn.Conv2d(64, out_channels, 1)
        
    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.pool(x1)
        x3 = self.enc2(x2)
        x4 = self.up(x3)
        x5 = torch.cat([x4, x1], dim=1)
        out = self.dec1(x5)
        return torch.sigmoid(self.final(out))

# 🔥 LSTM block
class TimeLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=3, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_dim, 128)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # x shape: (batch, time, features)
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # take last time step
        out = self.relu(self.fc(out))
        return out

# 🔥 Full upgraded fire prediction model
class FireSpreadModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.unet = UNet(in_channels=3, out_channels=1)
        self.lstm = TimeLSTM(input_dim=2, hidden_dim=256, num_layers=3, dropout=0.4)
        self.fc_combined = nn.Linear(128 + 1, 1)
        
    def forward(self, raster, timeseries):
        # raster: [B, 3, H, W]
        unet_out = self.unet(raster)
        unet_pool = torch.mean(unet_out.view(unet_out.size(0), -1), dim=1, keepdim=True)
        
        # timeseries: [B, T, 2]
        lstm_out = self.lstm(timeseries)
        
        combined = torch.cat([unet_pool, lstm_out], dim=1)
        fire_prob = torch.sigmoid(self.fc_combined(combined))
        return fire_prob, unet_out
