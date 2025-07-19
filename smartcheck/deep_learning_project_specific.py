import torch
from torch.utils.data import Dataset
import torch.nn as nn


class HourlyCounterDataset(Dataset):
    def __init__(self, y_series, exog_array, input_window, forecast_horizon):
        self.y = y_series  # 1D array
        self.exog = exog_array  # 2D array (num_steps, num_exog_features)
        self.input_window = input_window
        self.forecast_horizon = forecast_horizon

    def __len__(self):
        return len(self.y) - self.input_window - self.forecast_horizon

    def __getitem__(self, idx):
        X_y = self.y[idx:idx + self.input_window]
        X_exo = self.exog[idx:idx + self.input_window, :]
        y_target = self.y[
            idx + self.input_window:
            idx + self.input_window + self.forecast_horizon
        ]
        return (
            torch.tensor(X_y, dtype=torch.float32),
            torch.tensor(X_exo, dtype=torch.float32),
            torch.tensor(y_target, dtype=torch.float32)
        )


class CNNForecastWithExog(nn.Module):
    def __init__(self, input_window, exog_dim, output_horizon):
        super().__init__()
        self.conv_y = nn.Conv1d(1, 16, kernel_size=3)
        self.conv_exo = nn.Conv1d(exog_dim, 16, kernel_size=3)
        conv_out_size = input_window - 2  # assuming kernel_size=3

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2 * 16 * conv_out_size, 64),
            nn.ReLU(),
            nn.Linear(64, output_horizon)
        )

    def forward(self, x_y, x_exo):
        # x_y: (batch, seq_len) → reshape to (batch, 1, seq_len)
        x_y = x_y.unsqueeze(1)
        x_exo = x_exo.permute(0, 2, 1)  # to (batch, exog_dim, seq_len)

        y_feat = self.conv_y(x_y)
        exo_feat = self.conv_exo(x_exo)

        combined = torch.cat([y_feat, exo_feat], dim=1)
        out = self.fc(combined)
        return out
