import torch
import torch.nn as nn


# ============================================================
# Prepare the CNN model
# ============================================================

class WeatherCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.model = nn.Sequential(

            # 13 → 64
            nn.Conv2d(
                in_channels=13,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),

            # 64 → 64
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),

            # 64 → 32
            nn.Conv2d(
                in_channels=64,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),

            # 32 → 4 forecast steps
            nn.Conv2d(
                in_channels=32,
                out_channels=4,
                kernel_size=1
            )
        )

    def forward(self, x):

        return self.model(x)