# Operational Project Blueprint: Training a Simple Weather AI

This blueprint provides the structure for the "Lab" section of the textbook.

## Project: Downscaling Temperature using CNNs

### 1. Data Acquisition
- **Source**: ERA5 (Low-Res) and High-Res Surface Observations.
- **Tools**: `cdsapi` for python.

### 2. The Model (PyTorch Pseudocode)
```python
import torch.nn as nn

class WeatherDownscaler(nn.Module):
    def __init__(self):
        super().__init__()
        # Use a 3-layer CNN to capture spatial patterns
        self.network = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1)
        )

    def forward(self, x):
        # x: Low-res temperature grid
        return self.network(x)
```

### 3. Operational Deployment
- **Input**: GFS (Global Forecast System) 0.25-degree forecast.
- **Output**: 1km localized temperature forecast for urban heat island analysis.
- **Validation**: Compare against a held-out set of mountain/coastal stations where NWP typically fails.
