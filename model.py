import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

# device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
# print(f"Using {device} device")

class NBAtorchnnModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(23, 512), # Initial input should always match amount of features from dataframe
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1)
        )
    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

# Tensors can be processed faster than np arrays, and are required to use pytorch models
def convert_to_tensor(data) -> torch.Tensor:
    if isinstance(data, pd.Series):
        data = data.values
    elif isinstance(data, pd.DataFrame):
        data = data.to_numpy()

    data = data.astype(np.float32)  # Explicitly cast to float32
    return torch.from_numpy(data)

# Tensor datasets allow you to treat the data as one single unit
def convert_to_tensor_dataset(x, y) -> TensorDataset:
    return TensorDataset(convert_to_tensor(x), convert_to_tensor(y))

