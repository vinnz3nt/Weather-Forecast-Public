import pickle
import math

import torch
from torch.utils.data import Dataset,Subset

from config import BASE_DIR, REPORT_FIGS, DATA_DIR, EXPERIMENTS_DIR


# print("Torch version:", torch.__version__)

def nanstd(tensor: torch.Tensor, dim=None, keepdim=False):
    """Compute standard deviation while ignoring NaNs (like numpy.nanstd)."""
    mask = ~torch.isnan(tensor)
    masked_tensor = torch.where(mask, tensor, torch.tensor(0.0, device=tensor.device, dtype=tensor.dtype))

    count = mask.sum(dim=dim, keepdim=keepdim)
    sum_ = masked_tensor.sum(dim=dim, keepdim=keepdim)
    mean = sum_ / count.clamp(min=1)

    squared_diff = torch.where(mask, (tensor - mean) ** 2, torch.tensor(0.0, device=tensor.device, dtype=tensor.dtype))
    sum_squared_diff = squared_diff.sum(dim=dim, keepdim=keepdim)
    std = torch.sqrt(sum_squared_diff / count.clamp(min=1))
    return std

def split_dataset(dataset, train_frac=0.7, val_frac=0.15, test_frac=0.15):
    total = len(dataset)
    train_end = int(total * train_frac)
    val_end = train_end + int(total * val_frac)

    indices = list(range(total))
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    return (Subset(dataset, train_indices),
            Subset(dataset, val_indices),
            Subset(dataset, test_indices))

class WeatherDataset(Dataset):
    def __init__(self, data_folder, locations, context_window, rollout_steps, stride, variables):
        self.context_window = context_window
        self.rollout_steps = rollout_steps
        self.stride = stride
        self.locations = locations
        self.variables = variables

        data = self._load_data(data_folder) # [T, N, F]
        data = self._add_fourier_features(data) # [T, N, F+4]
        self.data = self._normalize(data)  # [T, N, F+4]

        self.T, self.N, self.F = self.data.shape
        # self.samples = self.T - context_window
        # self.samples = self.T - self.context_window - self.rollout_steps #Gemini
        self.total_sequences = self.T - self.context_window - self.rollout_steps
        self.samples = self.total_sequences // self.stride

    def _load_data(self, folder):
        node_data = []
        for location in self.locations:
            path = folder / f"{location}_data"
            with open(path, 'rb') as f:
                d = pickle.load(f)  # [T, F] array
                d = d[self.variables]
                d = torch.tensor(d.values, dtype=torch.float32)
                node_data.append(d.unsqueeze(1))  # [T, 1, F]
        return torch.cat(node_data, dim=1)  # [T, N, F]
    
    def _add_fourier_features(self, data):
        T, N, F = data.shape
        time_idx = torch.arange(T, dtype=torch.float32)

        daily = 24.0
        yearly = 24.0 * 365.0

        daily_sin = torch.sin(2 * math.pi * time_idx / daily)
        daily_cos = torch.cos(2 * math.pi * time_idx / daily)
        yearly_sin = torch.sin(2 * math.pi * time_idx / yearly)
        yearly_cos = torch.cos(2 * math.pi * time_idx / yearly)

        fourier = torch.stack([daily_sin, daily_cos, yearly_sin, yearly_cos], dim=-1)
        fourier = fourier.unsqueeze(1).repeat(1, N, 1)
        return torch.cat([data, fourier], dim=-1)

    def _normalize(self, data):
        mean = torch.nanmean(data, dim=(0, 1), keepdim=True)
        std = nanstd(data, dim=(0, 1), keepdim=True)
        self.mean = mean
        self.std = std
        return (data - mean) / (std + 1e-6)

    def __len__(self):
        return self.samples

    def __getitem__(self, idx):
        # T,N,F = self.data
        actual_idx = idx * self.stride

        context = self.data[actual_idx : actual_idx + self.context_window] 

        targets_weather = self.data[actual_idx + self.context_window : 
                                    actual_idx + self.context_window + self.rollout_steps, :, :-4]
        targets_fourier = self.data[actual_idx + self.context_window : 
                                    actual_idx + self.context_window + self.rollout_steps, :, -4:]
        return context, targets_weather, targets_fourier



if __name__ == "__main__":
    # locations = ["Berlin", "Copenhagen", "Gdansk", "Helsinki", "Oslo", "Stockholm", "Sundsvall", "Trondheim", "Visby"]
    # data_folder = "DataCombined"
    # context_window = 40
    # batch_size = 16
    # dataset = WeatherDataset(data_folder, locations, context_window)
    # train_set, val_set, test_set = split_dataset(dataset, train_frac=0.7, val_frac=0.15, test_frac=0.15)

    # train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader(val_set, shuffle=False)
    # test_loader = DataLoader(test_set,batch_size=int(len(train_set)), shuffle=False)

    # for context, targets in train_loader:
    #     print("Context shape:", context.shape)  # [B, T_window, N, F]
    #     print("Target shape:", targets.shape)   # [B, H, N, F]
    #     break
    pass 