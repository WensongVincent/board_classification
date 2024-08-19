import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from train_main import train_val_main as val

def test(model: nn.Module, dataloader: DataLoader, dataset_sizes, device, result_path=None):
    model = model.to(device)
    _, _ = val(model, dataloader, dataset_sizes, criterion=None, optimizer=None, device=device, num_epochs=0, result_path=result_path)
