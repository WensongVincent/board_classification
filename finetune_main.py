import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from train_main import train_val_main as train


def finetune(model: nn.Module, dataloader: DataLoader, dataset_sizes, configs, device, result_path=None):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=configs['lr'], weight_decay=configs['weight_decay'])
    if 'lr_decay_step' in configs:
        if 'lr_decay' in configs:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=configs['lr_decay_step'], gamma=configs['lr_decay'])
        else:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=configs['lr_decay_step'])
    else:
        scheduler = None
    model, stat = train(model, dataloader, dataset_sizes, criterion, optimizer, device, configs['num_epochs'], result_path, scheduler)
    return model, optimizer, stat


