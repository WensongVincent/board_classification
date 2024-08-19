import json
from pathlib import Path

import torch
from torch import nn
import torchvision
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from BoardClassification import BoardClssifier, build_model
from BoardDataset import BoardDataset
from utils import save_checkpoint, save_validation_results
from test_main import test


##############
# main function
def main(config_path):
    ########## loading config ##########
    config_path = Path(config_path)
    config = None
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    save_path = config["general"]["save_path"]

    model_num_classes = config["model"]["num_classes"]
    if "scale" in config["model"]:
        model_scale = config["model"]["scale"]
    if "ckpt_path" in config["model"]:
        model_ckpt_path = config["model"]["ckpt_path"]

    test_meta_dir = config["test_dataset"]["path"]
    test_shuffle = config["test_dataset"]["shuffle"]
    test_num_workers = config["test_dataset"]["num_workers"]

    test_batch_size = config["test"]["batch_size"]


    ########## end loading config ##########

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # data processing
    # Define transformations
    data_transforms = {
        'val': transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }


    # Create datasets
    print(f"Using test dataset from {test_meta_dir}")
    val_dataset = BoardDataset(test_meta_dir, transform=data_transforms['val'])

    # Create WeightedRandomSampler
    # sample_weights = [train_dataset.class_weights[label] for label in train_dataset.labels]
    # sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)


    # Create dataloaders
    val_loader = DataLoader(val_dataset, batch_size=test_batch_size, shuffle=test_shuffle, num_workers=test_num_workers)

    # Data loader dictionary
    dataloader = {
        'val': val_loader
    }

    # Dataset sizes
    dataset_sizes = {
        'val': len(val_dataset)
    }

    print(f"Validation dataset size: {dataset_sizes['val']}")


    # model 
    print(f"Reading checkpoint from {model_ckpt_path}")
    model = build_model(model_ckpt_path)

    # distributed launch
    # model = nn.DataParallel(model)

    # finetune
    test(model, dataloader, dataset_sizes, device, save_path)
    print(f"Test result are saved to: {save_path}")


if __name__ == '__main__':
    ####### CHANGE this for different training ######
    main('/mnt/afs/huwensong/workspace/R4_board_classification/config/config_0813_3_train.json')