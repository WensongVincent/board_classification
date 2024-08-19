import copy
from pathlib import Path, PosixPath
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from utils import save_checkpoint, save_validation_results



# training code
def train_val_main(model: nn.Module, 
          dataloader: DataLoader, 
          dataset_sizes, 
          criterion, 
          optimizer, 
          device, 
          num_epochs=10,
          result_path=None,
          scheduler=None):
    
    if result_path is None:
        raise ValueError('Result path should not be None')

    best_acc = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc_loss = 0
    best_epoch = 0
    
    if num_epochs == 0:
        model.eval()
        running_corrects = 0

        # Collect validation results
        val_results = []
        val_results_timestamp = []

        # Iterate over data.
        for inputs, labels, image_paths in tqdm(dataloader['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # forward
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            # Collect results
            for i in range(inputs.size(0)):
                time_stamp = PosixPath(image_paths[i]).parts[-2]
                predicted_label = preds[i].item()
                gt_label = labels[i].item()
                if time_stamp not in val_results_timestamp:
                    val_results.append((time_stamp, gt_label, predicted_label, image_paths[i]))
                    val_results_timestamp.append(time_stamp)

            # statistics
            running_corrects += torch.sum(preds == labels.data)

        best_acc = running_corrects.double() / dataset_sizes['val']

        # Save validation results
        save_validation_results(val_results, f'{result_path}/test_results.txt')

        epoch_loss = None
    
    else:
        # looping epochs
        for epoch in range(num_epochs):
            print(f'Epoch {epoch} / {num_epochs - 1}')
            print('-' * 10)
            
            # training and validation
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0

                # Collect validation results
                val_results = []
                val_results_timestamp = []

                # Iterate over data
                for inputs, labels, image_paths in tqdm(dataloader[phase]):
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the gradient in gradient flow
                    optimizer.zero_grad()
                    
                    # forward pass
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backwards + backprop optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                    
                        # Collect results if in validation phase
                        if phase == 'val':
                            for i in range(inputs.size(0)):
                                time_stamp = PosixPath(image_paths[i]).parts[-2]
                                predicted_label = preds[i].item()
                                gt_label = labels[i].item()
                                if time_stamp not in val_results_timestamp:
                                    val_results.append((time_stamp, gt_label, predicted_label, image_paths[i]))
                                    val_results_timestamp.append(time_stamp)
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

                # Save validation results
                # if phase == 'val':
                #     save_validation_results(val_results, f'{RESULT_PATH}/epoch{epoch}_validation_results.txt')

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_acc_loss = epoch_loss
                    best_epoch = epoch
                    best_model_wts = copy.deepcopy(model.state_dict())
                    save_validation_results(val_results, f'{result_path}/best_validation_results.txt')
            
            # update learning rate
            if scheduler is not None:
                scheduler.step()

    print(f'Best val Epoch: {best_epoch}')
    print('Best val Acc: {:4f}'.format(best_acc))
    print('Best val Acc loss: {:4f}'.format(best_acc_loss))

    # load best model weights
    if num_epochs > 0:
        model.load_state_dict(best_model_wts)
    return model, {'epoch': best_epoch, 'acc': best_acc, 'loss': best_acc_loss} #best_acc_loss