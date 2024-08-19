import os
from pathlib import Path, PosixPath

import torch


def save_validation_results(results, txt_output_file, image_output_dir = None):
    output_parent_path = str(PosixPath(txt_output_file).parent)
    os.makedirs(output_parent_path, exist_ok=True)
    with open(txt_output_file, 'w') as f:
        for time_stamp, gt_label, predicted_label, image_path in results:
            if gt_label != predicted_label:
                f.write(f"{time_stamp}, GT: {gt_label}, Pred: {predicted_label}, ImagePath: {image_path} \n")
                # if image_output_dir is not None:
                #     #######
    

def save_checkpoint(model, optimizer, epoch, loss, file_path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, file_path)
    print(f"Checkpoint saved at epoch {epoch} with loss {loss:.4f}")
