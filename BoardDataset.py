import json
from PIL import Image
import os
import cv2
import random

import torch
from torch import nn
import torch.utils
import torchvision
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


# metadata is a json file which have image dir

class Augment():
    def __init__(self, aug_meta_dirs) -> None: # [ [prob, dir], [...], ... ]
        
        # { "1": {"prob": 0.3, "aug_img_paths": [], "aug_mask_paths":[]},
        #   "2": {"prob": 0.3, "aug_img_paths": [], "aug_mask_paths":[]},
        #   ...}
        self.aug_dict = {}

        for i, category in enumerate(aug_meta_dirs):
            prob, category_meta_dir = category

            aug_img_paths = []
            aug_mask_paths = []
            with open(category_meta_dir, 'r') as f:
                for line in f:
                    time_stamp_dir = line.strip()
                    if os.path.exists(os.path.join(time_stamp_dir, 'image.png')):
                        if os.path.exists(os.path.join(time_stamp_dir, 'mask.png')):
                            aug_img_paths.append(os.path.join(time_stamp_dir, 'image.png'))
                            aug_mask_paths.append(os.path.join(time_stamp_dir, 'mask.png'))
                    elif os.path.exists(os.path.join(time_stamp_dir, 'image2.png')):
                        if os.path.exists(os.path.join(time_stamp_dir, 'mask.png')):
                            aug_img_paths.append(os.path.join(time_stamp_dir, 'image2.png'))
                            aug_mask_paths.append(os.path.join(time_stamp_dir, 'mask.png'))
            category_dict = {"prob": prob, "aug_img_paths": aug_img_paths, "aug_mask_paths": aug_mask_paths}
            self.aug_dict[i] = category_dict


    def augment(self, img_path):
        img = cv2.imread(img_path, cv2.COLOR_BGR2RGB)

        for key, value in self.aug_dict.items():
            if torch.rand(1) < value["prob"]:
                idx = torch.randint(0, len(value["aug_img_paths"]), (1,)).item()
                aug_img = cv2.imread(value["aug_img_paths"][idx], cv2.COLOR_BGR2RGB)
                aug_mask = cv2.imread(value["aug_mask_paths"][idx], cv2.COLOR_BGR2RGB)
                aug_mask[aug_mask==255] = 0
                img[aug_mask!=0] = aug_img[aug_mask!=0]
        
        return Image.fromarray(img)






class BoardDataset(Dataset):
    def __init__(self, meta_dir, transform=None, aug_meta_dirs=None): 
        self.transform = transform
        self.image_paths = []
        self.labels = []

        self.use_aug = True if aug_meta_dirs is not None else False
        if self.use_aug:
            self.aug = Augment(aug_meta_dirs)


        content = None
        with open(meta_dir, 'r') as file:
            content = json.load(file)

        # for each image
        for item in content:
            # get gt
            gt = None
            if "GO" in item['imagePath'] and "9x9" in item['imagePath']:
                gt = 1
            elif "GO" in item['imagePath'] and "13x13" in item['imagePath']:
                gt = 2
            elif "ChineseChess" in item['imagePath']:
                gt = 3
            elif "WithoutChessboard" in item['imagePath']:
                gt = 0
            else:
                gt = -1

            if gt != -1:
                # save image path
                self.image_paths.append(item['imagePath'])
                # save gt
                self.labels.append(gt)


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        gt = self.labels[idx]

        if self.use_aug:
            image = self.aug.augment(image_path)
        else:
            image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(gt, dtype=torch.long), image_path
