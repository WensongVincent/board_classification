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
import albumentations as A
from albumentations.pytorch import ToTensorV2


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
        
        # return Image.fromarray(img)
        return img


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

        if self.use_aug: # and "Segmentation" not in image_path:
            image = self.aug.augment(image_path)
        else:
            # image = Image.open(image_path).convert("RGB")
            image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)

        # if self.transform[0] == 'RandomResizeCrop':
        #     image = self.random_resized_crop(image, (224, 224))
        # elif self.transform[0] == 'RandomResize':
        #     image = self.random_resize(image, (224, 224))
        # elif self.transform[0] == 'Resize':
        #     image = self.resize(image, (224, 224))
        # else:
        #     raise Exception("Unknown resize type")

        # image = Image.fromarray(image)

        if self.transform:
            # image = self.transform(Image.fromarray(image))
            # image = self.transform[1](image)
            image = self.transform(image=image)['image']
            # image_torch = self.transform[0]['train'](Image.fromarray(image))
            # image_album = self.transform[1]['train'](image=image)['image']
            # image_torch = self.transform[0]['val'](Image.fromarray(image))
            # image_album = self.transform[1]['val'](image=image)['image']

        return image, torch.tensor(gt, dtype=torch.long), image_path
        # return image_torch, image_album



    def random_resized_crop(self, image, output_size, scale=(0.08, 1.0), ratio=(3./4., 4./3.)):
        height, width, _ = image.shape
        area = height * width

        # List of possible interpolation methods
        interpolation_methods = [
            cv2.INTER_LINEAR,
            cv2.INTER_NEAREST,
            cv2.INTER_AREA,
            cv2.INTER_CUBIC,
            cv2.INTER_LANCZOS4
        ]

        for attempt in range(10):
            target_area = random.uniform(*scale) * area
            aspect_ratio = random.uniform(*ratio)

            w = int(round((target_area * aspect_ratio) ** 0.5))
            h = int(round((target_area / aspect_ratio) ** 0.5))

            if w <= width and h <= height:
                top = random.randint(0, height - h)
                left = random.randint(0, width - w)

                crop = image[top: top + h, left: left + w]

                # Randomly select an interpolation method
                interpolation = random.choice(interpolation_methods)

                # Resize the crop to the desired output size using the selected interpolation
                resized_crop = cv2.resize(crop, (output_size[1], output_size[0]), interpolation=interpolation)
                return resized_crop

        # Fallback: If the loop fails, do central crop and resize
        in_ratio = width / height
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image fits in the aspect ratio
            w = width
            h = height

        top = (height - h) // 2
        left = (width - w) // 2
        crop = image[top: top + h, left: left + w]

        # Randomly select an interpolation method for fallback
        interpolation = random.choice(interpolation_methods)

        resized_crop = cv2.resize(crop, (output_size[1], output_size[0]), interpolation=interpolation)
        return resized_crop

    def random_resize(self, image, output_size):
        # List of possible interpolation methods
        interpolation_methods = [
            cv2.INTER_LINEAR,
            cv2.INTER_NEAREST,
            cv2.INTER_AREA,
            cv2.INTER_CUBIC,
            cv2.INTER_LANCZOS4
        ]

        # Randomly select an interpolation method
        interpolation = random.choice(interpolation_methods)

        # Resize the image to the desired output size using the selected interpolation
        resized_image = cv2.resize(image, (output_size[1], output_size[0]), interpolation=interpolation)

        return resized_image
    
    def resize(self, image, output_size):
        # Default interpolation method
        interpolation = cv2.INTER_LINEAR

        # Resize the image to the desired output size using the default interpolation
        resized_image = cv2.resize(image, (output_size[1], output_size[0]), interpolation=interpolation)

        return resized_image
    


def main():
    data_transforms_torch = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    data_transforms_album = {
        'train': A.Compose([
            A.RandomResizedCrop(height=224, width=224),
            A.HorizontalFlip(p=0.5),
            # A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()  # Converts NumPy array to PyTorch tensor
        ]),
        'val': A.Compose([
            A.Resize(height=224, width=int(1920 * (224/1080))),
            A.CenterCrop(height=224, width=224),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()  # Converts NumPy array to PyTorch tensor
        ])
    }

    dataset = BoardDataset(meta_dir="/mnt/afs/huwensong/workspace/R4_board_classification/metadata/metadata_0813_2_train.json",
                           transform=[data_transforms_torch, data_transforms_album],
                           aug_meta_dirs=[[0.2, "/mnt/afs/lishengye/code/stdc_general/meta/R2_train_sundry.txt"],
                          [0.2, "/mnt/afs/lishengye/code/stdc_general/meta/R2_train_hand.txt"],
                          [0.2, "/mnt/afs/lishengye/code/stdc_general/meta/R3_V010_train_robot_clean.txt"]])

    for i in range(len(dataset)):
        img_torch, img_album = dataset[i]
        to_pil = transforms.ToPILImage()
        img_torch = to_pil(img_torch)
        img_album = to_pil(img_album)

        img_torch.save(f'example_torch_{i}_val.png')
        img_album.save(f'example_album_{i}_val.png')

if __name__ =='__main__':
    main()