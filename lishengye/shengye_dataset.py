import os
import sys
import json
import cv2 as cv
import numpy as np

import albumentations as A

import torch
from torch.utils.data import Dataset

from functools import partial, reduce

from transforms import *



COLOR2LOGIT = {'V1': {0: 0, 60: 2, 100: 0, 160: 0, 200: 1, 255: 1}, 
               'V2': {0: 0, 60: 2, 100: 0, 160: 0, 200: 3, 255: 1},}

LABEL2COLOR = {'bkgd': 0, 'robot': 60, 'sundry': 100, 'hand': 160, 'chess': 200, 'board': 255}



def load_txt(path): 
    with open(path) as f:
        lines = [line.strip() for line in f]
    return lines



def clean_gt(x):
    x[(x!=0) & (x!=60) & (x!=100) & (x!=160) & (x!=200) & (x!=255)] = 0
    return x



def color2logit(lookup, gt):
    for key in lookup:
        gt[gt==key] = lookup[key]
    return gt

                        


class SegDataset(Dataset):
    def __init__(self, config):
        super(SegDataset, self).__init__()

        self.meta = reduce(lambda x, y: x+y, map(load_txt, config.meta))
        self.ctx_aug = ContexAug(config.ctxt_aug)
        self.albu_aug = A.Compose(list(map(build_transform, config.albu_aug)))

        ignore_chess = config.anno in ['V1']
        self.mask_suffix = 'mask.png' if ignore_chess else 'mask_c.png'
        self.color2logit = partial(color2logit, lookup=COLOR2LOGIT[config.anno])
        


    def __getitem__(self, idx):
        
        try:
            im = cv.cvtColor(cv.imread(os.path.join(self.meta[idx], 'image.png')), cv.COLOR_BGR2RGB)
            gt = cv.imread(os.path.join(self.meta[idx], self.mask_suffix))[:,:,0]
        except:
            return self[idx+1]

        im, gt = self.ctx_aug(im, gt)

        transformed = self.albu_aug(image=im, mask=gt)
        im, gt = transformed['image'], transformed['mask']
        gt = self.color2logit(gt=gt)

        im = torch.from_numpy(im).permute(2, 0, 1).float()
        gt = torch.from_numpy(gt).long()

        return self.meta[idx], im, gt

    

    def __len__(self):
        return len(self.meta)



if __name__ == '__main__':
    import yaml
    from easydict import EasyDict

    with open('/mnt/afs/lishengye/code/stdc_R3/config/train.yaml') as f:
        config = EasyDict(yaml.load(stream=f, Loader=yaml.FullLoader))

    dataset = SegDataset(config.data.trainset)

    print(len(dataset))

    for i in range(0, 10):
        path, im, gt = dataset[i]

        gt[gt==1] = 255
        gt[gt==2] = 60
        gt[gt==3] = 200

        im = im.permute(1, 2, 0).numpy().astype(np.uint8)
        gt = gt.numpy().astype(np.uint8)

        cv.imwrite(f'/mnt/afs/lishengye/code/stdc_R3/exps/debug/{i}_image.png', cv.cvtColor(im, cv.COLOR_RGB2BGR))
        cv.imwrite(f'/mnt/afs/lishengye/code/stdc_R3/exps/debug/{i}_mask.png', gt)
