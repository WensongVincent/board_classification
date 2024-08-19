import os
import cv2 as cv
import numpy as np
import albumentations as A

from functools import partial, reduce

LABEL2COLOR = {'bkgd': 0, 'robot': 60, 'sundry': 100, 'hand': 160, 'pieces': 200, 'board': 255}



def build_transform(x):
    k, v = list(x.items())[0]
    transform = getattr(A, k, None)
    if transform is None:
        transform = getattr(A.augmentations.transforms, k, None)
    if transform is not None:
        return transform(**v)
    else:
        raise NotImplementedError



class ContexAug:
    def __init__(self, config):
        def load_txt(path): 
            with open(path) as f:
                lines = [line.strip() for line in f]
            return lines


        self.do_nothing = config is None
        self.label2color = LABEL2COLOR

        if not self.do_nothing:
            self.config = config

            for k, v in config.items():
                if v['meta'] is not None:
                    v['meta'] = reduce(lambda x, y: x+y, map(load_txt, v['meta']))



    def __call__(self, im, gt):
        if self.do_nothing:
            return im, gt

        else:
            for k, params in self.config.items():
                if np.random.uniform(0, 1) < params.p:

                    if k == 'shadow':
                        im, gt = shadow_aug_(im, gt, self.label2color)

                    elif k == 'background':
                        path = np.random.choice(params.meta)

                        if os.path.exists(os.path.join(path, 'image.png')):
                            mat_im = cv.imread(os.path.join(path, 'image.png'))
                        else:
                            mat_im = cv.imread(os.path.join(path, 'image2.png'))

                        mat_im = cv.cvtColor(mat_im, cv.COLOR_BGR2RGB)
                        
                        if os.path.exists(os.path.join(path, 'board.png')):
                            mat_mk = cv.imread(os.path.join(path, 'board.png' ))[:,:,0] 
                        else:
                            mat_mk = np.zeros(im.shape[:2], dtype=np.uint8)

                        mask = (gt == 0) | (gt == 255) 
                        gt[mask] = mat_mk[mask]
                        im[mask, :] = mat_im[mask, :]

                    else:
                        path = np.random.choice(params.meta)
                        mat_im = cv.imread(os.path.join(path, 'image.png'))
                        mat_im = cv.cvtColor(mat_im, cv.COLOR_BGR2RGB)
                        mat_mk = cv.imread(os.path.join(path, 'mask_c.png' if k == 'pieces' else 'mask.png' ))[:,:,0]
                        mask = mat_mk == LABEL2COLOR[k]

                        gt[mask] = LABEL2COLOR[k]
                        im[mask, :] = mat_im[mask, :]

            return im, gt



# 给遮挡物增加阴影 20230313
def fillHoles(im_th):
    im_floodfill = im_th.copy()
    h, w = im_th.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv.floodFill(im_floodfill, mask, (0,0), 255)
    im_floodfill_inv = cv.bitwise_not(im_floodfill)
    im_out = im_th | im_floodfill_inv
    return im_out



def generate_shadow_mask(mk, mode, target):
    anno = {'board': 255, 'hand': 160, 'robot':60, 'sundry': 100}
    if np.sum(mk == anno[target]) == 0:
        return np.zeros_like(mk)
    
    else:
        temps = []
        temp = mk.copy()
        temp[temp != anno[target]] = 0
        contours, _ = cv.findContours(temp,
                                      mode=cv.RETR_LIST,
                                      method=cv.CHAIN_APPROX_SIMPLE)
        
        
        theta = np.random.uniform(-np.pi/12, np.pi/12)
        scale = np.random.uniform(0.9, 1.2)
        sheer = np.random.uniform(0.8, 1.2)
        dx = np.random.uniform(75, 120)
        dy = np.random.uniform(75, 120)
        rotation_m = np.array([[np.cos(theta), np.sin(-theta)], 
                               [np.sin(theta), np.cos(theta)]])
        scale_m = np.array([[scale, 0], [0, scale]])
        for pts in contours:
            mask = np.zeros_like(mk)
            pts = pts.reshape(-1, 2)
            pts_copy = pts.copy()
            ori_idx = np.random.randint(len(pts))
            ori_x, ori_y = pts_copy[ori_idx]
            pts_copy[:, 0] -= ori_x
            pts_copy[:, 1] -= ori_y
            pts_copy = (scale_m @ rotation_m @ pts_copy.transpose()).transpose()
            pts_copy[:,0] += ori_x + dx
            pts_copy[:,1] += ori_y + dy
            pts_copy = pts_copy.astype(np.int32)
            
            if mode == 'low':
                mask = cv.fillPoly(mask, [pts, pts_copy], 1)
                for i in range(len(pts)):
                    # mask = cv.line(mask, pts[i], pts_copy[i] ,1)
                    mask = cv.line(mask, tuple(pts[i]), tuple(pts_copy[i]) ,1)

                mask = fillHoles(mask)
                mask = cv.fillPoly(mask, [pts], 0)
                temps.append(mask)
            else:
                mask = cv.fillPoly(mask, [pts_copy], 1)
                mask = cv.fillPoly(mask, [pts], 0)
                temps.append(mask)

        return np.clip(np.sum(temps, axis=0), 0, 1)



def add_shadow(im, mask):
    for i in range(3):
        im[:,:,i][mask>0] = im[:,:,i][mask>0] * np.random.uniform(0.5, 0.7)
    return im



def shadow_aug_(im, mk, LABEL2COLOR):
    
    mode = np.random.choice(['low', 'high'])
    mask_sundry_shadow = generate_shadow_mask(mk, 'low', 'sundry')
    mask_robot_shadow = generate_shadow_mask(mk, 'high', 'robot')
    mask_hand_shadow = generate_shadow_mask(mk, mode, 'hand')
    
    if mode == 'low':
        im = add_shadow(im, mask_sundry_shadow)
        im = add_shadow(im, mask_hand_shadow)
        im = add_shadow(im, mask_robot_shadow)
    else:
        im = add_shadow(im, mask_sundry_shadow)
        im = add_shadow(im, mask_robot_shadow)
        im = add_shadow(im, mask_hand_shadow)

    return im, mk

