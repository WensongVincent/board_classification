from pathlib import Path, PosixPath
import json
from tqdm import tqdm
import cv2
import os

# 0813_2
test_dir = ['/mnt/afs/share_data/R3/v0.3/data/media/raw_data/5.Segmentation/20240628_R1_WithoutChessboard/20221122_R1_BGDataWithoutChessboard_User_Scenario',
            '/mnt/afs/share_data/R3/v0.3/data/media/raw_data/5.Segmentation/20240628_R1_WithoutChessboard/20221122_25_R1_BGDataWithoutChessboard_User_Scenario',
            '/mnt/afs/share_data/R3/v0.3/data/media/raw_data/5.Segmentation/20240628_R1_WithoutChessboard/20220812_R1_BGDataWithoutChessboard_FactoryScene',
            '/mnt/afs/share_data/R4/v0.3/data/media/raw_data/5.Segmentation/20240725-26_R4-GO_5.Segmentation_TestData_13x13',
            '/mnt/afs/share_data/R4/v0.3/data/media/raw_data/5.Segmentation/20240725-26_R4-GO_5.Segmentation_TestData_9x9',
            '/mnt/afs/share_data/R4/v0.3/data/media/raw_data/5.Segmentation/20240730_R4v0.3.6.ChineseChess_5.Segmentation_TestData']


# ===== change version for new metadata =====
metadata_save_version = '0813_2' 
# ===== =============================== =====
metadata_save_dir = f'/mnt/afs/huwensong/workspace/R4_board_classification/data/{metadata_save_version}'
os.makedirs(metadata_save_dir, exist_ok=True)
metadata_save_name_test = f'metadata_{metadata_save_version}_test.json'


# ===== test =====
test_meta_content = []
statistics = {'go9x9': 0,
              'go13x13': 0,
              'chnchess': 0,
              'none': 0}
for dir in tqdm(test_dir):
    dir = Path(dir)
    image_paths = dir.rglob(r'*.png')
    for image_path in image_paths:
        if "calib" not in str(image_path) and "right" not in str(image_path): # don't save calib data and right camera data
            gt = None
            if "GO" in str(image_path) and "9x9" in str(image_path):
                gt = 1
                statistics['go9x9'] += 1
            elif "GO" in str(image_path) and "13x13" in str(image_path):
                gt = 2
                statistics['go13x13'] += 1
            elif "ChineseChess" in str(image_path):
                gt = 3
                statistics['chnchess'] += 1
            elif "WithoutChessboard" in str(image_path):
                gt = 0
                statistics['none'] += 1
            else:
                gt = -1
                print(f'Error for image {str(image_path)}')
            
            if gt != -1:
                time_stamp = image_path.parts[-2]
                image = cv2.imread(str(image_path))
                cv2.imwrite(f'{metadata_save_dir}/{time_stamp}_{gt}.png', image)

print('test stat: ', statistics)