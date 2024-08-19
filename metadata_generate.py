from pathlib import Path, PosixPath
import json
from tqdm import tqdm

## 0808_1
# train_dir = ['/mnt/afs/share_data/R3/v0.3/data/media/raw_data/5.Segmentation/20240628_R1_WithoutChessboard/20221122_25_R1_BGDataWithoutChessboard_User_Scenario',
#              '/mnt/afs/share_data/R3/v0.3/data/media/raw_data/5.Segmentation/20240628_R1_WithoutChessboard/20221122_R1_BGDataWithoutChessboard_User_Scenario',
#              '/mnt/afs/share_data/R3/v0.3/data/media/raw_data/5.Segmentation/20240628_R1_WithoutChessboard/20221122_R1_BGDataWithoutChessboard_User_Scenario',
#              '/mnt/afs/share_data/R3/v0.3/data/media/raw_data/5.Segmentation/20240628_R1_WithoutChessboard/20220812_R1_BGDataWithoutChessboard_FactoryScene',
#              '/mnt/afs/share_data/R4/v0.3/data/media/raw_data/5.Segmentation/20240710_R4v0.3.9.GO_5.Segmentation_TestData_13x13',
#              '/mnt/afs/share_data/R4/v0.3/data/media/raw_data/5.Segmentation/20240710_R4v0.3.9.GO_5.Segmentation_TestData_9x9',
#              '/mnt/afs/share_data/R4/v0.3/data/media/raw_data/5.Segmentation/20240725-26_R4-GO_5.Segmentation_TestData_13x13',
#              '/mnt/afs/share_data/R4/v0.3/data/media/raw_data/5.Segmentation/20240725-26_R4-GO_5.Segmentation_TestData_9x9',
#              '/mnt/afs/share_data/R4/v0.3/data/media/raw_data/5.Segmentation/20240730_R4v0.3.6.ChineseChess_5.Segmentation_TestData']
# test_dir = ['/mnt/afs/share_data/R4/v0.3/data/media/raw_data/3.Detection/20240731_R4v0.3.6.ChineseChess_3.Detection-Pieces_TrainData', 
#             '/mnt/afs/share_data/R4/v0.3/data/media/raw_data/3.Detection/20240702_R4.9.GO_3.Detection-Pieces_TestData_13x13_P1',
#             '/mnt/afs/share_data/R4/v0.3/data/media/raw_data/3.Detection/20240702_R4.9.GO_3.Detection-Pieces_TestData_9x9_P1',
#             '/mnt/afs/share_data/R4/v0.3/data/media/raw_data/3.Detection/20240708_R4v0.3.9.GO_3.Detection-Pieces_TestData_13x13_P2P3P4',
#             '/mnt/afs/share_data/R4/v0.3/data/media/raw_data/3.Detection/20240708_R4v0.3.9.GO_3.Detection-Pieces_TestData_9x9_P2P3P4',
#             '/mnt/afs/share_data/R3/v0.3/data/media/raw_data/5.Segmentation/20240628_R1_WithoutChessboard/20221206_R1_BGDataWithoutChessboard_User_Scenario']

## 0813_1
# train_dir = ['/mnt/afs/share_data/R4/v0.3/data/media/raw_data/3.Detection/20240731_R4v0.3.6.ChineseChess_3.Detection-Pieces_TrainData', 
#             '/mnt/afs/share_data/R4/v0.3/data/media/raw_data/3.Detection/20240702_R4.9.GO_3.Detection-Pieces_TestData_13x13_P1',
#             '/mnt/afs/share_data/R4/v0.3/data/media/raw_data/3.Detection/20240702_R4.9.GO_3.Detection-Pieces_TestData_9x9_P1',
#             '/mnt/afs/share_data/R4/v0.3/data/media/raw_data/3.Detection/20240708_R4v0.3.9.GO_3.Detection-Pieces_TestData_13x13_P2P3P4',
#             '/mnt/afs/share_data/R4/v0.3/data/media/raw_data/3.Detection/20240708_R4v0.3.9.GO_3.Detection-Pieces_TestData_9x9_P2P3P4',
#             '/mnt/afs/share_data/R3/v0.3/data/media/raw_data/5.Segmentation/20240628_R1_WithoutChessboard/20221206_R1_BGDataWithoutChessboard_User_Scenario']
# test_dir = ['/mnt/afs/share_data/R3/v0.3/data/media/raw_data/5.Segmentation/20240628_R1_WithoutChessboard/20221122_25_R1_BGDataWithoutChessboard_User_Scenario',
#              '/mnt/afs/share_data/R3/v0.3/data/media/raw_data/5.Segmentation/20240628_R1_WithoutChessboard/20221122_R1_BGDataWithoutChessboard_User_Scenario',
#              '/mnt/afs/share_data/R3/v0.3/data/media/raw_data/5.Segmentation/20240628_R1_WithoutChessboard/20221122_R1_BGDataWithoutChessboard_User_Scenario',
#              '/mnt/afs/share_data/R3/v0.3/data/media/raw_data/5.Segmentation/20240628_R1_WithoutChessboard/20220812_R1_BGDataWithoutChessboard_FactoryScene',
#              '/mnt/afs/share_data/R4/v0.3/data/media/raw_data/5.Segmentation/20240710_R4v0.3.9.GO_5.Segmentation_TestData_13x13',
#              '/mnt/afs/share_data/R4/v0.3/data/media/raw_data/5.Segmentation/20240710_R4v0.3.9.GO_5.Segmentation_TestData_9x9',
#              '/mnt/afs/share_data/R4/v0.3/data/media/raw_data/5.Segmentation/20240725-26_R4-GO_5.Segmentation_TestData_13x13',
#              '/mnt/afs/share_data/R4/v0.3/data/media/raw_data/5.Segmentation/20240725-26_R4-GO_5.Segmentation_TestData_9x9',
#              '/mnt/afs/share_data/R4/v0.3/data/media/raw_data/5.Segmentation/20240730_R4v0.3.6.ChineseChess_5.Segmentation_TestData']

# 0813_2
train_dir = ['/mnt/afs/share_data/R4/v0.3/data/media/raw_data/3.Detection/20240731_R4v0.3.6.ChineseChess_3.Detection-Pieces_TrainData', 
            '/mnt/afs/share_data/R4/v0.3/data/media/raw_data/3.Detection/20240702_R4.9.GO_3.Detection-Pieces_TestData_13x13_P1',
            '/mnt/afs/share_data/R4/v0.3/data/media/raw_data/3.Detection/20240702_R4.9.GO_3.Detection-Pieces_TestData_9x9_P1',
            '/mnt/afs/share_data/R4/v0.3/data/media/raw_data/3.Detection/20240708_R4v0.3.9.GO_3.Detection-Pieces_TestData_13x13_P2P3P4',
            '/mnt/afs/share_data/R4/v0.3/data/media/raw_data/3.Detection/20240708_R4v0.3.9.GO_3.Detection-Pieces_TestData_9x9_P2P3P4',
            '/mnt/afs/share_data/R4/v0.3/data/media/raw_data/5.Segmentation/20240710_R4v0.3.9.GO_5.Segmentation_TestData_13x13',
            '/mnt/afs/share_data/R4/v0.3/data/media/raw_data/5.Segmentation/20240710_R4v0.3.9.GO_5.Segmentation_TestData_9x9',
            '/mnt/afs/share_data/R3/v0.3/data/media/raw_data/5.Segmentation/20240628_R1_WithoutChessboard/20221122_R1_BGDataWithoutChessboard_User_Scenario',
            '/mnt/afs/share_data/R3/v0.3/data/media/raw_data/5.Segmentation/20240628_R1_WithoutChessboard/20221206_R1_BGDataWithoutChessboard_User_Scenario']
test_dir = ['/mnt/afs/share_data/R3/v0.3/data/media/raw_data/5.Segmentation/20240628_R1_WithoutChessboard/20221122_R1_BGDataWithoutChessboard_User_Scenario',
            '/mnt/afs/share_data/R3/v0.3/data/media/raw_data/5.Segmentation/20240628_R1_WithoutChessboard/20221122_25_R1_BGDataWithoutChessboard_User_Scenario',
            '/mnt/afs/share_data/R3/v0.3/data/media/raw_data/5.Segmentation/20240628_R1_WithoutChessboard/20220812_R1_BGDataWithoutChessboard_FactoryScene',
            '/mnt/afs/share_data/R4/v0.3/data/media/raw_data/5.Segmentation/20240725-26_R4-GO_5.Segmentation_TestData_13x13',
            '/mnt/afs/share_data/R4/v0.3/data/media/raw_data/5.Segmentation/20240725-26_R4-GO_5.Segmentation_TestData_9x9',
            '/mnt/afs/share_data/R4/v0.3/data/media/raw_data/5.Segmentation/20240730_R4v0.3.6.ChineseChess_5.Segmentation_TestData']


# ===== change version for new metadata =====
metadata_save_version = '0813_2' 
# ===== =============================== =====
metadata_save_dir = '/mnt/afs/huwensong/workspace/R4_board_classification/metadata'
metadata_save_name_train = f'metadata_{metadata_save_version}_train.json'
metadata_save_name_test = f'metadata_{metadata_save_version}_test.json'


# ===== train =====
train_meta_content = []
statistics = {'go9x9': 0,
              'go13x13': 0,
              'chnchess': 0,
              'none': 0}
for dir in tqdm(train_dir):
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
                train_meta_content.append({'imagePath': str(image_path), 'gt': gt})
with open(f'{metadata_save_dir}/{metadata_save_name_train}', 'w') as f:
    json.dump(train_meta_content, f, ensure_ascii=False, indent=4)

print('train stat: ', statistics)



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
                test_meta_content.append({'imagePath': str(image_path), 'gt': gt})
with open(f'{metadata_save_dir}/{metadata_save_name_test}', 'w') as f:
    json.dump(test_meta_content, f, ensure_ascii=False, indent=4)

print('test stat: ', statistics)