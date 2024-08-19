import cv2
import os

input_path = "/mnt/afs/huwensong/workspace/R4_board_classification/result/result_0813_3/best_validation_results.txt"
output_dir = '/mnt/afs/huwensong/workspace/R4_board_classification/result/result_0813_3/vis/'

image_paths = []
GTs = []
preds = []

# Open the file and read it line by line
with open(input_path, 'r') as file:
    for line in file:
        # Split the line by commas
        parts = line.split(',')
        # Find the part that starts with 'ImagePath'
        for part in parts:
            if 'ImagePath:' in part:
                # Extract the path and remove any leading/trailing spaces
                image_path = part.split('ImagePath: ')[1].strip()
                image_paths.append(image_path)
            if 'GT' in part:
                gt = part.split('GT: ')[1].strip()
                GTs.append(gt)
            if 'Pred' in part:
                pred = part.split('Pred: ')[1].strip()
                preds.append(pred)

os.makedirs(output_dir, exist_ok=True)
for image_path, gt, pred in zip(image_paths, GTs, preds):
    img = cv2.imread(image_path)
    time_stamp = image_path.split('/')[-2]
    cv2.imwrite(f"{output_dir}/{time_stamp}_{gt}_{pred}.jpg", img)