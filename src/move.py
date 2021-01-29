import numpy as np
import os
import cv2

path_dir = 'I:/Data/Fingerprint/Semantic_Segmentation_Dataset/label_255'
# path_dir = 'H:\\workspace_Mark\\Dataset\\Published_database_FV-USM_Dec2013\\1st_session\\extractedvein'
output_dir = 'I:/Data/Fingerprint/Semantic_Segmentation_Dataset/labels'

# dir_list = os.listdir(path_dir)
# print(dir_list)
i = 0
# for dir_name in dir_list:
#     dir_path = os.path.join(path_dir, dir_name)

file_list = os.listdir(path_dir)

for file_name in file_list:
    img_path = os.path.join(path_dir, file_name)

    img = cv2.imread(img_path, 0)

    label = img // 128

    # img = cv2.resize(img, (600, 240))
    output_path = os.path.join(output_dir, file_name)
    cv2.imwrite(output_path, label)

    # i += 1

print("finish")