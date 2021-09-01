import os
import numpy as np
import json
import shutil


def find_dir_path(path, keyword_name, dir_list):
    files = os.listdir(path)
    for file_name in files:
        file_path = path+ '/' + file_name  #os.path.join(path, file_name)
        if os.path.isdir(file_path) and keyword_name not in file_path:
            find_dir_path(file_path, keyword_name, dir_list)
        elif os.path.isdir(file_path) and keyword_name in file_path:
            dir_list.append(file_path)


all_result_path = []
src_path = "/home/wqy/Unet_preprocess/IRdata/json"                  #r'C:\Users\eadhaw\Desktop\0120test'
label_save_path = "/home/wqy/Unet_preprocess/IRdata/png"
image_save_path = "/home/wqy/Unet_preprocess/IRdata/trainimg"
find_dir_path(src_path, '_json', all_result_path)  # 找出所有带着关键词(_json)的所有目标文件夹
# print(all_result_path)


for dir_path in all_result_path:
    # print(dir_path)
    file_name = dir_path.split('/')[-1]
    key_word = file_name[:-5]
    # print(key_word)
    label_file = dir_path + "/" + "label.png"
    new_label_save_path = label_save_path + "/" + key_word + ".png"  # 复制生成的label.png到新的文件夹segmentations
    # print(new_label_save_path)
    shutil.copyfile(label_file, new_label_save_path)

    img_dir = os.path.dirname(dir_path)  # 复制原图到新的文件夹images
    img_file =  "../IRdata/image/" + key_word + ".tif"
    new_img_save_path = image_save_path + "/" + key_word + ".tif"
    shutil.copyfile(img_file, new_img_save_path)