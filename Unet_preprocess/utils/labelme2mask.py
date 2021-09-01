import cv2
import numpy as np
import json
import skimage.io as io
import os

category_types = ["Background", "via", "metal"]

path = r"C:\Users\wei_q\Desktop\Unet_preprocess/IRdata/image/"
json_path = r"C:\Users\wei_q\Desktop\Unet_preprocess/IRdata/json/"
new_path = r"C:\Users\wei_q\Desktop\Unet_preprocess/IRdata/mask3/"

# label_save_path = "../IRdata/png"
# image_save_path = "../IRdata/trainimg"
files = os.listdir(path)

for filename in files:

    img = cv2.imread(path+filename)
    h, w = img.shape[:2]
    mask = np.zeros([h, w, 1], np.uint8)    # 创建一个大小和原图相同的空白图像
    # print(img)
    json_path1 = json_path+ os.path.splitext(filename)[0]+ ".json"
    with open(json_path1, "r") as f:
        label = json.load(f)

    shapes = label["shapes"]
    for shape in shapes:
        category = shape["label"]
        points = shape["points"]
        # 填充
        points_array = np.array(points, dtype=np.int32)
        if category == "Background":
            mask = cv2.fillPoly(mask, [points_array], 0)
        elif category == "via":
            mask = cv2.fillPoly(mask, [points_array], 255)
        elif category == "metal":
            mask = cv2.fillPoly(mask, [points_array], 128)
        # mask = cv2.fillPoly(mask, [points_array], category_types.index(category))

    io.imsave(new_path+ os.path.splitext(filename)[0]+ ".png",mask)
