import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import skimage
print("skimage:", skimage.__version__)    #  '0.14.2'

color2index = {
    (0,0,0) : 0,
    (0,0,128) : 255,     #红（和标准呢rgb是反的） 
    (0,128,0) : 255,   #绿
#    (0,128,128) : 180   #黄
}

#baseboard 1
#line 2
#solderjoint 3

def rgb2mask(img):

    assert len(img.shape) == 3
    height, width, ch = img.shape
    assert ch == 3

    W = np.power(256, [[0],[1],[2]])

    img_id = img.dot(W).squeeze(-1)
    values = np.unique(img_id)
    mask = np.zeros(img_id.shape)

    for i, c in enumerate(values):
        print(c)
        try:
            mask[img_id==c] = color2index[tuple(img[img_id==c][0])]
        except:
            pass
    return mask

path = "/home/wqy/Unet_preprocess/IRdata/png/"
new_path = "/home/wqy/Unet_preprocess/IRdata/mask/"

# label_save_path = "../IRdata/png"
# image_save_path = "../IRdata/trainimg"
files = os.listdir(path)

for filename in files:
    f_path = path + filename
    print(f_path)
    # img = io.imread(f_path)
    img = cv2.imread(f_path, 1)
    print(img.shape)
    mask = rgb2mask(img)
    mask = mask.astype(np.uint8)
    print(mask.shape)
    f_new_path = new_path + filename
    io.imsave(f_new_path,mask)
