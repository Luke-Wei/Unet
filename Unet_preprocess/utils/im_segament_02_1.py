import os
import numpy as np
import cv2
import shutil

#剪裁
def crop_one_picture(path, filename, cols, rows,save_path1, geshi ):
    img = cv2.imread(path + filename,
                     -1)  ##读取彩色图像，图像的透明度(alpha通道)被忽略，默认参数;灰度图像;读取原始图像，包括alpha通道;可以用1，0，-1来表示
    sum_rows = img.shape[0]  # 高度
    sum_cols = img.shape[1]  # 宽度
    name = os.path.splitext(file)[0]  #name取文件名的除去后缀的部分
    save_path = save_path1
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print("裁剪所得{0}列图片，{1}行图片.".format(int(sum_cols / cols), int(sum_rows / rows)))

    for i in range(int(sum_cols / cols)):
        for j in range(int(sum_rows / rows)):
            cv2.imwrite(save_path + os.path.splitext(filename)[0] + '_' + str(j) + '_' + str(i) + geshi,
                        #os.path.splitext(filename)[1],
                        img[j * rows:(j + 1) * rows, i * cols:(i + 1) * cols])
            # print(path+"\crop\\"+os.path.splitext(filename)[0]+'_'+str(j)+'_'+str(i)+os.path.splitext(filename)[1])
    print("裁剪完成，得到{0}张图片.".format(int(sum_cols / cols) * int(sum_rows / rows)))
    print("文件保存在{0}".format(save_path))

#合并
"""遍历文件夹下某格式图片"""
def file_name(root_path,picturetype):
    filename=[]
    for root,dirs,files in os.walk(root_path):
        for file in files:
            if os.path.splitext(file)[1]==picturetype:
                filename.append(os.path.join(root,file))
    return filename

def merge_picture(merge_path,num_of_cols,num_of_rows):
    filename=file_name(merge_path,".jpg")
    shape=cv2.imread(merge_path,-1).shape    #三通道的影像需把-1改成1
    cols=shape[1]
    rows=shape[0]
    #channels=shape[2]
    dst=np.zeros((rows*num_of_rows,cols*num_of_cols),np.uint8)
    for i in range(len(filename)):
        img=cv2.imread(merge_path,-1)
        cols_th=int(filename[i].split("_")[-1].split('.')[0])
        rows_th=int(filename[i].split("_")[-2])
        roi=img[0:rows,0:cols]
        dst[rows_th*rows:(rows_th+1)*rows,cols_th*cols:(cols_th+1)*cols]=roi
    #save_path = "D:\image_sega\im_se2\\tif\merge" + name
    cv2.imwrite(merge_path + "merge.tif",dst)
    move_path = "D:\image_sega\im_se2\\tif\merge\\" + name + "merge.tif"
    shutil.move(merge_path + "merge.tif", move_path)  #将还原的图片加入merge文件夹中，具体路径需要根据情况改变


#
# 调用剪裁
##原图
# path='../IRdata/trainimg/'    #要裁剪的图片所在的文件夹
# save_path = '../IRdata/croped/cropedjpg/'

##png

path='/home/wqy/Unet_preprocess/IRdata/ready2seg/'    #要裁剪的图片所在的文件夹
save_path = '/home/wqy/Unet_preprocess/IRdata/croped/cropedpng/'


#mask
# path='../IRdata/mask/'    #要裁剪的图片所在的文件夹
# save_path = '../IRdata/croped/cropedmask/'

files = os.listdir(path)
for file in files:  #依次读取文件夹下所有文件
    if file.endswith('.tif'):     #判断后缀为.tif的文件
        filename = file
        crop_one_picture(path, filename, 512, 512,save_path,'.jpg')
    elif   file.endswith('.png'):     #判断后缀为.tif的文件
        filename = file
        crop_one_picture(path, filename, 512, 512,save_path,'.png')

# #调用合并
# path='../IRdata/croped/cropedjpg/'  #要裁剪的图片所在的文件夹
# files = os.listdir(path)
# for file in files:  #依次读取文件夹下所有文件
#     if file.endswith('.jpg'):     #判断后缀为.tif的文件
#         filename = file
#         # name = os.path.splitext(file)[0]
#         merge_path= path + filename                  #"D:\\image_sega\\im_se2\\tif\\" + name +"\\crop512_512\\"  #要合并的小图片512*512所在的文件夹
#         num_of_cols=16    #列数
#         num_of_rows=16  #行数
#         merge_picture(merge_path,num_of_cols,num_of_rows)
#         print(merge_path)
