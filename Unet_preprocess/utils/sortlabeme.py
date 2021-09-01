# -*- coding:utf-8 -*-

'''
仿照labelme的json文件写入自己的数据
'''
import cv2
import json
import base64
import contextlib
import io
import json
import os.path as osp
import PIL.Image
import imgviz
# import label as Lb
from labelme.logger import logger
from labelme import PY2
from labelme import QT4
from labelme import utils
#import  Unet.shape as US
import  os

def dict_json(imageData,shapes,imagePath,version=None,flags=None, imageHeight=None,imageWidth=None):
    '''

    :param imageData: str
    :param shapes: list
    :param imagePath: str
    :param fillColor: list
    :param lineColor: list
    :return: dict
    '''
    return {"version":version,"flags":flags,"shapes":shapes,'imagePath':imagePath,"imageData":imageData,'imageHeight':imageHeight,'imageWidth':imageWidth}


def dict_shapes(points,label,group_id=None,shape_type=None,flags=None):
    return {'label':label,'points':points,'group_id':group_id,'shape_type':shape_type,"flags":flags}


json_path =  r'/home/wqy/Unet_preprocess/IRdata/json'
img_path =  r'/home/wqy/Unet_preprocess/IRdata/image'

img_f = os.listdir(json_path)
n = 0
for i in img_f:

    img_filename = img_path + i
    json_file = json_path +'\\'+ i.split(".")[0] +'.json'
    print(json_file)
    data = json.load(open(json_file))

    newshape = sorted(data["shapes"], key=lambda x: x["label"])
    data = dict_json(data["imageData"], newshape, data["imagePath"], data["version"], data["flags"], data["imageHeight"], data["imageWidth"])
    json_file = r'/home/wqy/Unet_preprocess/IRdata/newjson' +'\\'+  i.split(".")[0] +'.json'
    json.dump(data,open(json_file,'w'))

    # version =  "4.5.7"
    # shapes= []
    # data = json.load(open(json_file))
    # # print(imageData)
    #
    # flags  ={}
    # imagePath = "D:/ICCV/code\jiaoben/Unet/IRdata/image/" + i
    # imageHeight = 8192
    # imageWidth=8192
    #
    # label_name_to_value = {"_background_": 0, 'via':1, 'metal':2}
    # for shape in sorted(data["objects"], key=lambda x: x["type"]):
    #     flag = {}
    #     label_name = shape["type"]
    #     points= shape["point_indices"]
    #     if label_name == "via":
    #         shape_type1 = "rectangle"
    #     else:
    #         shape_type1 = "polygon"
    #     shapes.append(dict_shapes(points,label_name,shape_type=shape_type1,flags=flag))
    #
    # data=dict_json(imageData,shapes,imagePath,version,flags,imageHeight,imageWidth)
    # json_file = '../IRdata/json/' +  i.split(".")[0] +'.json'
    # json.dump(data,open(json_file,'w'))


