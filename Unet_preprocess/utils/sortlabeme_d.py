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
import  Unet.shape as US
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


json_path =  '../data/json/'
img_path =  '../data/img/'

img_f = os.listdir(json_path)
n = 0
for i in img_f:

    img_filename = img_path + i
    json_file = json_path + i.split(".")[0] +'.json'
    print(json_file)
    data = json.load(open(json_file))
    shapes = []
    newshape = sorted(data["shapes"], key=lambda x: x["label"])
    for shape in newshape:
        flag = shape["flags"]
        label_name = shape["label"]
        shape_type1 = shape["shape_type"]
        points= shape["points"]
        print(label_name)
        if label_name == "solderjoint":
            shapes.append(dict_shapes(points,label_name,shape_type=shape_type1,flags=flag))

    data = dict_json(data["imageData"], shapes, data["imagePath"], data["version"], data["flags"], data["imageHeight"], data["imageWidth"])
    json_file = '../data/newjson/' +  i.split(".")[0] +'.json'
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

    #
    # data=dict_json(imageData,shapes,imagePath,version,flags,imageHeight,imageWidth)
    # json_file = '../IRdata/json/' +  i.split(".")[0] +'.json'
    # json.dump(data,open(json_file,'w'))


