import os

img_path =  '../IRdata/ajson/'
# xml_path =  'F:\\数据流项目应用二数据集(列车)\\标注\\new\\data2\\VOCdevkit2007\\VOC2007\\Annotations\\'
# 'F:\\数据流项目应用二数据集(列车)\\标注\\new\\data2\\VOCdevkit2007\\VOC2007\\JPEGImages\\'
img_f = os.listdir(img_path)
# xml_f = os.listdir(xml_path)
# f = f.sorted()
n = 0
for i in img_f:
    # print(img_f[n].split(".")[0])
    oldname = img_path + img_f[n]
    newname = img_path + str(n) + '.json'
    #
    # xmlname = xml_path + xml_f[n]
    # newxmlname = xml_path + "COCO_train2014_" + str(n + 20180000001).zfill(12) + '.xml'
    #
    # # 用os模块中的rename方法对文件改名
    os.rename(oldname, newname)
    # os.rename(xmlname, newxmlname)
    # print(oldname, '======>', newname)
    # print(oldname)
    # print(xmlname, '======>', newxmlname)
    n+=1