### Unet

##Unet_preprocess

这个文件夹下存放的是预处理程序
（1）执行labelme_json_to_png.py 将json转为png mask
（2）执行spilit_labelme_dataset.py 从文件夹提取 png 图片
（3）执行png2label.py将png图片转为 训练用的标4签裁剪 
（4）执行im_segament_02_1.py 裁剪 裁剪后文件在/IRdata/croped/cropedpng/


##unet-nested-multiple-classification-master

这个文件夹下存放的是Unet模型

#train

python train.py

#inference

python inference.py -m ./data/checkpoints/epoch_80.pth -i ./data/test/input -o ./data/test/output



