im_segament_02_1.py    分割代码

labelme2mask.py    将json文件转成mask （桑童用到）


启宇：
Unet 预处理
（1）执行labelme_json_to_png.py 将json转为png mask （比较慢）
（2）执行spilit_labelme_dataset.py 从文件夹提取 png 图片
（3）执行png2label.py将png图片转为 训练用的标签


桑童也可以用启宇的这个方法，这个方法更加精确，但是需要调整png2label.py的颜色

