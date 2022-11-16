import  tensorflow as tf
from    tensorflow import keras
from    tensorflow.keras import layers
import matplotlib.pyplot as  plt
import  numpy as np

import os

# img x jpg 文件
# os.listdir()
img = tf.io.read_file(r'D:\zhangyifei\study\Python\Tenorflow Dataset\FCN\images1\images\Abyssinian_1.jpg')
img = tf.image.decode_jpeg(img)   # 解码jpg文件 [358,500,1]
img = tf.squeeze(img)          # [358,500] 去掉维度
plt.imshow(img)
plt.show()

# trimaps 语义分割的图像 png 文件
# 目标数据 里面都是分类数据 每个像素点都是label
# os.listdir()
img_y = tf.io.read_file(r'D:\zhangyifei\study\Python\Tenorflow Dataset\FCN\annotations1\annotations\trimaps\Abyssinian_1.png')
img_y = tf.image.decode_png(img_y)   # 解码png文件 [1,358,500]
img_y = tf.squeeze(img_y)          # [358,500] 去掉维度
plt.imshow(img_y)
plt.show()

print(img_y.numpy().max())
print(img_y.numpy().min())
print(np.unique(img_y.numpy()))



