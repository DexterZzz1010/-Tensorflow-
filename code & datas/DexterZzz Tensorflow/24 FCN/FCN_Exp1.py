import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as  plt
import numpy as np
import glob
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.random.set_seed(2345)

np.random.seed(2345)


def read_jpg(path):
    # JPG 读取+解码
    img=tf.io.read_file(path)
    img=tf.image.decode_jpeg(img)
    return  img

def read_png(path):
    # PNG 读取+解码
    img=tf.io.read_file(path)
    img=tf.image.decode_png(img)
    return  img


# 分成三类

# 读取图像 glob
images = glob.glob(r'D:\zhangyifei\study\Python\Tenorflow Dataset\FCN\images1\images\*.jpg')
anno = glob.glob(r'D:\zhangyifei\study\Python\Tenorflow Dataset\FCN\annotations1\annotations\trimaps\*.png')
# print(images[-5::])

# 构建数据集和测试集
# 乱序
index = np.random.permutation(len(images))
images = np.array(images)[index]
anno = np.array(anno)[index]

dataset = tf.data.Dataset.from_tensor_slices((images, anno))
test_count = int(len(images) * 0.2)  # 五分之一作为测试集
train_count = len(images) - test_count
# print(test_count, train_count)

data_train = dataset.skip(test_count)
data_test = dataset.take(test_count)

#预处理
def nomal_img(input_images,input_anno):
    input_images= #改变数据类型