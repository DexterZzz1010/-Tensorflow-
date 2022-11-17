import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os

import glob

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

images = glob.glob(r'D:\zhangyifei\study\Python\Tenorflow Dataset\FCN\images1\images\*.jpg')
# 然后读取目标图像
anno = glob.glob(r'D:\zhangyifei\study\Python\Tenorflow Dataset\FCN\annotations1\annotations\trimaps\*.png')
# 现在对读取进来的数据进行制作batch
np.random.seed(2019)



def read_jpg(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    return img


def read_png(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=1)
    return img


# 现在编写归一化的函数
def normal_img(input_images, input_anno):
    input_images = tf.cast(input_images, tf.float32)
    input_images = input_images / 127.5 - 1
    input_anno -= 1
    return input_images, input_anno


# 加载函数
def load_images(input_images_path, input_anno_path):
    input_image = read_jpg(input_images_path)
    input_anno = read_png(input_anno_path)
    input_image = tf.image.resize(input_image, (224, 224))
    input_anno = tf.image.resize(input_anno, (224, 224))
    return normal_img(input_image, input_anno)

index = np.random.permutation(len(images))

images = np.array(images)[index]
anno = np.array(anno)[index]
# 创建dataset
dataset = tf.data.Dataset.from_tensor_slices((images, anno))
test_count = int(len(images) * 0.2)
train_count = len(images) - test_count
data_train = dataset.skip(test_count)
data_test = dataset.take(test_count)
data_train = data_train.map(load_images, num_parallel_calls=tf.data.experimental.AUTOTUNE)
data_test = data_test.map(load_images, num_parallel_calls=tf.data.experimental.AUTOTUNE)

# 现在开始batch的制作
BATCH_SIZE = 3  # 根据显存进行调整
data_train = data_train.repeat().shuffle(100).batch(BATCH_SIZE)

data_test = data_test.batch(BATCH_SIZE)

conv_base = tf.keras.applications.VGG16(weights='imagenet',
                                        input_shape=(224, 224, 3),
                                        include_top=False)
# 现在创建子model用于继承conv_base的权重，用于获取模型的中间输出
# 使用这个方法居然能够继承，而没有显式的指定到底继承哪一个模型，确实神奇
# 确实是可以使用这个的，这个方法就是在模型建立完之后再进行的调用
# 这样就会继续自动继承之前的网络结构
# 而如果定义
sub_model = tf.keras.models.Model(inputs=conv_base.input,
                                  outputs=conv_base.get_layer('block5_conv3').output)

# 现在创建多输出模型,三个output
layer_names = [
    'block5_conv3',
    'block4_conv3',
    'block3_conv3',
    'block5_pool'
]

layers_output = [conv_base.get_layer(layer_name).output for layer_name in layer_names]

# 创建一个多输出模型，这样一张图片经过这个网络之后，就会有多个输出值了
# 不过输出值虽然有了，怎么能够进行跳级连接呢？
multiout_model = tf.keras.models.Model(inputs=conv_base.input,
                                       outputs=layers_output)

multiout_model.trainable = False

inputs = tf.keras.layers.Input(shape=(224, 224, 3))
# 这个多输出模型会输出多个值，因此前面用多个参数来接受即可。
out_block5_conv3, out_block4_conv3, out_block3_conv3, out = multiout_model(inputs)
# 现在将最后一层输出的结果进行上采样,然后分别和中间层多输出的结果进行相加，实现跳级连接
# 这里表示有512个卷积核，filter的大小是3*3
x1 = tf.keras.layers.Conv2DTranspose(512, 3,
                                     strides=2,
                                     padding='same',
                                     activation='relu')(out)
# 上采样之后再加上一层卷积来提取特征
x1 = tf.keras.layers.Conv2D(512, 3, padding='same',
                            activation='relu')(x1)
# 与多输出结果的倒数第二层进行相加，shape不变
x2 = tf.add(x1, out_block5_conv3)
# x2进行上采样
x2 = tf.keras.layers.Conv2DTranspose(512, 3,
                                     strides=2,
                                     padding='same',
                                     activation='relu')(x2)
# 直接拿到x3，不使用
x3 = tf.add(x2, out_block4_conv3)
# x3进行上采样
x3 = tf.keras.layers.Conv2DTranspose(256, 3,
                                     strides=2,
                                     padding='same',
                                     activation='relu')(x3)
# 增加卷积提取特征
x3 = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu')(x3)
x4 = tf.add(x3, out_block3_conv3)
# x4还需要再次进行上采样，得到和原图一样大小的图片，再进行分类
x5 = tf.keras.layers.Conv2DTranspose(128, 3,
                                     strides=2,
                                     padding='same',
                                     activation='relu')(x4)
# 继续进行卷积提取特征
x5 = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(x5)
# 最后一步，图像还原
preditcion = tf.keras.layers.Conv2DTranspose(3, 3,
                                             strides=2,
                                             padding='same',
                                             activation='softmax')(x5)

model = tf.keras.models.Model(
    inputs=inputs,
    outputs=preditcion
)

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['acc']  # 这个参数应该是用来打印正确率用的，现在终于理解啦啊
)

model.fit(data_train,
          epochs=1,
          steps_per_epoch=train_count // BATCH_SIZE,
          validation_data=data_test,
          validation_steps=train_count // BATCH_SIZE)