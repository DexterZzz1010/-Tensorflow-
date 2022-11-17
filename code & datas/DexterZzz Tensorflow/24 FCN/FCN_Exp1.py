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
    # JPG 读取+解码 Channel
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    return img


def read_png(path):
    # PNG 读取+解码
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=1)
    return img


# 预处理
def normal_img(input_images, input_anno):
    # 改变数据类型
    input_images = tf.cast(input_images, tf.float32)
    # 归一化
    input_images = input_images / 127.5 - 1
    input_anno -= 1  # [1,2,3]-->[0,1,2]
    return input_images, input_anno


# 加载图像函数
def load_images(input_image_path, input_anno_path):
    # input_image:jpeg,input_anno:png
    input_image = read_jpg(input_image_path)
    input_anno = read_png(input_anno_path)
    # 统一大小
    input_image = tf.image.resize(input_image, (224, 224))
    input_anno = tf.image.resize(input_anno, (224, 224))
    return normal_img(input_image, input_anno)


# 构建数据集和测试集
def Dataset(images, anno):
    # 乱序
    index = np.random.permutation(len(images))
    images = np.array(images)[index]
    anno = np.array(anno)[index]

    dataset = tf.data.Dataset.from_tensor_slices((images, anno))
    test_count = int(len(images) * 0.2)  # 五分之一作为测试集
    train_count = len(images) - test_count
    print(test_count, train_count)

    data_train = dataset.skip(test_count)
    data_test = dataset.take(test_count)

    data_train = data_train.map(load_images,
                                num_parallel_calls=tf.data.experimental.AUTOTUNE)  # 多线程读取
    data_test = data_test.map(load_images,
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)

    data_train = data_train.repeat().shuffle(100).batch(BATCH_SIZE)
    data_test = data_test.batch(BATCH_SIZE)

    # take(1) 是一个batch
    # for img, anno in data_train.take(1):
    #     plt.subplot(1, 2, 1)
    #     # 显示第一张
    #     plt.imshow(tf.keras.preprocessing.image.array_to_img(img[0]))
    #     plt.subplot(1, 2, 2)
    #     plt.imshow(tf.keras.preprocessing.image.array_to_img(anno[0]))
    #     plt.show()

    return data_train, data_test, test_count, train_count



# 分成三类

BATCH_SIZE = 8

# 读取图像 glob
images = glob.glob(r'D:\zhangyifei\study\Python\Tenorflow Dataset\FCN\images1\images\*.jpg')
anno = glob.glob(r'D:\zhangyifei\study\Python\Tenorflow Dataset\FCN\annotations1\annotations\trimaps\*.png')

# print(images[-5::])
# data_train, data_test, test_count, train_count = Dataset(images, anno)

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



# 加载卷积基
conv_base = tf.keras.applications.VGG16(weights='imagenet',
                                        input_shape=(224, 224, 3),
                                        include_top=False)
conv_base.summary()

# 获取中间层输出 conv_base.get_layer().output
# conv_base.get_layer('block5_conv3').output         # [b,14,14,512]

# 定义一个子网络：没有block5_pool这一层（定义input和output） 继承了conv_base权重
'''sub_model=tf.keras.models.Model(inputs=conv_base.input,
                                outputs=conv_base.get_layer('block5_conv3').output)'''

# 创建多输出模型 用列表定义输出
layer_names = [
    'block5_conv3',
    'block4_conv3',
    'block3_conv3',
    'block5_pool'
]
# 获取列表中所有层的输出 conv_base.get_layer().output
layers_output = [conv_base.get_layer(layer_name).output for layer_name in layer_names]

# 四个输出
multi_output_model = tf.keras.models.Model(inputs=conv_base.input,
                                           outputs=layers_output)
multi_output_model.trainable = False  # 冻结

# 创建语义分割FCN网络
inputs = tf.keras.layers.Input(shape=(224, 224, 3))
out_block5_conv3, out_block4_conv3, out_block3_conv3, out = multi_output_model(inputs)  # 返回四个输出
print(out.shape, out_block5_conv3.shape)

# 对out进行上采样
# tf.add 相加
# 上采样（deconv,tanspose） block5_pool(out)
x1 = tf.keras.layers.Conv2DTranspose(512, 3,
                                     strides=2, padding='same',
                                     activation='relu')(out)  # [b,7,7,512]-->deconv-->[b,14,14,512]
x1 = tf.keras.layers.Conv2D(512, 3, padding='same',
                            activation='relu')(x1)  # 增加卷积进一步提取特征 不改变维度

x2 = tf.add(x1, out_block5_conv3)
x2 = tf.keras.layers.Conv2DTranspose(512, 3,
                                     strides=2, padding='same',
                                     activation='relu')(x2)  # [b,14,14,512]-->deconv-->[b,28,28,512]
x2 = tf.keras.layers.Conv2D(512, 3, padding='same',
                            activation='relu')(x2)

x3 = tf.add(x2, out_block4_conv3)
x3 = tf.keras.layers.Conv2DTranspose(256, 3,
                                     strides=2, padding='same',
                                     activation='relu')(x3)  # [b,28,28,512]-->deconv-->[b,56,56,256]
x3 = tf.keras.layers.Conv2D(256, 3, padding='same',
                            activation='relu')(x3)

x4 = tf.add(x3, out_block3_conv3)
# 分成3类
x5 = tf.keras.layers.Conv2DTranspose(128, 3,
                                     strides=2, padding='same',
                                     activation='relu')(x4)  # [b,56,56,256]-->deconv-->[b,112,112,128]
x5 = tf.keras.layers.Conv2D(128, 3, padding='same',
                            activation='relu')(x5)

prediction = tf.keras.layers.Conv2DTranspose(3, 3,
                                             strides=2, padding='same',
                                             activation='softmax')(x5)  # [b,112,112,128]-->deconv-->[b,224,224,3]

model = tf.keras.models.Model(
    inputs=inputs,
    outputs=prediction
)
model.summary()

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['acc'])

# print(data_train.shape)
model.fit(data_train,
          epochs=5,
          steps_per_epoch=train_count // BATCH_SIZE,
          validation_data=data_test,
          validation_steps=test_count // BATCH_SIZE)
