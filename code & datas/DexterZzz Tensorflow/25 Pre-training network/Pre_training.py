import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
# %matplotlib inline
import numpy as np
import glob
import os
import warnings

warnings.filterwarnings('ignore')

keras = tf.keras
layers = tf.keras.layers
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# tf.random.set_seed(2345)
# np.random.seed(2345)


def load_preprosess_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [360, 360])
    image = tf.image.random_crop(image, [256, 256, 3])
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, 0.5)
    image = tf.image.random_contrast(image, 0, 1)
    image = tf.cast(image, tf.float32)
    image = image / 255
    label = tf.reshape(label, [1])
    return image, label


BATCH_SIZE = 32

# 导入数据集 预处理
train_image_path = glob.glob(r'D:\zhangyifei\study\Python\Tenorflow Dataset\dc_2000\train\*\*.jpg')
train_image_label = [int(p.split('\\')[1] == 'cat') for p in train_image_path]
train_image_ds = tf.data.Dataset.from_tensor_slices((train_image_path, train_image_label))
AUTOTUNE = tf.data.experimental.AUTOTUNE
train_image_ds = train_image_ds.map(load_preprosess_image, num_parallel_calls=AUTOTUNE)
train_image_ds = train_image_ds.shuffle(train_count).batch(BATCH_SIZE)
train_image_ds = train_image_ds.prefetch(AUTOTUNE)
train_count = len(train_image_path)

# 导入测试集 预处理
test_image_path = glob.glob(r'D:\zhangyifei\study\Python\Tenorflow Dataset\dc_2000\test\*\*.jpg')
test_image_label = [int(p.split('\\')[1] == 'cat') for p in test_image_path]
test_image_ds = tf.data.Dataset.from_tensor_slices((test_image_path, test_image_label))
test_image_ds = test_image_ds.map(load_preprosess_image, num_parallel_calls=AUTOTUNE)
test_image_ds = test_image_ds.batch(BATCH_SIZE)
test_image_ds = test_image_ds.prefetch(AUTOTUNE)
test_count = len(test_image_path)

# 预训练模型
covn_base = keras.applications.VGG16(weights='imagenet', include_top=False)
# weights='imagenet'采用在imagenet训练集上训练好的权重
# include_top=False 不引入分类器（全连接层）
covn_base.summary()

# 预训练模型不可训练
covn_base.trainable = False

# 建立网络
model = keras.Sequential()
model.add(covn_base)  # 添加卷积基  不可训练！！！
model.add(layers.GlobalAveragePooling2D())  # 类似flatten 参数更少
model.add(layers.Dense(512, activation='relu'))  # 输出512
model.add(layers.Dense(1, activation='sigmoid'))  # 输出1
model.summary()

# compile
model.compile(optimizer=keras.optimizers.Adam(lr=0.001),
              loss='binary_crossentropy',
              metrics=['acc'])
# 训练模型
model.fit(test_image_ds, steps_per_epoch=train_count // BATCH_SIZE, epochs=15, validation_data=test_image_ds,
          validation_steps=test_count//BATCH_SIZE)

# 微调
# 冻结模型库的底部卷积层
# 共同训练新添加的分类器层和顶部卷积层
#只有分类器已经训练好了，才能微调卷积基的顶部卷积层 刚开始训练误差很大 会破坏权值
