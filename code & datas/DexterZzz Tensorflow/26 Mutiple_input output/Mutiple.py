import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import warnings
import random
import pathlib

# import IPython.display as display

keras = tf.keras
layers = tf.keras.layers
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf.random.set_seed(2345)
np.random.seed(2345)


def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    image = tf.cast(image, tf.float32)
    image = image / 255.0  # normalize to [0,1] range
    image = 2 * image - 1
    return image


data_dir = r'D:\zhangyifei\study\Python\Tenorflow Dataset\multi-output-classification多输出模型数据集\multi-output-classification\dataset'
data_root = pathlib.Path(data_dir)
BATCH_SIZE = 16

for item in data_root.iterdir():
    print(item)
all_image_paths = list(data_root.glob('*/*'))
image_count = len(all_image_paths)

all_image_paths = [str(path) for path in all_image_paths]
# 乱序
random.shuffle(all_image_paths)

# 建立label列表
'''总结，不同转换数据的方式执行顺序如下：
   创建实例
   重组（较大的buffer_size）
   重复
   数据预处理、数据扩增，使用多线程等
   批次化
   预取数据'''

# 提取上级目录 label名
label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
print(label_names)
color_label_names = set(name.split('_')[0] for name in label_names)
item_label_names = set(name.split('_')[1] for name in label_names)

# 编码
# black0 blue1 red2
color_label_to_index = dict((name, index) for index, name in enumerate(color_label_names))
# dress0 jeans1 shirt2 shoes3
item_label_to_index = dict((name, index) for index, name in enumerate(item_label_names))

all_image_labels = [pathlib.Path(path).parent.name for path in all_image_paths]
all_image_labels[:5]

# 将颜色和物品转换成数字类型
color_labels = [color_label_to_index[label.split('_')[0]] for label in all_image_labels]
item_labels = [item_label_to_index[label.split('_')[1]] for label in all_image_labels]

# for n in range(3):
#     image_index = random.choice(range(len(all_image_paths)))
#     display.display(display.Image(all_image_paths[image_index], width=100, height=100))
#     print(all_image_labels[image_index])
#     print()

img_path = all_image_paths[0]
img_raw = tf.io.read_file(img_path)
img_tensor = tf.image.decode_image(img_raw)

img_tensor = tf.cast(img_tensor, tf.float32)
img_tensor = tf.image.resize(img_tensor, [224, 224])
img_final = img_tensor / 255.0

# 创建dataset
path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
AUTOTUNE = tf.data.experimental.AUTOTUNE
image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)  # image 通道
label_ds = tf.data.Dataset.from_tensor_slices((color_labels, item_labels))

# for ele in label_ds.take(3):
#     print(ele[0].numpy(), ele[1].numpy())


# zip在一起 (image_ds, label_ds) =（（224，224，3），（（），（）））
image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))
# 划分数据集和训练集
test_count = int(image_count * 0.2)
train_count = image_count - test_count

train_data = image_label_ds.skip(test_count)
test_data = image_label_ds.take(test_count)

# shuffle 打乱训练集
train_data = train_data.shuffle(buffer_size=train_count).repeat(-1)
train_data = train_data.batch(BATCH_SIZE)

train_data = train_data.prefetch(buffer_size=AUTOTUNE)
test_data = test_data.batch(BATCH_SIZE)

# 建立模型
# 预训练网络 部署在移动设备上
mobile_net = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3),
                                               include_top=False,
                                               weights='imagenet')
mobile_net.trianable = False
inputs = tf.keras.Input(shape=(224, 224, 3))
x = mobile_net(inputs)
x = tf.keras.layers.GlobalAveragePooling2D()(x)  # [None,1280]

# 通过两个全连接层分别输出
x1 = tf.keras.layers.Dense(1024, activation='relu')(x)

# 输出为label数量 规定name属性
out_color = tf.keras.layers.Dense(len(color_label_names),
                                  activation='softmax',
                                  name='out_color')(x1)

x2 = tf.keras.layers.Dense(1024, activation='relu')(x)
out_item = tf.keras.layers.Dense(len(item_label_names),
                                 activation='softmax',
                                 name='out_item')(x2)

model = tf.keras.Model(inputs=inputs,
                       outputs=[out_color, out_item])
model.summary()

# 设定参数 配置模型 (优化器 loss lr metrics)
# 用字典写loss {}
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss={'out_color': 'sparse_categorical_crossentropy',
                    'out_item': 'sparse_categorical_crossentropy'},
              metrics=['acc']
              )

# 训练多少步
train_steps = train_count // BATCH_SIZE
test_steps = test_count // BATCH_SIZE

# 训练网络
model.fit(train_data,
          epochs=5,
          steps_per_epoch=train_steps,
          validation_data=test_data,
          validation_steps=test_steps
          )

model.evaluate(test_data)

# predict
my_image = load_and_preprocess_image(
    r'D:\zhangyifei\study\Python\Tenorflow Dataset\multi-output-classification多输出模型数据集\multi-output-classification\dataset\blue_jeans\00000004.jpg')
# [224, 224, 3]-->[1, 224, 224, 3] 扩充batch维度
my_image = tf.expand_dims(my_image, 0)
pred = model.predict(my_image)  # [3]

index_to_item= {index:item for item,index in item_label_to_index.items()}
print(index_to_item)
index_to_color= {index:color for color,index in color_label_to_index.items()}
# black0 blue1 red2
# dress0 jeans1 shirt2 shoes3

pre_color = index_to_color.get(np.argmax(pred[0][0]))
pre_item = index_to_item.get(np.argmax(pred[1][0]))
print(pre_color,pre_item)
print(np.argmax(pred[0]), np.argmax(pred[1]))
