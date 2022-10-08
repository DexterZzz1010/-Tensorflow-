import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

(x, y), (x_val, y_val) = datasets.mnist.load_data()  # 直接下载数据集 （x，y）=（60k，label）
x = tf.convert_to_tensor(x, dtype=tf.float32) / 255.
y = tf.convert_to_tensor(y, dtype=tf.int32)  # 转化为tensor
y = tf.one_hot(y, depth=10)
print('datasets', x.shape, y.shape)
train_dataset = tf.data.Dataset.from_tensor_slices((x, y))
train_dataset = train_dataset.batch(200)  # 并行化 batch函数 一次返回200张 x；[200,28,28]

#三层
model = keras.Sequential(
    layers.Dense(512, activation='relu'),  # 512 256 10 输出的维度
    layers.Dense(256, activation='relu'),
    layers.Dense(10)])  # dense函数，建立每一层 dense connected 全连接

optimizer = optimizers.SGD(learning_rate=0.001)  # 自动按照规则进行更新

#step2 前向运算
def train_epoch(epoch):#所有数据集一遍epoch   batch一遍step
    # Step4.loop
    for step, (x, y) in enumerate(train_dataset):

        with tf.GradientTape() as tape:
            # [b, 28, 28] => [b, 784]
            x = tf.reshape(x, (-1, 28 * 28)) #打平 一维
            # Step1. compute output
            # [b, 784] => [b, 10]
            out = model(x)
            # Step2. compute loss
            loss = tf.reduce_sum(tf.square(out - y)) / x.shape[0]

        # Step3. optimize and update w1, w2, w3, b1, b2, b3
        grads = tape.gradient(loss, model.trainable_variables) #tape.gradient 自动求导
        # w' = w - lr * grad 更新
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if step % 100 == 0:
            print(epoch, step, 'loss:', loss.numpy())
#数据集迭代次数 batch迭代次数 loss函数值

def train():
    for epoch in range(30):#进行30个epoch 即batch迭代300*30次
        train_epoch(epoch)


if __name__ == '__main__':
    train()
