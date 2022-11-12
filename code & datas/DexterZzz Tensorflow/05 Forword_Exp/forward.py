import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets
import os

# 设计环境变量 只打印error
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#tf.compat.v1.enable_eager_execution
tf.enable_eager_execution()
# x: [60k, 28, 28],图片
# y: [60k] label
(x, y), _ = datasets.mnist.load_data()
# 除以255  x: [0~255] => [0~1.]
x = tf.convert_to_tensor(x, dtype=tf.float32) / 255.  # 转化成tensor
y = tf.convert_to_tensor(y, dtype=tf.int32)  # 整形

print(x.shape, y.shape, x.dtype, y.dtype)
print(tf.reduce_min(x), tf.reduce_max(x))  # 查看最小值，最大值
print(tf.reduce_min(y), tf.reduce_max(y))

# 创建数据集 决定batch数值
train_db = tf.data.Dataset.from_tensor_slices((x, y)).batch(128)
train_iter = iter(train_db)  # 叠加器
sample = next(train_iter)  # 不断调用，迭代
print('batch:', sample[0].shape, sample[1].shape)

# 创建权值 一层一层往下刮
# [b, 784] => [b, 256] => [b, 128] => [b, 10]
# [dim_in, dim_out], [dim_out]
w1 = tf.Variable(tf.random.truncated_normal([784, 256], stddev=0.1))
b1 = tf.Variable(tf.zeros([256]))  # shape=dim_out
w2 = tf.Variable(tf.random.truncated_normal([256, 128], stddev=0.1))
b2 = tf.Variable(tf.zeros([128]))
w3 = tf.Variable(tf.random.truncated_normal([128, 10], stddev=0.1))
b3 = tf.Variable(tf.zeros([10]))

lr = 1e-3
# Epoch（时期）：
# 当一个完整的数据集通过了神经网络一次并且返回了一次，这个过程称为一次>epoch。（也就是说，所有训练样本在神经网络中都 进行了一次正向传播 和一次反向传播 ）
# 再通俗一点，一个Epoch就是将所有训练样本训练一次的过程。
# 然而，当一个Epoch的样本（也就是所有的训练样本）数量可能太过庞大（对于计算机而言），就需要把它分成多个小块，也就是就是分成多个Batch 来进行训练。**
#
# Batch（批 / 一批样本）：
# 将整个训练样本分成若干个Batch。
#
# Batch_Size（批大小）：
# 每批样本的大小。
#
# Iteration（一次迭代）：
# 训练一个Batch就是一次Iteration（这个概念跟程序语言中的迭代器相似）。
#



for epoch in range(10):  # iterate db for 10   对数据集迭代
    for step, (x, y) in enumerate(train_db):  # for every batch 外层循环 对batch迭代（进度）
        # x:[128, 28, 28]
        # y: [128]
        # 数据预处理
        # [b, 28, 28] => [b, 28*28]
        x = tf.reshape(x, [-1, 28 * 28])  # 维度变化

        with tf.GradientTape() as tape:  # 默认只会跟踪tf.Variable
            # x: [b, 28*28]
            # h1 = x@w1 + b1
            # [b, 784]@[784, 256] + [256] => [b, 256] + [256] => [b, 256] + [b, 256]
            # h1 = x@w1 + tf.broadcast_to(b1, [x.shape[0], 256])
            h1 = x @ w1 + b1
            h1 = tf.nn.relu(h1)
            # [b, 256] => [b, 128]
            h2 = h1 @ w2 + b2
            h2 = tf.nn.relu(h2)#非线性环节 relu
            # [b, 128] => [b, 10]
            out = h2 @ w3 + b3

            # compute loss 计算误差
            # out: [b, 10]
            # y: [b] => [b, 10]
            # mse 均方差 10类
            y_onehot = tf.one_hot(y, depth=10)

            # mse = mean(sum(y-out)^2)
            # [b, 10]
            loss = tf.square(y_onehot - out)
            # mean: scalar 变成标量 放缩不影响梯度方向
            loss = tf.reduce_mean(loss)

        # compute gradients 自动计算梯度
        grads = tape.gradient(loss, [w1, b1, w2, b2, w3, b3])
        # print(grads)
        # w1 = w1 - lr * w1_grad  但是会返回tf.tensor类型数据
        # 将梯度更新 完成梯度下降
        w1.assign_sub(lr * grads[0])
        # 可以完成原地更新 数据类型不变 依旧是tf.vriable
        b1.assign_sub(lr * grads[1])
        w2.assign_sub(lr * grads[2])
        b2.assign_sub(lr * grads[3])
        w3.assign_sub(lr * grads[4])
        b3.assign_sub(lr * grads[5])

        #nan 梯度爆炸 初始化值 stddev=0.1
        if step % 100 == 0: #每100个batch
            print('epoch:',epoch, 'step',step, 'loss:', float(loss))
