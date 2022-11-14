import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from scipy.misc import toimage
import glob
from gan import Generator, Discriminator

from dataset import make_anime_dataset


# 保存数据
def save_result(val_out, val_block_size, image_path, color_mode):
    def preprocess(img):
        img = ((img + 1.0) * 127.5).astype(np.uint8)
        # img = img.astype(np.uint8)
        return img

    preprocesed = preprocess(val_out)
    final_image = np.array([])
    single_row = np.array([])
    for b in range(val_out.shape[0]):
        # concat image into a row
        if single_row.size == 0:
            single_row = preprocesed[b, :, :, :]
        else:
            single_row = np.concatenate((single_row, preprocesed[b, :, :, :]), axis=1)

        # concat image row to final_image
        if (b + 1) % val_block_size == 0:
            if final_image.size == 0:
                final_image = single_row
            else:
                final_image = np.concatenate((final_image, single_row), axis=0)

            # reset single row
            single_row = np.array([])

    if final_image.shape[2] == 1:
        final_image = np.squeeze(final_image, axis=2)
    toimage(final_image).save(image_path)


# 真 loss 计算 真=1 ones
def celoss_ones(logits):
    # 全是1 有多少张图 多少个1
    # [b, 1]
    # [b] = [1, 1, 1, 1,]
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                                   labels=tf.ones_like(logits))
    return tf.reduce_mean(loss)

# 假 loss 计算 假=0 zeros
def celoss_zeros(logits):
    # [b, 1]
    # [b] = [0, 0, 0, 0,]
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                                   labels=tf.zeros_like(logits))
    return tf.reduce_mean(loss)


def d_loss_fn(generator, discriminator, batch_z, batch_x, is_training):
    # Discriminator 有两个目标：
    # 1. treat real image as real
    # 2. treat generated image as fake
    fake_image = generator(batch_z, is_training) # generator 得到Fake

    # 分别对真图和假图求解 （2个dataset）
    # 得到logits
    d_fake_logits = discriminator(fake_image, is_training)
    d_real_logits = discriminator(batch_x, is_training)
    # batch_x 真实值 dataset采样得到

    # 分别计算两种不同情况下的loss
    d_loss_real = celoss_ones(d_real_logits)
    d_loss_fake = celoss_zeros(d_fake_logits)
    # Loss相加 两个都小
    loss = d_loss_fake + d_loss_real

    return loss


def g_loss_fn(generator, discriminator, batch_z, is_training):
    # 只涉及到Fake image
    fake_image = generator(batch_z, is_training)
    d_fake_logits = discriminator(fake_image, is_training)
    # 希望生产的图片接近于1 所以 loss=1
    loss = celoss_ones(d_fake_logits)

    return loss


def main():
    tf.random.set_seed(22)
    np.random.seed(22)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    assert tf.__version__.startswith('2.')

    # hyper parameters
    z_dim = 100
    epochs = 3000000
    batch_size = 512
    learning_rate = 0.002
    is_training = True

    # 加载数据集
    img_path = glob.glob(path)

    dataset, img_shape, _ = make_anime_dataset(img_path, batch_size)
    print(dataset, img_shape)
    # 采样 迭代
    sample = next(iter(dataset))
    print(sample.shape, tf.reduce_max(sample).numpy(),
          tf.reduce_min(sample).numpy())
    dataset = dataset.repeat()  # 无限采样
    db_iter = iter(dataset)  # 迭代器

    generator = Generator()
    generator.build(input_shape=(None, z_dim))  # 定义参数
    discriminator = Discriminator()
    discriminator.build(input_shape=(None, 64, 64, 3))

    # 优化器  beta_1=0.5
    g_optimizer = tf.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)
    d_optimizer = tf.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)

    for epoch in range(epochs):

        # 随机采样得到Generator输入
        batch_z = tf.random.uniform([batch_size, z_dim], minval=-1., maxval=1.)
        # 真实图片
        batch_x = next(db_iter)

        # train D
        with tf.GradientTape() as tape:
            # 得到loss Gradient 更新参数
            d_loss = d_loss_fn(generator, discriminator, batch_z, batch_x, is_training)
        grads = tape.gradient(d_loss, discriminator.trainable_variables)
        d_optimizer.apply_gradients(zip(grads, discriminator.trainable_variables))

        # train G
        with tf.GradientTape() as tape:
            # 得到G的 loss Gradient 更新参数
            g_loss = g_loss_fn(generator, discriminator, batch_z, is_training)
        grads = tape.gradient(g_loss, generator.trainable_variables)
        g_optimizer.apply_gradients(zip(grads, generator.trainable_variables))

        if epoch % 100 == 0:
            print(epoch, 'd-loss:', float(d_loss), 'g-loss:', float(g_loss))

            z = tf.random.uniform([100, z_dim]) # 随机sample100张图
            # 模仿随机采样的图片画图
            fake_image = generator(z, training=False) # 测试
            # 保存图片
            img_path = os.path.join('images', 'gan-%d.png' % epoch) #设置文件名和路径
            # 两张图拼一起
            save_result(fake_image.numpy(), 10, img_path, color_mode='P')


if __name__ == '__main__':
    main()
