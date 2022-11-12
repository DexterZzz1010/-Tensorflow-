import  tensorflow as tf
import os

gpu_device_name=tf.test.is_gpu_available()
print(gpu_device_name)

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
print('GPU',tf.test.is_gpu_available())
a=tf.constant(2.)
b=tf.constant(2.)
print(a+b)