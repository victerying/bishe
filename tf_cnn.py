import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST", one_hot=True)

batch_size = 100
num_batchs = mnist.train.num_examples // batch_size


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def biases_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='VALID')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')


x = tf.placeholder(shape=[None, 784], dtype=tf.float32)
y = tf.placeholder(shape=[None, 10], dtype=tf.float32)
print(x, y)
x_images = tf.reshape(x, shape=[-1, 28, 28, 1])
print(x_images)
# 第一层参数
w_conv1 = weight_variable([3, 3, 1, 1])
b_conv1 = biases_variable([1])
print(w_conv1, b_conv1)
# 第一层输出，conv->relu->maxpool     输出的shape:[None, 13, 13, 1]
h_conv1 = tf.nn.relu(conv2d(x_images, w_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
print(h_conv1, h_pool1)

# 全连接层参数
w_fc1 = weight_variable([13 * 13 * 1, 20])
b_fc1 = biases_variable([20])
# 全连接层输出
h_pool2_flat = tf.reshape(h_pool1, [-1, 13 * 13 * 1])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)
print(h_fc1)

# 输出层
w_fc2 = weight_variable([20, 10])
b_fc2 = biases_variable([10])
prediction = tf.matmul(h_fc1, w_fc2) + b_fc2
print(prediction)
# 损失函数
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
# 优化器
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
# 准确率
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# 初始化操作
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(100):
        for batch in range(num_batchs):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, {x: batch_xs, y: batch_ys})
            # print("finishi epoch : {}  batch : {}".format(epoch, batch))
        acc, l = sess.run([accuracy, loss], {x: mnist.test.images, y: mnist.test.labels})
        print("epoch: {} ; accuracy: {} ; loss: {} ;".format(epoch, acc, l))
