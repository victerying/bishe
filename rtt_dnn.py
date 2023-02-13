import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from util.myaccuracy import myAccuracy

tf.logging.set_verbosity(tf.logging.ERROR)
enable_rtt = True
# 加载数据集
mnist = input_data.read_data_sets("MNIST", one_hot=True)

num_epoch = 10
batch_size = 100
num_batch = mnist.train.num_examples // batch_size

"""
train_x = mnist.train.images
train_y = mnist.train.labels
num_feature = train_x.shape[1]
num_class = train_y.shape[1]
assert num_feature == 784
assert num_class == 10
"""
num_feature = 784
num_class = 10
print("num_feature: {}, num_class: {}, num_batchs: {}, batch_size: {}".format(num_class, num_feature, num_batch,
                                                                              batch_size))
test_x = mnist.test.images[0:100]
test_y = mnist.test.labels[0:100]
print("shape test_x: {}, test_y: {}\n"
      "dtype test_x: {}, test_y: {}".format(test_x.shape, test_y.shape, test_x.dtype, test_y.dtype))
if enable_rtt:
    import latticex.rosetta as rtt
    rtt.activate("SecureNN")
    """    
    test_x = rtt.private_input(0, test_x)
    test_y = rtt.private_input(1, test_y)
    """
placeholder_x = tf.placeholder(tf.float64, [None, 784])
placeholder_y = tf.placeholder(tf.float64, [None, 10])
w_1 = tf.Variable(
    initial_value=tf.random_normal(shape=[784, 90], seed=1, dtype=tf.float64),
    name="w_1",
    dtype=tf.float64)
b_1 = tf.Variable(
    initial_value=tf.random_normal(shape=[90], seed=1, dtype=tf.float64),
    name="b_1",
    dtype=tf.float64)
layer1 = tf.nn.relu(tf.matmul(placeholder_x, w_1) + b_1)

w_2 = tf.Variable(
    initial_value=tf.random_normal(shape=[90, 10], seed=1, dtype=tf.float64),
    name="w_2",
    dtype=tf.float64)
b_2 = tf.Variable(
    initial_value=tf.random_normal(shape=[10], seed=1, dtype=tf.float64),
    name="b_2",
    dtype=tf.float64)
prediction = tf.matmul(layer1, w_2) + b_2

# 损失函数
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=prediction, labels=placeholder_y))
# 优化器
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
# 准确率 tf.argmax 在rtt下有问题
"""
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(placeholder_y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
"""

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(num_epoch):
        for batch in range(num_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            if enable_rtt:
                batch_xs = rtt.private_input(0, batch_xs)
                batch_ys = rtt.private_input(0, batch_ys)
            sess.run(train_step, {placeholder_x: batch_xs, placeholder_y: batch_ys})
            if batch % 55 == 54:
                print("finish epoch: {}, batch: {}".format(epoch, batch))
        pred, l = sess.run(
            fetches=[prediction, loss],
            feed_dict={
                placeholder_x: rtt.private_input(0, test_x),
                placeholder_y: rtt.private_input(1, test_y)
            })
        # acc = sess.run(accuracy, feed_dict={placeholder_x: test_x, placeholder_y: test_y})

        if enable_rtt:
            # acc = sess.run(rtt.SecureReveal(acc))
            l = sess.run(rtt.SecureReveal(l))
            pred = sess.run(rtt.SecureReveal(pred))
            pred = pred.astype(np.float64)
            acc = myAccuracy(pred, test_y)
        print("epoch: {} ; loss: {} ; accuracy: {};".format(epoch, l, acc))

rtt.deactivate()
