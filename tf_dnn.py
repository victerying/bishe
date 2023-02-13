import numpy
import tensorflow
import os
from tensorflow.examples.tutorials.mnist import input_data
from typing import List

tensorflow.logging.set_verbosity(tensorflow.logging.ERROR)

mnist = input_data.read_data_sets('MNIST', one_hot=True)
# print(mnist)


train_x = mnist.train.images
train_y = mnist.train.labels
test_x = mnist.test.images
test_y = mnist.test.labels
print('X_train: {}  Y_train: {} X_test: {} Y_test: {}'.format(train_x.shape, train_y.shape, test_x.shape, test_y.shape))
print('type:{} dtype:{}'.format(type(test_x), [test_x.dtype, train_x.dtype, train_y.dtype, train_y.dtype]))
assert (train_x.shape[0] == train_y.shape[0])

# 三层（784 * 128 relu， 128 * 256 relu, 256 * 10）
layers_param = [128, 256]
learning_rate = 0.01
batch_size = 50
num_batch = mnist.train.num_examples // batch_size
num_epoch = 30

num_feature = train_x.shape[1]
num_class = train_y.shape[1]
print("batch_size:{} num_batch:{} num_feature:{} num_class:{} ".format(
    batch_size, num_batch, num_feature, num_class))

placeholder_x: tensorflow.Tensor = tensorflow.placeholder(tensorflow.float64, [None, num_feature], name="x-input")
placeholder_y: tensorflow.Tensor = tensorflow.placeholder(tensorflow.float64, [None, num_class], name="y-input")
keep_prob: tensorflow.Tensor = tensorflow.placeholder(tensorflow.float64)


# 构建全连接神经网络
def build_dnn(x: tensorflow.Tensor, layers_param: List[int]) -> (
        List[tensorflow.Variable], List[tensorflow.Variable], tensorflow.Tensor):
    w = []
    b = []


    num_layers = len(layers_param) + 1
    layers_param.insert(0, num_feature)
    layers_param.append(num_class)
    for layer_count in range(num_layers):
        w.append(tensorflow.Variable(
            initial_value=tensorflow.random_normal(
                shape=[layers_param[layer_count], layers_param[layer_count + 1]],
                dtype=tensorflow.float64,
                seed=1),
            name="w_{:04}".format(layer_count),
            dtype=tensorflow.float64))
        b.append(tensorflow.Variable(
            initial_value=tensorflow.random_normal(
                shape=[layers_param[layer_count + 1]],
                dtype=tensorflow.float64,
                seed=1),
            name="b_{:04}".format(layer_count),
            dtype=tensorflow.float64))
        output_layer = x


    for layer_count in range(num_layers):
        output_layer = tensorflow.matmul(output_layer, w[layer_count]) + b[layer_count]
        # 最后一层没有激活函数
        if layer_count != num_layers - 1:
            output_layer = tensorflow.nn.relu(output_layer)
            # output_layer = tensorflow.nn.dropout(output_layer, keep_prob=keep_prob)

    return w, b, output_layer


w, b, output_layer = build_dnn(placeholder_x, layers_param)
# 损失函数
loss = tensorflow.reduce_mean(
    tensorflow.nn.softmax_cross_entropy_with_logits(logits=output_layer, labels=placeholder_y))
# 训练 operation
train_op = tensorflow.train.GradientDescentOptimizer(learning_rate).minimize(loss)
# 初始化 operation
global_init_op = tensorflow.global_variables_initializer()
# 准确率
correct_prediction = tensorflow.equal(tensorflow.argmax(output_layer, 1), tensorflow.argmax(placeholder_y, 1))
accuracy = tensorflow.reduce_mean(tensorflow.cast(correct_prediction, tensorflow.float32))


with tensorflow.Session() as sess:
    sess.run(global_init_op)
    for epoch_count in range(num_epoch):
        for batch_count in range(num_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            sess.run(fetches=train_op, feed_dict={placeholder_x: batch_x, placeholder_y: batch_y, keep_prob: 0.5})
        val_accuracy = sess.run(accuracy, feed_dict={placeholder_x: test_x, placeholder_y: test_y, keep_prob: 1.0})
        print("epoch_count: {} val_accuracy:{}".format(epoch_count, val_accuracy))

print("end process")
