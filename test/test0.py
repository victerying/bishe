#!/usr/bin/env python3
import latticex.rosetta as rtt
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("../MNIST", one_hot=True)
train_x, train_y = mnist.train.next_batch(10)

print("shape train_x: {}, train_y: {}".format(train_x.shape, train_y.shape))
rtt.activate("SecureNN")
placeholder_x = tf.placeholder(tf.float32, [None, train_x.shape[1]])
placeholder_y = tf.placeholder(tf.float32, [None, train_y.shape[1]])
w = tf.Variable(
    initial_value=tf.ones(
        shape=train_x.shape,
        dtype=tf.float32
    ),
    dtype=tf.float32,
    name="w_{:04d}".format(1)
)
print("w.shape: {}".format(w.shape))
assign_w = tf.assign(w, w * placeholder_x)

Alice = tf.Variable(rtt.private_input(0, np.array([1000, 2000, 3000])))
Bob = tf.Variable(rtt.private_input(1, [999, 1999, 3001]))
test = tf.Variable(np.array([2., 3., 4.]), dtype=tf.float32)
const_mul_share = test * Alice
res = tf.greater(Alice, Bob)
print("alice.shape: {}, alice.dtype: {}, test.shape: {}, test.dtype: {}".format(
    Alice.shape, Alice.dtype, test.shape, test.dtype))
print("alice: {}\nbob: {} \ntest: {}\nres: {}\nconst_mul_var: {}".format(Alice, Bob, test, res, const_mul_share))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    val_alice = sess.run(Alice)
    print("val_alice: {}".format(val_alice))
    val_alice_plain = sess.run(rtt.SecureReveal(Alice))
    print("val_alice(plaintext): {}".format(val_alice_plain))
    val_test = sess.run(test)
    print("val_test: {}".format(val_test))
    val_const_mul_share = sess.run(const_mul_share)
    print("val_const_mul_share: {}".format(val_const_mul_share))
    print("val_const_mul_share(plaintext): {}".format(sess.run(rtt.SecureReveal(const_mul_share))))
    plain_res = sess.run(rtt.SecureReveal(res))
    print('ret:', plain_res)  # ret: 1.0

    val_w = sess.run(w)
    print("w(before assign): {}".format(val_w))
    sess.run(assign_w, feed_dict={placeholder_x : rtt.private_input(0, train_x), placeholder_y: rtt.private_input(1, train_y)})
    val_w = sess.run(w)
    print("w(after  assign): {} w.shape: {}".format(val_w, val_w.shape))

    plain_w = sess.run(rtt.SecureReveal(w))
    print("w(after  assign): {}".format(plain_w))

rtt.deactivate()
