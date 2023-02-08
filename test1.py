import latticex.rosetta as rtt
import numpy as np
import tensorflow as tf

rtt.activate("SecureNN")
x = tf.placeholder(dtype=tf.float32, shape=[3, 4])
w = tf.argmax(x, dimension=1)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    val_w = sess.run(w, feed_dict={x: rtt.private_input(0, [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]])})
    val_w = sess.run(rtt.SecureReveal(val_w))
    print("val_w:\n{}".format(val_w))
