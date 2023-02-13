import latticex.rosetta as rtt
import numpy as np
import tensorflow as tf


tf.logging.set_verbosity(tf.logging.ERROR)
rtt.activate("SecureNN")

Alice = tf.Variable(rtt.private_input(0, np.array([1000, 2000, 3000])))
test = tf.Variable(np.array([2., 3., 4.]), dtype=tf.float32)
mul = test * Alice
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    mul: np.ndarray = sess.run(rtt.SecureReveal(mul))
    print(mul)
    print("type: {}, dtype: {}".format(type(mul), mul.dtype))
    mul = mul.astype(np.float64)
    print(mul)
    print("type: {}, dtype: {}".format(type(mul), mul.dtype))
