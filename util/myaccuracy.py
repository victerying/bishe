import numpy as np
import sys
from tensorflow.examples.tutorials.mnist import input_data


def myAccuracy(x: np.ndarray, y:np.ndarray) -> float:
    """
    :param x: prediction
    :param y: label (one-hot)
    :return: accuracy
    """
    assert (x.ndim == 2 and y.ndim == 2), "ndim is not equal to 2!"
    assert (x.shape == y.shape), "in line: {} shape not equal".format(str(sys._getframe().f_lineno))
    reduce_max_x = myargueMax(x)
    reduce_max_y = myargueMax(y)
    equals = reduce_max_x == reduce_max_y
    equals = np.array([1 if x else 0 for x in equals], dtype=np.float64)
    accuracy = np.average(equals)
    return accuracy


def myargueMax(x: np.ndarray) -> np.ndarray:
    index = np.zeros(shape=x.shape[0],dtype=np.int64)
    max_value = x[:, 0].copy()
    assert max_value.shape == index.shape
    for i in range(1, x.shape[1]):
        temp = max_value > x[:, i]
        temp = temp.astype(np.int16)
        max_value = temp * max_value + (1-temp) * x[:, i]
        index = temp * index + (1-temp) * i
    return index


if __name__ == "__main__":
    a = np.array([2, 3, 4])
    b = np.array([3, -1, -2])
    c = a > b
    d = c * a + (1 - c) * b
    print(c)
    print(d)
    z = np.array([[2, 3, 4], [5, 6, 7]])
    y = z[:, 1]
    print(y.shape)
    y[0] = 100
    print(z)
    x = z[1, :]
    print(x.shape)
    a = np.array([2, 3, 4])
    b = np.array([3, -1, -2])
    print(b is a)
    b = a * 1
    print(b is a)

    mnist = input_data.read_data_sets("../MNIST", one_hot=True)
    test_y = mnist.test.labels[0:100]

    label_y = myargueMax(test_y)

    for i in range(100):
        assert test_y[i, label_y[i]] == 1.0, "eror"
    print("done")
