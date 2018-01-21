from __future__ import print_function
import tensorflow as tf
import os

def euclideanDistance(X, Z):
    t_X = tf.convert_to_tensor(X)
    t_Z = tf.convert_to_tensor(Z)
    x_d = t_X.get_shape().as_list()[-1]
    z_d = t_X.get_shape().as_list()[-1]
    assert x_d==z_d, "x and z vectors have different dimesions"
    d = x_d
    N1 = t_X.get_shape().as_list()[0]
    N2 = t_Z.get_shape().as_list()[0]
    t_X_reshaped = tf.reshape(t_X, [N1,1,d])
    Deuc = tf.square(t_X_reshaped - t_Z)
    Deuc = tf.reduce_sum(Deuc,2)
    with tf.Session() as session:
        print(session.run(Deuc))


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    X = [[1, 2, 3],[4,5,6]]
    Z = [[-1, -2, -3]]
    d = euclideanDistance(X, Z)

    M = [[1, 2, 3],[4,5,6]]
    N = [[-1, -2, -3],[1,2,3],[4,5,6]]
    d = euclideanDistance(M, N)
