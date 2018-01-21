from __future__ import print_function
import tensorflow as tf
import os

def euclideanDistance(x, z):
    t_x = tf.convert_to_tensor(x)
    t_z = tf.convert_to_tensor(z)
    x_d = t_x.get_shape().as_list()[-1]
    z_d = t_x.get_shape().as_list()[-1]
    assert x_d==z_d, "x and z vectors have different dimesions"
    d = x_d
    N1 = t_x.get_shape().as_list()[0]
    N2 = t_z.get_shape().as_list()[0]
    t_x_reshaped = tf.reshape(t_x, [N1,1,d])
    Deuc = tf.square(t_x_reshaped -t_z)
    Deuc = tf.reduce_sum(Deuc,2)
    with tf.Session() as session:
        print(session.run(Deuc))


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    x = [[1, 2, 3],[4,5,6]]
    z = [[-1, -2, -3]]
    d = euclideanDistance(x, z)

    m = [[1, 2, 3],[4,5,6]]
    n = [[-1, -2, -3],[1,2,3],[4,5,6]]
    d = euclideanDistance(m, n)
