import tensorflow as tf

def euclideanDistance (X, Z):
    X_shape = X.get_shape().as_list()
    Z_shape = Z.get_shape().as_list()

    # Assert that the last index of both tensors is same
    assert X_shape[-1]==Z_shape[-1], "X and Z vectors have different dimesions"
    d = X_shape[-1]
    N1 = X_shape[0]
    N2 = Z_shape[0]
    X = tf.reshape(X, [N1,1,d])
    eucDist = tf.reduce_sum(tf.square(X - Z), axis=2)
    return eucDist
