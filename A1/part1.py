from __future__ import print_function
import tensorflow as tf
import numpy as np
import os

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

def KNearestNeighbours (distance, k):
    top_k_vals, top_k_indices = tf.nn.top_k(tf.negative(distance), k=k)
    responsibility = tf.Variable(tf.zeros(shape=tf.shape(distance[0]), dtype=tf.float32))
    # updates.shape = indices.shape + ref.shape[1:]
    updates_shape = top_k_indices.get_shape().as_list() + responsibilities.get_shape().as_list()[1:]
    updates = tf.fill(dims=updates_shape, value=tf.to_float(1/k))
    responsibility = tf.scatter_update(responsibility, indices=top_k_indices, updates=updates)
    return responsibility

if __name__ == "__main__":
    np.random.seed(521)
    Data = np.linspace(1.0 , 10.0 , num =100) [:, np.newaxis]
    Target = np.sin( Data ) + 0.1 * np.power( Data , 2) + 0.5 * np.random.randn(100 , 1)
    randIdx = np.arange(100)
    np.random.shuffle(randIdx)
    trainData, trainTarget = Data[randIdx[:80]], Target[randIdx[:80]]
    validData, validTarget = Data[randIdx[80:90]], Target[randIdx[80:90]]
    testData, testTarget = Data[randIdx[90:100]], Target[randIdx[90:100]]

    X_train = tf.placeholder(tf.float32, [80, 1])
    X_vt = tf.placeholder(tf.float32, [10, 1])

    feed_dict = {X_train: trainData, X_vt: validData}

    X = [[10], [20], [30], [40]]
    Z = [[12], [36]]

    X_tf = tf.placeholder(tf.float32, [4, 1])
    Z_tf = tf.placeholder(tf.float32, [2, 1])

    feed_dict1 = {X_tf: X, Z_tf: Z}

    X1 = [[1, 2], [3, 4], [5, 6]]
    Z1 = [[0, 0], [1.5, 2.5]]
    
    X1_tf = tf.placeholder(tf.float32, [3, 2])
    Z1_tf = tf.placeholder(tf.float32, [2, 2])

    feed_dict2 = {X1_tf:X1, Z1_tf: Z1}

    K = 2

    #distance = euclideanDistance(Z_tf, X_tf)
    distance = euclideanDistance(Z1_tf, X1_tf)
    #distance = euclideanDistance(X_vt, X_train)

    KNearestNeighbours(distance, K)

    sess = tf.InteractiveSession()
    init = tf.global_variables_initializer()

    sess.run(init)
