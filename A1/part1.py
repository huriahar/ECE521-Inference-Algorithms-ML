from __future__ import print_function
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

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

def KNearestNeighbours (distances, row, k):
    distance = tf.gather(distances, row)
    top_k_vals, top_k_indices = tf.nn.top_k(tf.negative(distance), k=k)
    responsibility = tf.Variable(tf.zeros(shape=distance.get_shape().as_list(), dtype=tf.float32))
    updates_shape = top_k_indices.get_shape().as_list() + distance.get_shape().as_list()[1:]
    updates = tf.fill(dims=updates_shape, value=tf.to_float(1.0/k))
    responsibility = tf.scatter_update(responsibility, indices=top_k_indices, updates=updates)
    # responsibility is a (dim,) vector -> Reshape to (1, dim)
    dim = responsibility.get_shape().as_list()[0]
    responsibility = tf.reshape(responsibility, shape=[dim, 1])
    return responsibility

def calculate_MSE(X_train, Y_train, X_test, Y_test, K):
    distances = euclideanDistance(X_test, X_train)
    N1 = distances.get_shape().as_list()[0]
    MSE = tf.Variable(0.0, name="mse")
    for i in range(N1):
        responsibility = KNearestNeighbours(distances, i, K)
        Y_head = tf.matmul(tf.transpose(Y_train), responsibility)
        #print("Y_train: ", Y_train.shape)
        #print("Y_train transpose: ", tf.transpose(Y_train).shape)
        #print("responsibility: ", responsibility)
        #print("Y_head, ", Y_head.shape)
        #print("Y_test[i], ", Y_test[i].shape)
        #print("Y_test[i] - Y_head, ", (Y_test[i] - Y_head).shape)
        err = tf.reduce_sum(tf.square(Y_test[i] - Y_head))
        MSE += err
    MSE /= 2*N1
    return MSE
    

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
    X_valid = tf.placeholder(tf.float32, [10, 1])
    X_test = tf.placeholder(tf.float32, [10, 1])

    Y_train = tf.placeholder(tf.float32, [80, 1])
    Y_valid = tf.placeholder(tf.float32, [10, 1])
    Y_test = tf.placeholder(tf.float32, [10, 1])

    #feed_dict = {X_train: trainData, X_valid: validData, X_test: testData, Y_train: trainTarget, Y_valid: validTarget, Y_test: testTarget}

    X = [[10], [20], [30], [40]]
    Z = [[12], [36]]

    X_tf = tf.placeholder(tf.float32, [4, 1])
    Z_tf = tf.placeholder(tf.float32, [2, 1])

    feed_dict1 = {X_tf: X, Z_tf: Z}

    X1 = [[1, 2], [3, 4], [5, 6]]       # N1xd
    Y1 = [[10], [20], [30]]             # N1x1
    Z1 = [[0, 0], [1.5, 2.5]]           # N2xd
    
    X1_tf = tf.placeholder(tf.float32, [3, 2])
    Y1_tf = tf.placeholder(tf.float32, [3, 1])
    Z1_tf = tf.placeholder(tf.float32, [2, 2])

    feed_dict2 = {X1_tf:X1, Y1_tf: Y1, Z1_tf: Z1}

    Ks = [1, 3, 5, 50]
    with tf.Session() as sess:
        #distance = euclideanDistance(Z_tf, X_tf)
        #distances = euclideanDistance(Z1_tf, X1_tf)     # N2xN1
        for K in Ks:
            print("K=%d:"%K)
            MSE_train = calculate_MSE(X_train, Y_train, X_train, Y_train, K)
            MSE_valid = calculate_MSE(X_train, Y_train, X_valid, Y_valid, K)
            MSE_test = calculate_MSE(X_train, Y_train, X_test, Y_test, K)
            init = tf.global_variables_initializer()
            sess.run(init)
            print("training MSE loss: ", sess.run(MSE_train, {X_train: trainData, Y_train: trainTarget}))
            print("valid MSE loss: ", sess.run(MSE_valid, {X_train: trainData, Y_train: trainTarget, X_valid: validData, Y_valid: validTarget}))
            print("test MSE loss: ", sess.run(MSE_test, {X_train: trainData, Y_train: trainTarget, X_test: testData, Y_test: testTarget}))
            #sess.run(init)
