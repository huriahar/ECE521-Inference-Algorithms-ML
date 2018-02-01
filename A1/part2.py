from __future__ import print_function
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import part1
from matplotlib import interactive

def KNearestNeighbours (distances, k):
    top_k_vals, top_k_indices = tf.nn.top_k(tf.negative(distances), k=k)
    N = distances.get_shape().as_list()[1]
    one_hot = tf.one_hot(top_k_indices, depth=N, on_value=tf.to_float(1.0/k))
    responsibility = tf.reduce_sum(one_hot, axis=1)
    return responsibility

def calculateMSE(X_train, Y_train, X_test, Y_test, K):
    distances = part1.euclideanDistance(X_test, X_train)
    N1 = distances.get_shape().as_list()[0]
    MSE = tf.Variable(0.0, name="mse")
    responsibility = KNearestNeighbours(distances, K)
    Y_head = tf.matmul(responsibility, Y_train)
    MSE = tf.reduce_sum(tf.square(Y_test - Y_head))
    MSE /= tf.to_float(2*N1)
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

    Ks = [1, 3, 5, 50]
    with tf.Session() as sess:
        for K in Ks:
            print("K=%d:"%K)
            MSE_train = calculateMSE(X_train, Y_train, X_train, Y_train, K)
            MSE_valid = calculateMSE(X_train, Y_train, X_valid, Y_valid, K)
            MSE_test = calculateMSE(X_train, Y_train, X_test, Y_test, K)
            init = tf.global_variables_initializer()
            sess.run(init)
            print("training MSE loss: ", sess.run(MSE_train, {X_train: trainData, Y_train: trainTarget}))
            print("valid MSE loss: ", sess.run(MSE_valid, {X_train: trainData, Y_train: trainTarget, X_valid: validData, Y_valid: validTarget}))
            print("test MSE loss: ", sess.run(MSE_test, {X_train: trainData, Y_train: trainTarget, X_test: testData, Y_test: testTarget}))
    
    New_Data = np.linspace(0.0 , 11.0 , num =1000) [:, np.newaxis]
    New_Target = np.sin( New_Data ) + 0.1 * np.power( New_Data , 2) + 0.5 * np.random.randn(1000 , 1)

    X_new_test = tf.placeholder(tf.float32, [1000,1])
    Deuc = part1.euclideanDistance(X_new_test, X_train)
    Predictions = dict.fromkeys(Ks)
    with tf.Session() as sess:
        for K in Ks:
            responsibility = KNearestNeighbours(Deuc, K)
            Y_head = tf.matmul(responsibility, Y_train)
            Y_head = tf.reduce_sum(Y_head, axis=1)
            init = tf.global_variables_initializer()
            sess.run(init)
            Predictions[K] = sess.run(Y_head, {X_train: trainData, X_new_test: New_Data, Y_train: trainTarget})
    plt.close('all')
    figs = dict.fromkeys(range(1,len(Ks)+1))
    for i, K in enumerate(Ks):
        figs[i+1] = plt.figure(i+1)
        plt.scatter(New_Data, New_Target, marker='.', label='True Target')
        plt.xlabel('input Data (X)')
        plt.ylabel('predictions (Y head) and target values (Y)')
        plt.plot(New_Data, Predictions[K], c='r', label=("Predictions when K=%d" % K))
        plt.legend()
    plt.show()
        
