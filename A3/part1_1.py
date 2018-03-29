from __future__ import print_function
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

sess = tf.InteractiveSession()

def loadData (fileName):
    with np.load(fileName) as data:
        Data, Target = data["images"], data["labels"]
        np.random.seed(521)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data = Data[randIndx]/255.0
        Data = Data.reshape(-1,28*28)
        Target = Target[randIndx]
        trainData, trainTarget = Data[:15000], Target[:15000]
        validData, validTarget = Data[15000:16000], Target[15000:16000]
        testData, testTarget = Data[16000:], Target[16000:]
        return trainData, trainTarget, validData, validTarget, testData, testTarget

def layerBuildingBlock (XLMinus1, numHiddenUnits):
    # Initialize W - Use Xavier initialization for weight matrix
    # dLMinus1 is actually dLMinus1 + 1 -> Dimension of previous layer with the bias node
    dimLMinus1 = XLMinus1.shape[0]
    assert(XLMinus1.shape[1] == 1)
    W = tf.Variable(shape=[dimLMinus1, numHiddenUnits], dtype=tf.float64, initializer=tf.contrib.layers.xavier_initializer(), name="weights")
    b = tf.Variable(tf.zeros(shape=[numHiddenUnits], dtype=tf.float64, name="bias"))
    return tf.matmul(XLMinus1, W) +b

if __name__ == '__main__':
    trainData, trainTarget, validData, validTarget, testData, testTarget = loadData("notMNIST.npz")

    # ReLU activation function, cross-entropy cost function, softmax output layer
    # NN with 1 hidden layer and 1000 units

    # Add 1s before as first element for every row in the matrix
    # trainData = np.insert(trainData, 0, [1], 1)
    d = trainData.shape[1]  # 28*28 = 784
    N = len(trainData)      # 15000

    batchSize = 500
    # Training and test data for each mini batch in each epoch
    XTrain = tf.placeholder(tf.float64, [batchSize, d])
    YTrain = tf.placeholder(tf.float64, [batchSize, 1])

    XValid = tf.placeholder(tf.float64, validData.shape)
    YValid = tf.placeholder(tf.float64, validTarget.shape)

    XTest = tf.placeholder(tf.float64, testData.shape)
    YTest = tf.placeholder(tf.float64, testTarget.shape)

    iteration = 20000.
    lda = 3e-4

    iterPerEpoch = int(N / batchSize)                       # 30
    epochs = int(np.ceil(iteration/float(iterPerEpoch)))    # 667

    learnRate = [0.005]#, 0.001, 0.0001]

    plt.close()
    init = tf.global_variables_initializer()
    sess.run(init)
