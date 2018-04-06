from __future__ import print_function
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import part1_1and2 as p1
import random
from math import exp
import time
import pickle

def layerBuildingBlock (XLMinus1, numHiddenUnits):
    # Initialize W - Use Xavier initialization for weight matrix
    dimLMinus1 = XLMinus1.get_shape().as_list()[1]
    xavierStdDev = np.sqrt(3.0/(dimLMinus1 + numHiddenUnits))
    # Alternate method
    # W = tf.Variable(shape=[dimLMinus1, numHiddenUnits], dtype=tf.float64, initializer=tf.contrib.layers.xavier_initializer(), name="weights")
    W = tf.Variable(tf.truncated_normal(shape=[dimLMinus1, numHiddenUnits], dtype=tf.float64, stddev=xavierStdDev, name="weights"))
    b = tf.Variable(tf.zeros(shape=[numHiddenUnits], dtype=tf.float64, name="bias"))
    return tf.matmul(XLMinus1, W) + b, W, b

#1. set the random seeds in Numpy and TensorFlow libraries
#diferently from other groups. (e.g. seed using the time of your experiments or some com-
#bination of your student IDs).
#2. Randomly sample the natural log of learning rate uniformly
#between -7.5 and -4.5
#3. the number of layers from 1 to 5
#4. the number of hidden units per layer between 100 and 500,
#5. the natural log of weight decay coeficient uniformly from
#the interval [-9, -6].
#6. Also, randomly choose your model to use dropout or not. Using these
#hyperparameters, train your neural network and report its validation and test classication
#error. Repeat this task for ve models, and report their results.

if __name__ == '__main__':
    trainData, trainTarget, validData, validTarget, testData, testTarget = p1.loadData("notMNIST.npz")

    # ReLU activation function, cross-entropy cost function, softmax output layer
    # NN with 1 hidden layer and 1000 units

    d = trainData.shape[1]  # 28*28 = 784
    N = len(trainData)  # 15000

    batchSize = 500

    XNN = tf.placeholder(tf.float64, [None, d])
    YNN = tf.placeholder(tf.int32, [None, 1])

    iteration = 6000.

    iterPerEpoch = int(N / batchSize)  # 30
    epochs = int(np.ceil(iteration / float(iterPerEpoch)))  # 200

    numHiddenUnits = 1000
    numClasses = 10

    plt.close()

    with tf.Session() as sess:
        for m in range(1):
            # Training parameters
            seed = int(time.time()) % 1000
            tf.set_random_seed(seed)
            random.seed(seed)
            learningRate = exp(random.uniform(-7.5, -4.5))
            numLayers = random.randint(1, 5) #number of hidden layers
            numLayers += 1 #including the output layer
            numHiddenUnits = random.randint(100, 500)
            lda = exp(random.uniform(-9, -6))
            keepProbability = random.sample(set([0.5, 1.0]), 1)[0]
            if keepProbability==0.5:
                lda = 0.
            print("learning rate %f, numlayers %d, numHiddenUnits %d, weight decay %f, keepProbability %f" % (learningRate, numLayers, numHiddenUnits, lda, keepProbability))
            #Training starts here
            hiddenLayerInputs = [None for i in range(numLayers)]
            hiddenLayers = [None for i in range(numLayers)]
            hiddenDropoutLayers = [None for i in range(numLayers)]
            hiddenLayerInputs[0], WHidden, BHidden = layerBuildingBlock(XNN, numHiddenUnits)
            hiddenLayers[0] = tf.nn.relu(hiddenLayerInputs[0])
            hiddenDropoutLayers[0] = tf.nn.dropout(hiddenLayers[0], keepProbability)
            for h in range(1,numLayers-1):
                hiddenLayerInputs[h], WHidden, BHidden = layerBuildingBlock(hiddenDropoutLayers[h-1], numHiddenUnits)
                hiddenLayers[h] = tf.nn.relu(hiddenLayerInputs[h])
                hiddenDropoutLayers[h] = tf.nn.dropout(hiddenLayers[h], keepProbability)
            hiddenLayerInputs[numLayers-1], WOutput, BOutput = layerBuildingBlock(hiddenDropoutLayers[numLayers-2], numClasses)
            hiddenLayers[numLayers-1] = tf.nn.softmax(hiddenLayerInputs[numLayers-1])
            crossEntropyLoss = p1.calculateCrossEntropyLoss(hiddenLayers[numLayers-1], WOutput, YNN, numClasses, lda)
            optimizer = tf.train.AdamOptimizer(learningRate).minimize(crossEntropyLoss)
            classificationError = p1.calculateClassificationError(hiddenLayers[numLayers-1], YNN)

            trainingLoss = [None for _ in range(epochs)]
            validationLoss = [None for _ in range(epochs)]
            testLoss = [None for _ in range(epochs)]
            trainingClassificationError = [None for _ in range(epochs)]
            validationClassificationError = [None for _ in range(epochs)]
            testClassificationError = [None for _ in range(epochs)]
            tf.global_variables_initializer().run()
            for epoch in range(epochs):
                for i in range(iterPerEpoch):
                    XBatch = trainData[i * batchSize:(i + 1) * batchSize]
                    YBatch = trainTarget[i * batchSize:(i + 1) * batchSize]
                    feed = {XNN: XBatch, YNN: YBatch}
                    _, trainingLoss[epoch], trainingClassificationError[epoch] = sess.run(
                        [optimizer, crossEntropyLoss, classificationError], feed_dict=feed)

                feed = {XNN: validData, YNN: validTarget}
                validationLoss[epoch], validationClassificationError[epoch] = sess.run(
                    [crossEntropyLoss, classificationError], feed_dict=feed)
                feed = {XNN: testData, YNN: testTarget}
                testLoss[epoch], testClassificationError[epoch] = sess.run([crossEntropyLoss, classificationError],
                                                                           feed_dict=feed)
            f = open("model%d_training" % m, "wb")
            pickle.dump(trainingClassificationError, f)
            f.close()
            f = open("model%d_validation" % m, "wb")
            pickle.dump(validationClassificationError, f)
            f.close()
            f = open("model%d_test" % m, "wb")
            pickle.dump(testClassificationError, f)
            f.close()

            fig = plt.figure(1 + m)
            plt.plot(range(epochs), validationClassificationError, c='m', label='Validation')
            plt.plot(range(epochs), testClassificationError, c='c', label='Test')
            plt.legend()
            plt.title("Classification Error vs no. of epochs for learning rate: %f" % learningRate)
            plt.xlabel("Number of epochs")
            plt.ylabel("Classification Error (%)")
            fig.savefig("part4_Model%d.png"%m)




