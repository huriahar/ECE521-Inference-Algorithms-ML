from __future__ import print_function
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def loadData (fileName):
    with np.load(fileName) as data:
        Data, Target = data["images"], data["labels"]
        np.random.seed(521)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data = Data[randIndx]/255.0
        Data = Data.reshape(-1,28*28)
        Target = Target[randIndx].reshape(-1, 1)
        trainData, trainTarget = Data[:15000], Target[:15000]
        validData, validTarget = Data[15000:16000], Target[15000:16000]
        testData, testTarget = Data[16000:], Target[16000:]
        return trainData, trainTarget, validData, validTarget, testData, testTarget

def layerBuildingBlock (XLMinus1, numHiddenUnits):
    # Initialize W - Use Xavier initialization for weight matrix
    dimLMinus1 = XLMinus1.get_shape().as_list()[1]
    xavierStdDev = np.sqrt(3.0/(dimLMinus1 + numHiddenUnits))
    # Alternate method
    # W = tf.Variable(shape=[dimLMinus1, numHiddenUnits], dtype=tf.float64, initializer=tf.contrib.layers.xavier_initializer(), name="weights")
    W = tf.Variable(tf.truncated_normal(shape=[dimLMinus1, numHiddenUnits], dtype=tf.float64, seed=521, stddev=xavierStdDev, name="weights"))
    b = tf.Variable(tf.zeros(shape=[numHiddenUnits], dtype=tf.float64, name="bias"))
    return tf.matmul(XLMinus1, W) + b, W, b

def calculateCrossEntropyLoss (logits, weights, y, numClasses, lambdaParam):
    labels = tf.squeeze(tf.one_hot(y, numClasses, dtype=tf.float64))
    loss_d = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
    loss_w = lambdaParam*tf.nn.l2_loss(weights)
    crossEntropyLoss = loss_d + loss_w
    return crossEntropyLoss

def calculateAccuracy (predictedValues, actualValues):
    correctPrediction = tf.equal(tf.squeeze(actualValues), tf.argmax(predictedValues, 1, output_type=tf.int32))
    accuracy = tf.reduce_mean(tf.cast(correctPrediction, tf.float64))
    return accuracy*100

def calculateClassificationError (predictedValues, actualValues):
    correctPrediction = tf.equal(tf.squeeze(actualValues), tf.argmax(predictedValues, 1, output_type=tf.int32))
    accuracy = tf.reduce_mean(tf.cast(correctPrediction, tf.float64))
    return (1.0 - accuracy)*100

if __name__ == '__main__':
    trainData, trainTarget, validData, validTarget, testData, testTarget = loadData("notMNIST.npz")

    # ReLU activation function, cross-entropy cost function, softmax output layer
    # NN with 2 hidden layers with 500 units each

    d = trainData.shape[1]                                  # 28*28 = 784
    N = len(trainData)                                      # 15000

    batchSize = 500

    XNN = tf.placeholder(tf.float64, [None, d])
    YNN = tf.placeholder(tf.int32, [None, 1])

    iteration = 6000.
    lda = 3e-4

    iterPerEpoch = int(N / batchSize)                       # 30
    epochs = int(np.ceil(iteration/float(iterPerEpoch)))    # 200

    learningRate = 0.005                                    # From 1.2

    numHiddenUnits = [500, 500]
    numClasses = 10

    trainingLoss = [None for _ in range(epochs)]
    validationLoss = [None for _ in range(epochs)]
    testLoss = [None for _ in range(epochs)]
    trainingClassificationError = [None for _ in range(epochs)]
    validationClassificationError = [None for _ in range(epochs)]
    testClassificationError = [None for _ in range(epochs)]

    plt.close()

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        with tf.variable_scope("hiddenLayer1"):
            firstHiddenLayerInput, WFirstHidden, BFirstHidden = layerBuildingBlock(XNN, numHiddenUnits[0])
            firstHiddenLayerOutput = tf.nn.relu(firstHiddenLayerInput)

        with tf.variable_scope("hiddenLayer2"):
            secondHiddenLayerInput, WSecondHidden, BSecondHidden = layerBuildingBlock(firstHiddenLayerOutput, numHiddenUnits[1])
            secondHiddenLayerOutput = tf.nn.relu(secondHiddenLayerInput)

        with tf.variable_scope("outputLayer"):
            outputLayerInput, WOutput, BOutput = layerBuildingBlock(secondHiddenLayerOutput, numClasses)
            outputLayerOutput = tf.nn.softmax(outputLayerInput)

        crossEntropyLoss = calculateCrossEntropyLoss(outputLayerInput, WOutput, YNN, numClasses, lda)
        optimizer = tf.train.AdamOptimizer(learningRate).minimize(crossEntropyLoss)
        classificationError = calculateClassificationError(outputLayerOutput, YNN)

        tf.global_variables_initializer().run()

        # Training
        for epoch in range(epochs):
            for i in range(iterPerEpoch):
                XBatch = trainData[i*batchSize:(i+1)*batchSize]
                YBatch = trainTarget[i*batchSize:(i+1)*batchSize]
                _ = sess.run(optimizer, feed_dict={XNN:XBatch, YNN:YBatch})

            trainingLoss[epoch], trainingClassificationError[epoch] = sess.run([crossEntropyLoss, classificationError], feed_dict={XNN:XBatch, YNN:YBatch})
            validationLoss[epoch], validationClassificationError[epoch] = sess.run([crossEntropyLoss, classificationError], feed_dict={XNN:validData, YNN:validTarget})
            testLoss[epoch], testClassificationError[epoch] = sess.run([crossEntropyLoss, classificationError], feed_dict={XNN:testData, YNN:testTarget})

        print("Last Validation cross-entropy loss:", validationLoss[-1])
        print("Min Validation cross-entropy loss:", min(validationLoss))
        print("Last Validation classification error:", validationClassificationError[-1])
        print("Min Validation classification error:", min(validationClassificationError))
        print('-'*100)
        print("Last Test cross-entropy loss:", testLoss[-1])
        print("Min Test cross-entropy loss:", min(testLoss))
        print("Last Test classification error:", testClassificationError[-1])
        print("Min Test classification error:", min(testClassificationError))

        colors = ['c', 'm']

        # Plot the training and validation classification errors
        fig = plt.figure(0)
        plt.plot(range(epochs), trainingClassificationError, c=colors[0], label='Training')
        plt.plot(range(epochs), validationClassificationError, c=colors[1], label='Validation')
        plt.legend()
        plt.title("Classification Error vs no. of epochs for 2 hidden layers")
        plt.xlabel("Number of epochs")
        plt.ylabel("Classification Error (%)")
        fig.savefig("part2_2ClError.png")

        # Plot the training and validation cross entropy loss
        fig = plt.figure(1)
        plt.plot(range(epochs), trainingLoss, c=colors[0], label='Training')
        plt.plot(range(epochs), validationLoss, c=colors[1], label='Validation')
        plt.legend()
        plt.title("Cross entropy loss vs no. of epochs for 2 hidden layers")
        plt.xlabel("Number of epochs")
        plt.ylabel("Cross entropy loss")
        fig.savefig("part2_2CELoss.png")
