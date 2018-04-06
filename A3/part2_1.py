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
    # NN with 1 hidden layer and {100, 500, 1000} units

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

    numHiddenUnits = [100, 500, 1000]
    numClasses = 10

    minimumValidationCELoss = float('inf')
    bestHiddenUnitsIdx = 0

    trainingLosses = [[None for _ in range(epochs)] for _ in range(len(numHiddenUnits))]
    validationLosses = [[None for _ in range(epochs)] for _ in range(len(numHiddenUnits))]
    testLosses = [[None for _ in range(epochs)] for _ in range(len(numHiddenUnits))]
    trainingClassificationErrors = [[None for _ in range(epochs)] for _ in range(len(numHiddenUnits))]
    validationClassificationErrors = [[None for _ in range(epochs)] for _ in range(len(numHiddenUnits))]
    testClassificationErrors = [[None for _ in range(epochs)] for _ in range(len(numHiddenUnits))]

    plt.close()

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        for idx, hiddenUnits in enumerate(numHiddenUnits):

            with tf.variable_scope("hiddenLayer"):
                hiddenLayerInput, WHidden, BHidden = layerBuildingBlock(XNN, hiddenUnits)
                hiddenLayerOutput = tf.nn.relu(hiddenLayerInput)

            with tf.variable_scope("outputLayer"):
                outputLayerInput, WOutput, BOutput = layerBuildingBlock(hiddenLayerOutput, numClasses)
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

                trainingLosses[idx][epoch], trainingClassificationErrors[idx][epoch] = sess.run([crossEntropyLoss, classificationError], feed_dict={XNN:XBatch, YNN:YBatch})
                validationLosses[idx][epoch], validationClassificationErrors[idx][epoch] = sess.run([crossEntropyLoss, classificationError], feed_dict={XNN:validData, YNN:validTarget})
                testLosses[idx][epoch], testClassificationErrors[idx][epoch] = sess.run([crossEntropyLoss, classificationError], feed_dict={XNN:testData, YNN:testTarget})

            # Check if this is the least validation error seen so far. Best number of hidden units selected through least validation error
            if (min(validationLosses[idx]) < minimumValidationCELoss):
                minimumValidationCELoss = min(validationLosses[idx])
                bestHiddenUnitsIdx = idx

        for idx in range(len(numHiddenUnits)):
            print("Last validation cross entropy loss for hidden units", numHiddenUnits[idx], ":", validationLosses[idx][-1])
            print("Best (Minimum) validation cross entropy loss for hidden units", numHiddenUnits[idx], ":", min(validationLosses[idx]))
            print('-'*100)
        
        bestHiddenUnits = numHiddenUnits[bestHiddenUnitsIdx]
        print("Best number of hidden units:", bestHiddenUnits)
        print("Last Test Classification Error using best number of hidden units", bestHiddenUnits, ":", testClassificationErrors[bestHiddenUnitsIdx][-1])
        print("Best (Min) Test Classification Error using best number of hidden units", bestHiddenUnits, ":", min(testClassificationErrors[bestHiddenUnitsIdx]))

        colors = ['y', 'c', 'm']

        # Plot the validation Cross entropy Losses for all hidden units
        fig = plt.figure(0)
        for idx in range(len(numHiddenUnits)):
            plt.plot(range(epochs), validationLosses[idx], c=colors[idx], label='Hidden Units = %d'%numHiddenUnits[idx])
        plt.legend()
        plt.title("Validation cross entropy loss vs no. of epochs for different hidden units")
        plt.xlabel("Number of epochs")
        plt.ylabel("Cross Entropy Loss")
        fig.savefig("part2_1_ValidationLoss_HiddenUnits.png")

        # Plot the validation classification errors for all hidden units
        fig = plt.figure(1)
        for idx in range(len(numHiddenUnits)):
            plt.plot(range(epochs), validationClassificationErrors[idx], c=colors[idx], label='Hidden Units = %d'%numHiddenUnits[idx])
        plt.legend()
        plt.title("Validation classification error vs no. of epochs for different hidden units")
        plt.xlabel("Number of epochs")
        plt.ylabel("Classification Error")
        fig.savefig("part2_1_ValidationClErrror_HiddenUnits.png")

