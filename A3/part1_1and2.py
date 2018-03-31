from __future__ import print_function
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time, copy

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
    loss_d = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits))
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
    # NN with 1 hidden layer and 1000 units

    d = trainData.shape[1]  # 28*28 = 784
    N = len(trainData)      # 15000

    batchSize = 500

    XNN = tf.placeholder(tf.float64, [None, d])
    YNN = tf.placeholder(tf.int32, [None, 1])

    iteration = 6000.
    lda = 3e-4

    iterPerEpoch = int(N / batchSize)                       # 30
    epochs = int(np.ceil(iteration/float(iterPerEpoch)))    # 200

    learningRates = [0.0001, 0.001, 0.005, 0.010, 0.1]

    numHiddenUnits = 1000
    numClasses = 10

    minimumTrainingLoss = float('inf')
    bestLearningRateIdx = 0

    trainingLosses = [[None for _ in range(epochs)] for _ in range(len(learningRates))]
    validationLosses = [[None for _ in range(epochs)] for _ in range(len(learningRates))]
    testLosses = [[None for _ in range(epochs)] for _ in range(len(learningRates))]
    trainingClassificationErrors = [[None for _ in range(epochs)] for _ in range(len(learningRates))]
    validationClassificationErrors = [[None for _ in range(epochs)] for _ in range(len(learningRates))]
    testClassificationErrors = [[None for _ in range(epochs)] for _ in range(len(learningRates))]

    #saver = tf.train.Saver()

    plt.close()

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        for idx, lr in enumerate(learningRates):

            hiddenLayerInput, WHidden, BHidden = layerBuildingBlock(XNN, numHiddenUnits)
            hiddenLayerOutput = tf.nn.relu(hiddenLayerInput)

            outputLayerInput, WOutput, BOutput = layerBuildingBlock(hiddenLayerOutput, numClasses)
            outputLayerOutput = tf.nn.softmax(outputLayerInput)

            crossEntropyLoss = calculateCrossEntropyLoss(outputLayerInput, WOutput, YNN, numClasses, lda)
            optimizer = tf.train.AdamOptimizer(lr).minimize(crossEntropyLoss)
            classificationError = calculateClassificationError(outputLayerOutput, YNN)

            tf.global_variables_initializer().run()

            start = time.time()
            # Training
            for epoch in range(epochs):
                for i in range(iterPerEpoch):
                    XBatch = trainData[i*batchSize:(i+1)*batchSize]
                    YBatch = trainTarget[i*batchSize:(i+1)*batchSize]
                    feed = {XNN:XBatch, YNN:YBatch}
                    _ = sess.run(optimizer, feed_dict=feed)

                trainingLosses[idx][epoch], trainingClassificationErrors[idx][epoch] = sess.run([crossEntropyLoss, classificationError], feed_dict={XNN:XBatch, YNN:YBatch})
                validationLosses[idx][epoch], validationClassificationErrors[idx][epoch] = sess.run([crossEntropyLoss, classificationError], feed_dict={XNN:validData, YNN:validTarget})
                testLosses[idx][epoch], testClassificationErrors[idx][epoch] = sess.run([crossEntropyLoss, classificationError], feed_dict={XNN:testData, YNN:testTarget})
                # Save at epochs 100, 200, 300 and 400
                #if (epoch != 0 and epoch % (epochs / 5) == 0):
                #    saver.save(sess, '1_1_%f'%lr, global_step=epoch)

            end = time.time()
            print("Time taken for lr", lr, "is", end - start, "seconds")
            print("Minimum loss for learning rate", lr, "is:", trainingLosses[idx][-1])
            print("Minimum classification error for learning rate", lr, "is", trainingClassificationErrors[idx][-1])
            # Check if this is the least loss seen so far. Best learning rate selected through least training loss
            if (trainingLosses[idx][-1] < minimumTrainingLoss):
                minimumTrainingLoss = trainingLosses[idx][-1]
                bestLearningRateIdx = idx

        bestLearningRate = learningRates[bestLearningRateIdx]
        print("Best learning rate is", bestLearningRate)
        # Plot the Cross entropy Losses for all learning rates

        print("Validation Cross Entropy loss:", validationLosses[bestLearningRateIdx][-1])
        print("Validation Classification error:", validationClassificationErrors[bestLearningRateIdx][-1])
        print("Test Cross Entropy loss:", testLosses[bestLearningRateIdx][-1])
        print("Test Classification error:", testClassificationErrors[bestLearningRateIdx][-1])
        fig = plt.figure(0)
        colors = ['r', 'g', 'b', 'c', 'm']
        for idx in range(len(learningRates)):
            plt.plot(range(epochs), trainingLosses[idx], c=colors[idx], label='learning rate = %f'%learningRates[idx])
        plt.legend()
        plt.title("Training cross entropy loss vs no. of epochs for different learning rates")
        plt.xlabel("Number of epochs")
        plt.ylabel("Cross Entropy Loss")
        fig.savefig("part1_2_LearningRates.png")

        # Plot the training, validation and test Cross entropy loss for best learning rate
        fig = plt.figure(1)
        plt.plot(range(epochs), trainingLosses[bestLearningRateIdx], c='m', label='Training')
        plt.plot(range(epochs), validationLosses[bestLearningRateIdx], c='g', label='Validation')
        plt.plot(range(epochs), testLosses[bestLearningRateIdx], c='b', label='Test')
        plt.legend()
        plt.title("Cross entropy loss vs no. of epochs for best learning rate: %f"%bestLearningRate)
        plt.xlabel("Number of epochs")
        plt.ylabel("Cross Entropy Loss")
        fig.savefig("part1_2_CELoss.png")

        fig = plt.figure(2)
        plt.plot(range(epochs), trainingClassificationErrors[bestLearningRateIdx], c='m', label='Training')
        plt.plot(range(epochs), validationClassificationErrors[bestLearningRateIdx], c='g', label='Validation')
        plt.plot(range(epochs), testClassificationErrors[bestLearningRateIdx], c='b', label='Test')
        plt.legend()
        plt.title("Classification Error vs no. of epochs for best learning rate: %f"%bestLearningRate)
        plt.xlabel("Number of epochs")
        plt.ylabel("Classification Error (%)")
        fig.savefig("part1_2_ClassificationError.png")
