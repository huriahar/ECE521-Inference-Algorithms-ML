from __future__ import print_function
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

NUMVALID = 5

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

# Returns true if the validationPartial list has values in ascending order
def validationIncreasing (validationPartial):
    result = all(validationPartial[i] <= validationPartial[i + 1] for i in range(len(validationPartial) - 1))
    return result

if __name__ == '__main__':
    trainData, trainTarget, validData, validTarget, testData, testTarget = loadData("notMNIST.npz")

    # ReLU activation function, cross-entropy cost function, softmax output layer
    # NN with 1 hidden layer and 1000 units

    d = trainData.shape[1]                                  # 28*28 = 784
    N = len(trainData)                                      # 15000

    batchSize = 500

    XNN = tf.placeholder(tf.float64, [None, d])
    YNN = tf.placeholder(tf.int32, [None, 1])

    iteration = 6000.
    lda = 3e-4

    iterPerEpoch = int(N / batchSize)                       # 30
    epochs = int(np.ceil(iteration/float(iterPerEpoch)))    # 200
    
    finalCELossEpochs = epochs
    CELossEarlyStopped = False

    finalClErrorEpochs = epochs
    ClErrorEarlyStopped = False

    learningRate = 0.005                                    # Chosen from part 1.2

    numHiddenUnits = 1000
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

        hiddenLayerInput, WHidden, BHidden = layerBuildingBlock(XNN, numHiddenUnits)
        hiddenLayerOutput = tf.nn.relu(hiddenLayerInput)

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
                feed = {XNN:XBatch, YNN:YBatch}
                _ = sess.run(optimizer, feed_dict=feed)

            trainingLoss[epoch], trainingClassificationError[epoch] = sess.run([crossEntropyLoss, classificationError], feed_dict={XNN:XBatch, YNN:YBatch})
            validationLoss[epoch], validationClassificationError[epoch] = sess.run([crossEntropyLoss, classificationError], feed_dict={XNN:validData, YNN:validTarget})
            testLoss[epoch], testClassificationError[epoch] = sess.run([crossEntropyLoss, classificationError], feed_dict={XNN:testData, YNN:testTarget})

            if epoch >= NUMVALID:
                # Check if validation error has been continuosly increasing for the last 5 epochs
                # If so, early stop
                if not CELossEarlyStopped and validationIncreasing(validationLoss[epoch-NUMVALID:epoch]):
                    # Only epochs up till finalCELossEpochs should be considered
                    finalCELossEpochs = epoch - NUMVALID
                    CELossEarlyStopped = True

                if not ClErrorEarlyStopped and validationIncreasing(validationClassificationError[epoch-NUMVALID:epoch]):
                    # Only epochs up till finalClErrorEpochs should be considered
                    finalClErrorEpochs = epoch - NUMVALID
                    ClErrorEarlyStopped = True

        print("Early Stopping Point for Cross Entropy Loss:", finalCELossEpochs)
        print("Early Stopping Point for Classification Error:", finalClErrorEpochs)

        print("Training Classification Error at stop point:", trainingClassificationError[finalClErrorEpochs])
        print("Validation Classification Error at stop point:", validationClassificationError[finalClErrorEpochs])
        print("Test Classification Error at stop point:", testClassificationError[finalClErrorEpochs])

        # Plot the training, validation and test cross entropy loss for best learning rate
        fig = plt.figure(0)
        plt.plot(range(epochs), trainingLoss, c='m', label='Training')
        plt.plot(range(epochs), validationLoss, c='c', label='Validation')
        plt.plot(range(epochs), testLoss, c='y', label='Test')
        plt.axvline(x=finalCELossEpochs, c='r', label='Early Stopping Point')
        plt.legend()
        plt.title("Cross Entropy Loss vs no. of epochs for learning rate: %f"%learningRate)
        plt.xlabel("Number of epochs")
        plt.ylabel("Cross Entropy Loss")
        fig.savefig("part1_3_CELoss.png")

        # Plot the training, validation and test classification error for best learning rate
        fig = plt.figure(1)
        plt.plot(range(epochs), trainingClassificationError, c='m', label='Training')
        plt.plot(range(epochs), validationClassificationError, c='c', label='Validation')
        plt.plot(range(epochs), testClassificationError, c='y', label='Test')
        plt.axvline(x=finalClErrorEpochs, c='r', label='Early Stopping Point')
        plt.legend()
        plt.title("Classification Error vs no. of epochs for learning rate: %f"%learningRate)
        plt.xlabel("Number of epochs")
        plt.ylabel("Classification Error (%)")
        fig.savefig("part1_3_ClassificationError.png")
