from __future__ import print_function
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time

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
    W = tf.Variable(tf.truncated_normal(shape=[dimLMinus1, numHiddenUnits], dtype=tf.float64, stddev=xavierStdDev, name="weights"))
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
    return accuracy

if __name__ == '__main__':
    trainData, trainTarget, validData, validTarget, testData, testTarget = loadData("notMNIST.npz")

    # ReLU activation function, cross-entropy cost function, softmax output layer
    # NN with 1 hidden layer and 1000 units

    d = trainData.shape[1]  # 28*28 = 784
    N = len(trainData)      # 15000

    batchSize = 500

    # Training and test data for each mini batch in each epoch
    XTrain = tf.placeholder(tf.float64, [batchSize, d])
    YTrain = tf.placeholder(tf.int32, [batchSize, 1])

    XValid = tf.placeholder(tf.float64, validData.shape)
    YValid = tf.placeholder(tf.int32, validTarget.shape)

    XTest = tf.placeholder(tf.float64, testData.shape)
    YTest = tf.placeholder(tf.int32, testTarget.shape)

    iteration = 7500.
    lda = 3e-4

    iterPerEpoch = int(N / batchSize)                       # 30
    epochs = int(np.ceil(iteration/float(iterPerEpoch)))    # 250

    learningRates = [0.001, 0.004, 0.007, 0.010]#, 0.001, 0.0001]    [0.001, 0.003, 0.0004, 0.005, 0.007, 0.01]

    numHiddenUnits = 1000
    numClasses = 10

    #saver = tf.train.Saver()

    plt.close()

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        for idx, lr in enumerate(learningRates):

            trainingLoss = [None for _ in range(epochs)]
            trainingAccuracy = [None for _ in range(epochs)]

            hiddenLayerInput, WHidden, BHidden = layerBuildingBlock(XTrain, numHiddenUnits)
            hiddenLayerOutput = tf.nn.relu(hiddenLayerInput)

            outputLayerInput, WOutput, BOutput = layerBuildingBlock(hiddenLayerOutput, numClasses)
            outputLayerOutput = tf.nn.softmax(outputLayerInput)
            #YPredicted = tf.reshape(tf.argmax(outputLayerOutput, 1, output_type=tf.int32), [batchSize,1])

            crossEntropyLoss = calculateCrossEntropyLoss(outputLayerInput, WOutput, YTrain, numClasses, lda)
            optimizer = tf.train.AdamOptimizer(lr).minimize(crossEntropyLoss)
            accuracy = calculateAccuracy(outputLayerOutput, YTrain)

            tf.global_variables_initializer().run()
            start = time.time()
            # Training
            for epoch in range(epochs):
                for i in range(iterPerEpoch):
                    XBatch = trainData[i*batchSize:(i+1)*batchSize]
                    YBatch = trainTarget[i*batchSize:(i+1)*batchSize]
                    feed = {XTrain:XBatch, YTrain:YBatch}
                    _ = sess.run(optimizer, feed_dict=feed)

                trainingLoss[epoch], trainingAccuracy[epoch] = sess.run([crossEntropyLoss, accuracy], feed_dict=feed)
                # Save at epochs 100, 200, 300 and 400
                #if (epoch != 0 and epoch % (epochs / 5) == 0):
                #    saver.save(sess, '1_1_%f'%lr, global_step=epoch)

            end = time.time()
            print("Time taken for lr", lr, "is", end - start, "seconds")
            print("Minimum loss for learning rate", lr, "is:", trainingLoss[-1])
            fig = plt.figure(idx*2 + 1)
            plt.plot(range(epochs), trainingLoss, label='Training Loss')
            plt.xlabel('the n-th epoch')
            plt.ylabel('Cross Entropy Loss')
            plt.title("Training Cross Entropy Loss vs number of epochs for learning rate of %f" % lr)
            fig.savefig("part1_1_celoss_%d.png"%idx)
            fig = plt.figure(idx * 2 + 2)
            print("Maximum accuracy for learning rate", lr, "is:", trainingAccuracy[-1])
            plt.plot(range(epochs), trainingAccuracy, label='Training Accuracy')
            plt.xlabel('the n-th epoch')
            plt.ylabel('Training Accuracy')
            plt.title("Training Accuracy vs number of epochs for learning rate of %f" % lr)
            fig.savefig("part1_1_accuracy_%d.png"%idx)
