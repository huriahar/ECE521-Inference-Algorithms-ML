from __future__ import print_function
import numpy as np
import tensorflow as tf
import copy
import matplotlib.pyplot as plt

sess = tf.InteractiveSession()

def loadData(datafile):
    with np.load(datafile) as data:
        Data, Target = data["images"], data["labels"]
        np.random.seed(521)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data = Data[randIndx] / 255.
        Data = Data.reshape(-1, 28 * 28)
        Target = Target[randIndx]
        trainData, trainTarget = Data[:15000], Target[:15000]
        validData, validTarget = Data[15000:16000], Target[15000:16000]
        testData, testTarget = Data[16000:], Target[16000:]
        return trainData, trainTarget, validData, validTarget, testData, testTarget

def calculateCrossEntropyLoss(X, Y, w, b, lda, numClass):
    logits = tf.matmul(X, w) + b
    labels = tf.one_hot(Y, numClass)
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    loss = tf.reduce_mean(loss)
    regularizer = tf.nn.l2_loss(w)
    loss = loss + lda * regularizer
    return loss

def calculateClassificationAccuracy(X, Y, w, b):
    logits = tf.matmul(X, w) + b
    Pi = tf.nn.softmax(logits)
    value, classifications = tf.nn.top_k(Pi)
    YExpanded = tf.expand_dims(Y, -1)
    accuracy, updateOp = tf.metrics.accuracy(labels=YExpanded, predictions=classifications)
    return accuracy, updateOp

if __name__ == "__main__":
    trainData, trainTarget, validData, validTarget, testData, testTarget = loadData("notMNIST.npz")

    batchSize = 500
    d = trainData.shape[1]                                  # 28*28 = 784
    iteration = 5000.
    learnRates = [0.005, 0.001, 0.0001]

    minimumTrainingLoss = float('inf')
    bestLearningRateIdx = 0
    Loss = bestWeight = bestBias = None

    lda = 0.01                                              # Lambda i.e. weight decay coefficient

    #This is a multi-class problem
    numClass = 10
    N = len(trainData)                                      # 1500
    iterPerEpoch = int(N / batchSize)                       # 30
    epochs = int(np.ceil(iteration / float(iterPerEpoch)))  # 167

    XTrain = tf.placeholder(tf.float64, [batchSize, d])
    YTrain = tf.placeholder(tf.int32, [batchSize])

    XValid = tf.placeholder(tf.float64, validData.shape)
    YValid = tf.placeholder(tf.int32, validTarget.shape)

    XTest = tf.placeholder(tf.float64, testData.shape)
    YTest = tf.placeholder(tf.int32, testTarget.shape)

    for idx, learnRate in enumerate(learnRates):
        w = tf.Variable(tf.truncated_normal([d, numClass], stddev=0.5, dtype=tf.float64), name="weights")
        b = tf.Variable(tf.zeros([numClass], dtype=tf.float64, name="biases"))
        trainingLoss = [None for ep in range(epochs)]
        Weights = [None for ep in range(epochs)]
        Biases = [None for ep in range(epochs)]

        loss = calculateCrossEntropyLoss(XTrain, YTrain, w, b, lda, numClass)
        optimizer = tf.train.AdamOptimizer(learnRate).minimize(loss)
        init = tf.global_variables_initializer()
        sess.run(init)
        for ep in range(epochs):
            for i in range(iterPerEpoch):
                XBatch = trainData[i * batchSize:(i + 1) * batchSize]
                YBatch = trainTarget[i * batchSize:(i + 1) * batchSize]
                _ = sess.run(optimizer, feed_dict={XTrain: XBatch, YTrain: YBatch})
            trainingLoss[ep], Weights[ep], Biases[ep] = sess.run([loss, w, b], feed_dict={XTrain: XBatch, YTrain: YBatch})

        print("Training cross-entropy loss with learning rate", learnRate, "is:", trainingLoss[-1])
        # Check if this is the least loss seen so far. Best learning rate selected through lest training loss
        if (trainingLoss[-1] < minimumTrainingLoss):
            minimumTrainingLoss = trainingLoss[-1]
            bestLearningRateIdx = idx
            Loss = copy.deepcopy(trainingLoss)
            bestWeight = copy.deepcopy(Weights)
            bestBias = copy.deepcopy(Biases)
    print("Best learning rate", learnRates[bestLearningRateIdx], "loss:", Loss[-1])
    # Use above learning rate to plot the training and validation curves for both cross-entropy loss
    # and classification accuracy vs the number of epochs

    trainingAccuracy = [None for ep in range(epochs)]
    validationLoss = [None for ep in range(epochs)]
    validationAccuracy = [None for ep in range(epochs)]

    XTrainAll = tf.placeholder(tf.float64, [N, d], name="X")
    YTrainAll = tf.placeholder(tf.int32, [N], name = "Y")

    for ep in range(epochs):
        weightEp = bestWeight[ep]
        biasEp = bestBias[ep]
        validLoss = calculateCrossEntropyLoss(XValid, YValid, weightEp, biasEp, lda, numClass)

        logits = tf.matmul(XTrainAll, weightEp) + biasEp
        Pi = tf.nn.softmax(logits)
        value, classifications = tf.nn.top_k(Pi)
        YExpanded = tf.expand_dims(YTrainAll, -1)
        accuracy, updateOp = tf.metrics.accuracy(labels=YExpanded, predictions=classifications)
        tf.local_variables_initializer().run()
        _, updateOpTrain = sess.run([accuracy, updateOp], feed_dict={XTrainAll:trainData, YTrainAll:trainTarget})
        trainingAccuracy[ep] = sess.run(accuracy)

        logits = tf.matmul(XValid, weightEp) + biasEp
        Pi = tf.nn.softmax(logits)
        value, classifications = tf.nn.top_k(Pi)
        YExpanded = tf.expand_dims(YValid, -1)
        validAccuracy, validUpdateOp = tf.metrics.accuracy(labels=YExpanded, predictions=classifications)
        tf.local_variables_initializer().run()
        _, updateOpValid = sess.run([validAccuracy, validUpdateOp], feed_dict={XValid:validData, YValid:validTarget})
        validationAccuracy[ep] = sess.run(validAccuracy)        

        validationLoss[ep] = sess.run(validLoss, feed_dict={XValid:validData, YValid: validTarget})

    accuracy, updateOp = calculateClassificationAccuracy(XTest, YTest, bestWeight[-1], bestBias[-1])
    tf.local_variables_initializer().run()
    _, updateOp = sess.run([accuracy, updateOp], feed_dict={XTest:testData, YTest:testTarget})
    testAccuracy = sess.run(accuracy)
    print("Best test Accuracy:", testAccuracy)

    plt.close('all')
    fig = plt.figure(1)
    plt.plot(range(epochs), Loss, c='r', label='training')
    plt.plot(range(epochs), validationLoss, c='b', label='validation')
    plt.xlabel('the n-th epoch')
    plt.ylabel('cross-entropy loss')
    plt.legend()
    plt.title("cross-entropy loss vs number of epochs for learning rate %f" % learnRates[bestLearningRateIdx])
    fig.savefig("part2_2_1_loss.png")
    fig = plt.figure(2)
    plt.plot(range(epochs), trainingAccuracy, c='r', label='training')
    plt.plot(range(epochs), validationAccuracy, c='b', label='validation')
    plt.xlabel('the n-th epoch')
    plt.ylabel('Classification accuracy')
    plt.legend()
    plt.title("Classification accuracy vs number of epochs for learning rate %f" % learnRates[bestLearningRateIdx])
    fig.savefig("part2_2_1_accuracy.png")
    print("Classification accuracy on training data: %f" % trainingAccuracy[-1])
    print("Classification accuracy on validation data: %f" % validationAccuracy[-1])