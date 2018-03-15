from __future__ import print_function
import numpy as np
import tensorflow as tf
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

if __name__ == "__main__":
    trainData, trainTarget, validData, validTarget, testData, testTarget = loadData("notMNIST.npz")

    batchSize = 500
    d = trainData.shape[1]                      # 28*28 = 784
    iteration = 5000.
    learnRates = [0.005, 0.001, 0.0001]
    # These values are saved for each epoch for each learning rate
    trainingLosses = [[] for i in range(len(learnRates))]
    weights = [[] for i in range(len(learnRates))]
    biases = [[] for i in range(len(learnRates))]

    # These are calculated for each epoch for the selected best learning rate
    validationLosses = []
    trainingAccuracies = []
    validationAccuracies = []

    lda = 0.01          # Lambda i.e. weight decay coefficient

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
        b = tf.Variable(tf.zeros([numClass] ,dtype=tf.float64, name="biases"))
        loss = calculateCrossEntropyLoss(XTrain, YTrain, w, b, lda, numClass)
        optimizer = tf.train.AdamOptimizer(learnRate).minimize(loss)
        init = tf.global_variables_initializer()
        sess.run(init)
        for ep in range(epochs):
            for i in range(iterPerEpoch):
                XBatch = trainData[i * batchSize:(i + 1) * batchSize]
                YBatch = trainTarget[i * batchSize:(i + 1) * batchSize]
                feed = {XTrain: XBatch, YTrain: YBatch}
                _, L, wEp, bEp = sess.run([optimizer, loss, w, b], feed_dict=feed)
            # Save training losses for each epoch all traiing rates
            trainingLosses[idx].append(L)
            weights[idx].append(wEp)
            biases[idx].append(bEp)

    # Determine the best learning rate
    minLossTrain = float('inf')
    bestLearningRateIdx = 0
    for i in range(len(trainingLosses)):
        if trainingLosses[i][-1] < minLossTrain:
            minLossTrain = trainingLosses[i][-1]
            bestLearningRateIdx = i
    print("Best learning rate is", learnRates[i])

    # Use above learning rate to plot the training and validation curves for both cross-entropy loss
    # and classification accuracy vs the number of epochs
    XTrainALL = tf.placeholder(tf.float64, [N, d])
    YTrainALL = tf.placeholder(tf.int32, [N])
    
    for ep in range(epochs):
        weightEp = weights[bestLearningRateIdx][ep]
        biasEp = biases[bestLearningRateIdx][ep]
        logits = tf.matmul(XTrainALL, weightEp) + biasEp
        Pi = tf.nn.softmax(logits)
        value, classifications = tf.nn.top_k(Pi)
        Y_expanded = tf.expand_dims(YTrainALL, -1)
        l, p, c, y = sess.run([logits, Pi, classifications, Y_expanded], feed_dict={XTrainALL: trainData, YTrainALL: trainTarget})
        accuracy, update_op = tf.metrics.accuracy(labels=Y_expanded, predictions=classifications)
        tf.local_variables_initializer().run()
        sess.run([accuracy, update_op], feed_dict={XTrainALL: trainData, YTrainALL: trainTarget})
        A = sess.run(accuracy)
        trainingAccuracies.append(A)
        # Validation loss and accuracy calculations
        validationLoss = calculateCrossEntropyLoss(XValid, YValid, weightEp, biasEp, lda, numClass)
        logits = tf.matmul(XValid, weightEp) + biasEp
        Pi = tf.nn.softmax(logits)
        value, classifications = tf.nn.top_k(Pi)
        Y_expanded = tf.expand_dims(YValid, -1)
        validationAccuracy, validationUpdateOp = tf.metrics.accuracy(labels=Y_expanded, predictions=classifications)
        tf.local_variables_initializer().run()
        vL, _, _u = sess.run([validationLoss, validationAccuracy, validationUpdateOp], feed_dict={XValid: validData, YValid: validTarget})
        vA = sess.run(validationAccuracy)
        validationLosses.append(vL)
        validationAccuracies.append(vA)

    plt.close('all')
    fig = plt.figure(1)
    plt.scatter(range(epochs), trainingLosses[bestLearningRateIdx], marker='.', c='r', label='training')
    plt.scatter(range(epochs), validationLosses, marker='.', c='b', label='validation')
    plt.xlabel('the n-th epoch')
    plt.ylabel('cross-entropy loss')
    plt.legend()
    plt.title("cross-entropy loss vs number of epochs for learning rate of %f" % learnRates[bestLearningRateIdx])
    fig.savefig("part2_2_1_loss.png")
    fig = plt.figure(2)
    plt.scatter(range(epochs), trainingAccuracies, marker='.', c='r', label='training')
    plt.scatter(range(epochs), validationAccuracies, marker='.', c='b', label='validation')
    plt.xlabel('the n-th epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.title("accuracy vs number of epochs for learning rate of %f" % learnRates[bestLearningRateIdx])
    fig.savefig("part2_2_1_accuracy.png")
    print("Classification accuracy on training data: %f" % trainingAccuracies[-1])
    print("Classification accuracy on validation data: %f" % validationAccuracies[-1])
