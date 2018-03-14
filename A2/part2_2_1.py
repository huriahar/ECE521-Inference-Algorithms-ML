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

def calculateValidationCrossEntropyLoss(X, Y, w, b, numClass):
    logits = tf.matmul(X, w) + b
    labels = tf.one_hot(Y, numClass)
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    loss = tf.reduce_mean(loss)
    regularizer = tf.nn.l2_loss(w)
    loss = loss + lda * regularizer
    return loss

if __name__ == "__main__":
    trainData, trainTarget, validData, validTarget, testData, testTarget = loadData("notMNIST.npz")
    XValid = tf.placeholder(tf.float64, validData.shape)
    YValid = tf.placeholder(tf.int32, validTarget.shape)

    XTest = tf.placeholder(tf.float64, testData.shape)
    YTest = tf.placeholder(tf.int32, testTarget.shape)
    batchSize = 500
    d = 784
    iteration = 20000.
    learnRates = [0.005, 0.001, 0.0001]
    training_losses = [[] for i in range(len(learnRates))]
    valid_losses = [[] for i in range(len(learnRates))]
    training_accuracies = [[] for i in range(len(learnRates))]
    valid_accuracies = [[] for i in range(len(learnRates))]
    bestLearnRateIdx = 0
    min_lossValid = 1000000.0
    lda = 0.01
    #This is a multi-class problem
    numClass = 10
    N = len(trainData)
    iterPerEpoch = int(N / batchSize)
    epochs = int(np.ceil(iteration / float(iterPerEpoch)))
    XTrain = tf.placeholder(tf.float64, [batchSize, d])
    YTrain = tf.placeholder(tf.int32, [batchSize])
    for idx, learnRate in enumerate(learnRates):
        w = tf.Variable(tf.truncated_normal([d, numClass], stddev=0.5, seed=521, dtype=tf.float64), name="weights")
        b = tf.Variable(tf.zeros([numClass] ,dtype=tf.float64, name="biases"))
        logits = tf.matmul(XTrain, w) + b
        labels = tf.one_hot(YTrain, numClass)
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        loss = tf.reduce_mean(loss)
        #regularizer = tf.nn.l2_loss(w)
        regularizer = tf.reduce_sum(tf.square(w)) / 2.0
        loss = loss + lda * regularizer
        optimizer = tf.train.GradientDescentOptimizer(learnRate).minimize(loss)
        init = tf.global_variables_initializer()
        sess.run(init)
        for ep in range(epochs):
            for i in range(iterPerEpoch):
                XBatch = trainData[i * batchSize:(i + 1) * batchSize]
                YBatch = trainTarget[i * batchSize:(i + 1) * batchSize]
                feed = {XTrain: XBatch, YTrain: YBatch}
                _, L= sess.run([optimizer, loss], feed_dict=feed)
            # Training loss and accuracy calculations####
            # cross entropy loss at the end of this epoch for this learn rate
            training_losses[idx].append(L)
            # Classification accuracy at the end of this epoch for this learn rate
            XTrainALL = tf.placeholder(tf.float64, [N, d])
            YTrainALL = tf.placeholder(tf.int32, [N])
            logits = tf.matmul(XTrainALL, w) + b
            Pi = tf.nn.softmax(logits)
            value, classifications = tf.nn.top_k(Pi)
            Y_expanded = tf.expand_dims(YTrainALL, -1)
            l, p, c, y = sess.run([logits, Pi, classifications,Y_expanded], feed_dict={XTrainALL: trainData, YTrainALL: trainTarget})
            accuracy, update_op = tf.metrics.accuracy(labels=Y_expanded, predictions=classifications)
            tf.local_variables_initializer().run()
            sess.run([accuracy, update_op], feed_dict={XTrainALL: trainData, YTrainALL: trainTarget})
            A = sess.run(accuracy)
            training_accuracies[idx].append(A)
            # Validation loss and accuracy calculations####
            validation_loss = calculateValidationCrossEntropyLoss(XValid, YValid, w, b, numClass)
            logits = tf.matmul(XValid, w) + b
            Pi = tf.nn.softmax(logits)
            value, classifications = tf.nn.top_k(Pi)
            Y_expanded = tf.expand_dims(YValid, -1)
            validation_accuracy, validation_update_op = tf.metrics.accuracy(labels=Y_expanded, predictions=classifications)
            tf.local_variables_initializer().run()
            vL, vA = sess.run([validation_loss, validation_accuracy], feed_dict={XValid: validData, YValid: validTarget})
            valid_losses[idx].append(vL)
            valid_accuracies[idx].append(vA)
        if valid_losses[idx][-1] < min_lossValid:
            bestLearnRateIdx = idx
    plt.close('all')
    fig = plt.figure(1)
    plt.scatter(range(epochs), training_losses[bestLearnRateIdx], marker='.', c='r', label='training')
    plt.scatter(range(epochs), valid_losses[bestLearnRateIdx], marker='.', c='b', label='validation')
    plt.xlabel('the n-th epoch')
    plt.ylabel('cross-entropy loss')
    plt.legend()
    plt.title("cross-entropy loss vs number of epochs for learning rate of %f" % learnRates[bestLearnRateIdx])
    fig.savefig("part2_2_1_loss.png")
    fig = plt.figure(2)
    plt.scatter(range(epochs), training_accuracies[bestLearnRateIdx], marker='.', c='r', label='training')
    plt.scatter(range(epochs), valid_accuracies[bestLearnRateIdx], marker='.', c='b', label='validation')
    plt.xlabel('the n-th epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.title("accuracy vs number of epochs for learning rate of %f" % learnRates[bestLearnRateIdx])
    fig.savefig("part2_2_1_accuracy.png")
    print("Classification accuracy on training data: %f" % training_accuracies[bestLearnRateIdx][-1])
    print("Classification accuracy on validation data: %f" % valid_accuracies[bestLearnRateIdx][-1])


