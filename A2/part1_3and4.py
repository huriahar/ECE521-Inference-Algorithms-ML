from __future__ import print_function
import numpy as np
import tensorflow as tf
import time

sess = tf.InteractiveSession()

def loadData(fileName):
    with np.load(fileName) as data:
        Data, Target = data["images"], data["labels"]
        posClass = 2
        negClass = 9
        dataIndx = (Target == posClass) + (Target == negClass)
        Data = Data[dataIndx] / 255.0
        Data = Data.reshape(-1, 28 * 28)
        Target = Target[dataIndx].reshape(-1, 1)
        Target[Target == posClass] = 1
        Target[Target == negClass] = 0
        np.random.seed(521)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data, Target = Data[randIndx], Target[randIndx]
        trainData, trainTarget = Data[:3500], Target[:3500]
        validData, validTarget = Data[3500:3600], Target[3500:3600]
        testData, testTarget = Data[3600:], Target[3600:]
        return trainData, trainTarget, validData, validTarget, testData, testTarget

def calculateMSELoss(X, Y, w, b, lda):
    YHead = tf.matmul(X, w) + b
    loss = tf.reduce_sum(tf.squared_difference(YHead, tf.cast(Y, tf.float64)))
    N = X.get_shape().as_list()[0]
    loss = tf.divide(loss, tf.to_double(2*N))
    regularizer = tf.nn.l2_loss(w)
    loss = loss + lda*regularizer
    return loss

def calculateAccuracy(X, Y, w, b):
    YHead = tf.matmul(X, w) + b
    # Creates a vector with each entry being either True/False
    # Entry is True when value in YHead < 0.5
    lessThanHalf = tf.less(YHead, tf.constant(0.5, shape=YHead.shape, dtype=tf.float64))
    # Classification value is 0 if True else 1
    YHeadClassified = tf.where(lessThanHalf, tf.zeros(tf.shape(YHead), dtype=tf.float64), tf.ones(tf.shape(YHead), dtype=tf.float64))
    accuracy, updateOp = tf.metrics.accuracy(labels=Y, predictions=YHeadClassified)
    return accuracy, updateOp

if __name__ == "__main__":
    trainData, trainTarget, validData, validTarget, testData, testTarget = loadData("notMNIST.npz")

    # part1.3
    batchSize = 500
    d = trainData.shape[1]      # 28*28 pixels/image = 784
    iteration = 20000.
    # chosen from part1.1
    learnRate = 0.005
    ldas = [0., 0.001, 0.1, 1.]
    bestLda = ldas[0]
    maxValidAccuracy = 0
    N = len(trainData)
    iterPerEpoch = int(N / batchSize)                       # 7
    epochs = int(np.ceil(iteration / float(iterPerEpoch)))  # 2858

    bestWeight = bestBias = None

    XTrain = tf.placeholder(tf.float64, [batchSize, d])
    YTrain = tf.placeholder(tf.int32, [batchSize, 1])

    XValid = tf.placeholder(tf.float64, validData.shape)
    YValid = tf.placeholder(tf.int32, validTarget.shape)

    XTest = tf.placeholder(tf.float64, testData.shape)
    YTest = tf.placeholder(tf.int32, testTarget.shape)
    
    for lda in ldas:
        w = tf.Variable(tf.truncated_normal([d, 1], stddev=0.5, seed=521, dtype=tf.float64), name="weights")
        b = tf.Variable(0.0, dtype=tf.float64, name="biases")
        loss = calculateMSELoss(XTrain, YTrain, w, b, lda)
        init = tf.global_variables_initializer()
        sess.run(init)
        optimizer = tf.train.GradientDescentOptimizer(learnRate).minimize(loss)
        L = [None for ep in range(epochs)]
        #record training time
        start = time.time()
        for ep in range(epochs):
            for i in range(iterPerEpoch):
                XBatch = trainData[i * batchSize:(i + 1) * batchSize]
                YBatch = trainTarget[i * batchSize:(i + 1) * batchSize]
                feed = {XTrain: XBatch, YTrain: YBatch}
                _, L[ep] = sess.run([optimizer, loss], feed_dict=feed)
        end = time.time()
        print("Lda:", lda, "training mse:", L[-1])
        validLoss = calculateMSELoss(XValid, YValid, w, b, lda)
        accuracy, updateOp = calculateAccuracy(XValid, YValid, w, b)
        tf.local_variables_initializer().run()
        validLoss, _, updateOp = sess.run([validLoss, accuracy, updateOp], feed_dict={XValid:validData, YValid:validTarget})
        accuracy = sess.run(accuracy)
        print("With lambda=%f, MSE in validation set: %f, classification accuracy in validation set: %f, computation time: %f seconds " % (lda, validLoss, accuracy, end-start))
        if (accuracy > maxValidAccuracy):
            bestLda = lda
            maxValidAccuracy = accuracy
            bestWeight = w
            bestBias = b

    print("Best lambda based on highest validation accuracy:", bestLda)

    # Test Accuracy
    testAccuracy, testUpdateOp = calculateAccuracy(XTest, YTest, bestWeight, bestBias)
    tf.local_variables_initializer().run()
    _, testUpdateOp = sess.run([testAccuracy, testUpdateOp], feed_dict={XTest:testData, YTest:testTarget})
    testAccuracy = sess.run(testAccuracy)
    print("Test Accuracy with the selected weight decay coefficient: %f" % testAccuracy)

    #####################
    #part1.4
    #normal equation:
    #wLS = (XT*X)-1 * XT * Y
    XTrain = tf.placeholder(tf.float64, [len(trainData), d])
    shape = XTrain.get_shape().as_list()
    shape[-1] = 1
    one = tf.ones(shape, dtype=tf.float64)
    XExtended = tf.concat([one, XTrain], 1)
    YTrain = tf.placeholder(tf.float64, [len(trainData), 1])
    XT = tf.transpose(XExtended)
    wLS = tf.matrix_inverse(tf.matmul(XT, XExtended))
    wLS = tf.matmul(wLS, XT)
    wLS = tf.matmul(wLS, YTrain)
    start = time.time()
    sess.run(wLS, feed_dict={XTrain:trainData, YTrain:trainTarget})
    end = time.time()
    
    shape = XValid.get_shape().as_list()
    shape[-1] = 1
    one = tf.ones(shape, dtype=tf.float64)
    XValidExtended = tf.concat([one, XValid], 1)
    YValid = tf.placeholder(tf.int32, [len(validData), 1])
    YHeadNormalEq = tf.matmul(XValidExtended, wLS)
    lossNormalEq = tf.reduce_sum(tf.squared_difference(YHeadNormalEq, tf.cast(YValid, tf.float64)))
    NValid = len(validData)
    lossNormalEq = tf.divide(lossNormalEq, tf.to_double(2 * NValid))
    cond = tf.less(YHeadNormalEq, tf.constant(0.5, shape=YHeadNormalEq.shape, dtype=tf.float64))
    YHeadNormalEqClassifiedValid = tf.where(cond, tf.zeros(tf.shape(YHeadNormalEq), dtype=tf.float64), tf.ones(tf.shape(YHeadNormalEq), dtype=tf.float64))
    cond = tf.equal(YHeadNormalEqClassifiedValid, tf.cast(YValid, tf.float64))
    accuracyNormalEq = tf.where(cond, tf.ones(tf.shape(YHeadNormalEq)), tf.zeros(tf.shape(YHeadNormalEq)))
    accuracyNormalEq = tf.to_double(tf.reduce_sum(accuracyNormalEq)) / tf.to_double(tf.size(accuracyNormalEq))
    accuracyNormalEq, lossNormalEq = sess.run([accuracyNormalEq, lossNormalEq], feed_dict={XValid:validData, YValid:validTarget, XTrain:trainData, YTrain:trainTarget})
    print("Using the normal equation, MSE in validation set: %f, classification accuracy in validation set: %f, computation time: %f seconds " % (lossNormalEq, accuracyNormalEq, end-start))
