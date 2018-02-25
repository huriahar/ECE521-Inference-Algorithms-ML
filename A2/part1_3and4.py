from __future__ import print_function
import numpy as np
import tensorflow as tf

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

if __name__ == "__main__":
    trainData, trainTarget, validData, validTarget, testData, testTarget = loadData("notMNIST.npz")
    XValid = tf.placeholder(tf.float64, validData.shape)
    YValid = tf.placeholder(tf.float64, validTarget.shape)

    XTest = tf.placeholder(tf.float64, testData.shape)
    YTest = tf.placeholder(tf.float64, testTarget.shape)

    # part1.1
    batchSize = 500
    d = 784
    iteration = 20000.
    # chosen from part1.1
    learnRate = 0.005
    ldas = [0., 0.001, 0.1, 1]
    best_ldas = ldas[0]
    min_lossValid = 1000000.0
    N = len(trainData)
    iterPerEpoch = int(N / batchSize)
    epochs = int(np.ceil(iteration / float(iterPerEpoch)))
    XTrain = tf.placeholder(tf.float64, [batchSize, d])
    YTrain = tf.placeholder(tf.float64, [batchSize, 1])
    for lda in ldas:
        w = tf.Variable(tf.truncated_normal([d, 1], dtype=tf.float64), name="weights")
        b = tf.Variable(0.0, dtype=tf.float64, name="biases")
        YHead = tf.matmul(XTrain, w) + b
        loss = tf.reduce_sum(tf.squared_difference(YHead, YTrain))
        loss = tf.divide(loss, tf.to_double(2 * N))
        regularizer = tf.nn.l2_loss(w)
        loss = loss + lda * regularizer
        init = tf.global_variables_initializer()
        sess.run(init)
        optimizer = tf.train.GradientDescentOptimizer(learnRate).minimize(loss)
        L = [None for ep in range(epochs)]
        for ep in range(epochs):
            for i in range(iterPerEpoch):
                XBatch = trainData[i * batchSize:(i + 1) * batchSize]
                YBatch = trainTarget[i * batchSize:(i + 1) * batchSize]
                feed = {XTrain: XBatch, YTrain: YBatch}
                _, L[ep] = sess.run([optimizer, loss], feed_dict=feed)
        YHeadValid = tf.matmul(XValid, w) + b
        lossValid = tf.reduce_sum(tf.squared_difference(YHeadValid, YValid))
        lossValid = tf.divide(lossValid, tf.to_double(2 * N))
        cond = tf.less(YHeadValid, tf.zeros(tf.shape(YHeadValid), dtype=tf.float64))
        neg1 = tf.constant([-1.], dtype=tf.float64)
        I = tf.ones(tf.shape(YHeadValid), dtype=tf.float64)
        neg1 = I*neg1
        YHeadClassifiedValid = tf.where(cond, neg1, tf.ones(tf.shape(YHeadValid), dtype=tf.float64))
        cond = tf.equal(YHeadClassifiedValid, YValid)
        accuracy = tf.where(cond, tf.ones(tf.shape(YHeadValid)), tf.zeros(tf.shape(YHeadValid)))
        accuracy = tf.to_double(tf.reduce_sum(accuracy)) / tf.to_double(tf.size(accuracy))
        accuracy , lossValid = sess.run([accuracy, lossValid], feed_dict={XValid:validData, YValid:validTarget})
        print("With lamba=%f, MSE in validation set is %f" % (lda, lossValid))
        print("With lamba=%f, classification accuracy is %f" % (lda, accuracy))
        if (lossValid < min_lossValid):
            best_ldas = lda
            min_lossValid = lossValid

        #####################

        # print(sess.run(YTrain,{YTrain:trainTarget}))

