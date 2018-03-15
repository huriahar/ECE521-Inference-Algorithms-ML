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


if __name__ == "__main__":
    trainData, trainTarget, validData, validTarget, testData, testTarget = loadData("notMNIST.npz")
    XValid = tf.placeholder(tf.float64, validData.shape)
    YValid = tf.placeholder(tf.float64, validTarget.shape)

    XTest = tf.placeholder(tf.float64, testData.shape)
    YTest = tf.placeholder(tf.float64, testTarget.shape)

    # part1.2
    batchSizes = [500, 1500, 3500]
    d = validData.shape[1]      # 28*28 pixels/image = 784
    iteration = 20000.
    # chosen from part1.1
    learnRate = 0.005
    lda = 0.0
    N = len(trainData)
    for batchSize in batchSizes:
        start = time.time()
        iterPerEpoch = int(np.ceil(N / batchSize))                  # 7
        epochs = int(np.ceil(iteration / float(iterPerEpoch)))      # 2858
        XTrain = tf.placeholder(tf.float64, [batchSize, d])
        YTrain = tf.placeholder(tf.float64, [batchSize, 1])
        w = tf.Variable(tf.truncated_normal([d, 1], stddev=0.5, seed=521, dtype=tf.float64), name="weights")
        b = tf.Variable(0.0, dtype=tf.float64, name="biases")
        YHead = tf.matmul(XTrain, w) + b
        loss = tf.reduce_sum(tf.squared_difference(YHead, YTrain))
        loss = tf.divide(loss, tf.to_double(2 * N))
        regularizer = tf.nn.l2_loss(w)
        loss = loss + lda * regularizer
        init = tf.global_variables_initializer()
        sess.run(init)
        optimizer = tf.train.GradientDescentOptimizer(learnRate).minimize(loss)
        L = 0.
        for ep in range(epochs):
            for i in range(iterPerEpoch):
                if((i+1)*batchSize > N):
                    XBatch = trainData[-batchSize:]
                    YBatch = trainTarget[-batchSize:]
                else:
                    XBatch = trainData[i * batchSize:(i + 1) * batchSize]
                    YBatch = trainTarget[i * batchSize:(i + 1) * batchSize]
                feed = {XTrain: XBatch, YTrain: YBatch}
                _, L = sess.run([optimizer, loss], feed_dict=feed)
        end = time.time()

        print("Batch size %d: loss=%f , took %f seconds to execute" % (batchSize, L, end-start))


