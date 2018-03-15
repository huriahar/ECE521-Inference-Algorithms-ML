from __future__ import print_function
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

sess = tf.InteractiveSession()

def loadData(dataPath, targetPath):
    # task = 0 >>  select the name ID targets for face recognition task
    data = np.load(dataPath)/255.0
    data = np.reshape(data, [-1,32*32])
    target = np.load(targetPath)

    np.random.seed(45689)
    rnd_idx = np.arange(np.shape(data)[0])
    np.random.shuffle(rnd_idx)

    trBatch = int(0.8*len(rnd_idx))
    validBatch = int(0.1*len(rnd_idx))

    trainData, validData, testData = data[rnd_idx[1:trBatch],:], \
                                     data[rnd_idx[trBatch+1:trBatch+validBatch],:], \
                                     data[rnd_idx[trBatch+validBatch+1:-1],:]

    trainTarget, validTarget, testTarget = target[rnd_idx[1:trBatch],0], \
                                           target[rnd_idx[trBatch+1:trBatch+validBatch],0], \
                                           target[rnd_idx[trBatch+validBatch+1:-1],0]

    return trainData, validData, testData, trainTarget, validTarget, testTarget

def calculateCrossEntropyLoss(X, Y, w, b, lda, numClass):
    logits = tf.matmul(X, w) + b
    labels = tf.one_hot(Y, numClass)
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    loss = tf.reduce_mean(loss)
    regularizer = tf.nn.l2_loss(w)
    loss = loss + lda * regularizer
    return loss

if __name__ == "__main__":
    trainData, validData, testData, trainTarget, validTarget, testTarget = loadData('data.npy','target.npy')

    batchSize = 300
    d = trainData.shape[1]                      # 32*32 = 1024
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

    ldas = [0., 0.001, 0.01, 0.1, 1.]          # Lambda i.e. weight decay coefficient

    #This is a multi-class problem
    numClass = 6
    N = len(trainData)                                      # 747
    iterPerEpoch = int(np.ceil(float(N) / batchSize))       # 3
    epochs = int(np.ceil(iteration / float(iterPerEpoch)))  # 1667

    XTrain = tf.placeholder(tf.float64, [batchSize, d])
    YTrain = tf.placeholder(tf.int32, [batchSize])

    XValid = tf.placeholder(tf.float64, validData.shape)
    YValid = tf.placeholder(tf.int32, validTarget.shape)

    XTest = tf.placeholder(tf.float64, testData.shape)
    YTest = tf.placeholder(tf.int32, testTarget.shape)

    # Determine the best learnig rate using the training set and keeping lamda = 0
    for idx, learnRate in enumerate(learnRates):
        w = tf.Variable(tf.truncated_normal([d, numClass], stddev=0.5, dtype=tf.float64), name="weights")
        b = tf.Variable(tf.zeros([numClass] ,dtype=tf.float64, name="biases"))
        loss = calculateCrossEntropyLoss(XTrain, YTrain, w, b, ldas[0], numClass)
        optimizer = tf.train.AdamOptimizer(learnRate).minimize(loss)
        init = tf.global_variables_initializer()
        sess.run(init)
        for ep in range(epochs):
            for i in range(iterPerEpoch):
                if((i+1)*batchSize > N):
                    XBatch = trainData[-batchSize:]
                    YBatch = trainTarget[-batchSize:]
                else:
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
    print("Best learning rate is", learnRates[i], "with loss:", minLossTrain)

    for lda in ldas:
        # Get the weight and bias of the selected learning rate from its last epoch
        bestWeights = weights[bestLearningRateIdx][-1]
        bestBias = biases[bestLearningRateIdx][-1]
        validationLoss = calculateCrossEntropyLoss(XValid, YValid, bestWeights, bestBias, lda, numClass)
        logits = tf.matmul(XValid, weightEp) + biasEp
