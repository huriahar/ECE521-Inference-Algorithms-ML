from __future__ import print_function
import numpy as np
import tensorflow as tf
import copy
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

def calculateClassificationAccuracy(X, Y, w, b):
    logits = tf.matmul(X, w) + b
    Pi = tf.nn.softmax(logits)
    value, classifications = tf.nn.top_k(Pi)
    YExpanded = tf.expand_dims(Y, -1)
    accuracy, updateOp = tf.metrics.accuracy(labels=YExpanded, predictions=classifications)
    return accuracy, updateOp

if __name__ == "__main__":
    trainData, validData, testData, trainTarget, validTarget, testTarget = loadData('data.npy','target.npy')

    batchSize = 300
    d = trainData.shape[1]                      # 32*32 = 1024
    iteration = 5000.
    learnRates = [0.005, 0.001, 0.0001]
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

    maxValidAccuracy = 0
    bestLearningRateIdx = 0
    bestLdaIdx = 0
    bestWeight = bestBias = Loss = None

    # Determine the best learnig rate weight decay coefficient
    # using the best accuracy for validation set
    for lrIdx, learningRate in enumerate(learnRates):
        for ldaIdx, lda in enumerate(ldas):
            w = tf.Variable(tf.truncated_normal([d, numClass], stddev=0.5, seed=521, dtype=tf.float64), name="weights")
            b = tf.Variable(tf.zeros([numClass], dtype=tf.float64, name="biases"))
            loss = calculateCrossEntropyLoss(XTrain, YTrain, w, b, lda, numClass)
            optimizer = tf.train.AdamOptimizer(learningRate).minimize(loss)
            init = tf.global_variables_initializer()
            sess.run(init)

            trainingLoss = [None for ep in range(epochs)]
            Weights = [None for ep in range(epochs)]
            Biases = [None for ep in range(epochs)]

            for ep in range(epochs):
                for i in range(iterPerEpoch):
                    if((i+1)*batchSize > N):
                        XBatch = trainData[-batchSize:]
                        YBatch = trainTarget[-batchSize:]
                    else:
                        XBatch = trainData[i * batchSize:(i + 1) * batchSize]
                        YBatch = trainTarget[i * batchSize:(i + 1) * batchSize]
                    feed = {XTrain: XBatch, YTrain: YBatch}
                    _ = sess.run(optimizer, feed_dict={XTrain: XBatch, YTrain: YBatch})
                trainingLoss[ep], Weights[ep], Biases[ep] = sess.run([loss, w, b], feed_dict={XTrain: XBatch, YTrain: YBatch})

            # After finishing training, find the validation accuracy
            accuracy, updateOp = calculateClassificationAccuracy(XValid, YValid, Weights[-1], Biases[-1])
            tf.local_variables_initializer().run()
            _, updateOp = sess.run([accuracy, updateOp], feed_dict={XValid:validData, YValid:validTarget})
            validAccuracy = sess.run(accuracy)
            print("Learning rate:", learningRate, "Lda:", lda, "trainingLoss:", trainingLoss[-1], "validAccuracy:", validAccuracy)
            if (validAccuracy > maxValidAccuracy):
                maxValidAccuracy = validAccuracy
                bestLearningRateIdx = lrIdx
                bestLdaIdx = ldaIdx
                Loss = copy.deepcopy(trainingLoss)
                bestWeight = copy.deepcopy(Weights)
                bestBias = copy.deepcopy(Biases)
    print("Best learning rate:", learnRates[bestLearningRateIdx], "Best lda:", ldas[bestLdaIdx], "Best validation accuracy:", maxValidAccuracy)
    
    trainingAccuracy = [None for ep in range(epochs)]
    validationLoss = [None for ep in range(epochs)]
    validationAccuracy = [None for ep in range(epochs)]

    XTrainAll = tf.placeholder(tf.float64, [N, d], name="X")
    YTrainAll = tf.placeholder(tf.int32, [N], name = "Y")

    for ep in range(epochs):
        weightEp = bestWeight[ep]
        biasEp = bestBias[ep]
        validLoss = calculateCrossEntropyLoss(XValid, YValid, weightEp, biasEp, ldas[bestLdaIdx], numClass)

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
    plt.title("cross-entropy loss vs number of epochs for learning rate %f and lambda %f" % (learnRates[bestLearningRateIdx], ldas[bestLdaIdx]))
    fig.savefig("part2_2_2_loss.png")
    fig = plt.figure(2)
    plt.plot(range(epochs), trainingAccuracy, c='r', label='training')
    plt.plot(range(epochs), validationAccuracy, c='b', label='validation')
    plt.xlabel('the n-th epoch')
    plt.ylabel('Classification accuracy')
    plt.legend()
    plt.title("Classification accuracy vs number of epochs for learning rate %f and lambda %f" % (learnRates[bestLearningRateIdx], ldas[bestLdaIdx]))
    fig.savefig("part2_2_2_accuracy.png")
    print("Classification accuracy on training data: %f" % trainingAccuracy[-1])
    print("Classification accuracy on validation data: %f" % validationAccuracy[-1])

