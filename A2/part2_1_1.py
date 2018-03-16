from __future__ import print_function
import numpy as np
import tensorflow as tf
import time, copy
import matplotlib.pyplot as plt

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

def calculateCrossEntropyLoss(x, y, weights, bias, lambdaParam):
    logits = tf.matmul(x,weights) + bias
    loss_d = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y,logits=logits))
    loss_w = (lambdaParam)*(tf.nn.l2_loss(weights))
    crossEntropyLoss = loss_d + loss_w
    return crossEntropyLoss

def calculateClassificationAccuracy(x, y, weights, bias):
    YPred = tf.sigmoid(tf.matmul(x,weights) + bias)
    YClassfication = tf.cast(tf.greater(YPred , 0.5), tf.float64)
    YCorrect = tf.cast(tf.equal(YClassfication, y), tf.float64)
    accuracy = tf.reduce_mean(tf.cast(YCorrect,tf.float64))*100
    return accuracy

if __name__ == "__main__":
    trainData, trainTarget, validData, validTarget, testData, testTarget = loadData("notMNIST.npz")
    batchSize = 500
    d = trainData.shape[1]                                 # 28*28 = 784
    N = len(trainData)                                     # 3500

    XTrain = tf.placeholder(tf.float64, [batchSize, d])
    YTrain = tf.placeholder(tf.float64, [batchSize, 1])

    XValid = tf.placeholder(tf.float64, validData.shape)
    YValid = tf.placeholder(tf.float64, validTarget.shape)

    XTest = tf.placeholder(tf.float64, testData.shape)
    YTest = tf.placeholder(tf.float64, testTarget.shape)

    X = tf.placeholder(tf.float64, name="X")
    Y = tf.placeholder(tf.float64, name = "Y")

    l = 0.01                                                # lambda
    iteration = 5000.
    learnRate = [0.005, 0.001, 0.0001]
    iterPerEpoch = int(N / batchSize)                       # 7
    epochs = int(np.ceil(iteration/float(iterPerEpoch)))    # 784
    plt.close('all')
    #For training set
    bestWeight = None
    bestBias = None
    Loss = []
    minimumTrainingLoss = float('inf')
    bestLearningRateIdx = 0

    for index, lr in enumerate(learnRate):
        w = tf.Variable(tf.truncated_normal([d, 1], stddev=0.1, seed=521, dtype=tf.float64), name="weights")
        b = tf.Variable(tf.truncated_normal([1], stddev=0.1, seed=521, dtype=tf.float64), name="biases")
        TrainingLoss = [None for ep in range(epochs)]
        Weights = [None for ep in range(epochs)]
        Biases = [None for ep in range(epochs)]
     
        loss = calculateCrossEntropyLoss(XTrain, YTrain, w, b, l)
        optimizer = tf.train.GradientDescentOptimizer(lr).minimize(loss)

        init = tf.global_variables_initializer()
        sess.run(init)
        for ep in range(epochs):
            for iteration in range(iterPerEpoch):
                if((iteration +1)*batchSize > N):
                    XBatch = trainData[-batchSize:]
                    YBatch = trainTarget[-batchSize:]
                else:
                    XBatch = trainData[iteration*batchSize :(iteration+1)*batchSize]
                    YBatch = trainTarget[iteration*batchSize:(iteration+1)*batchSize]
                _ = sess.run(optimizer, feed_dict = {XTrain:XBatch, YTrain:YBatch})
            TrainingLoss[ep], Weights[ep], Biases[ep] = sess.run([loss, w, b], feed_dict = {XTrain:XBatch, YTrain:YBatch})
        
        print("Training cross-entropy loss with learning rate", lr, "is:", TrainingLoss[-1])
        # Check if this is the least loss seen so far. Best learning rate selected through lest training loss
        if (TrainingLoss[-1] < minimumTrainingLoss):
            minimumTrainingLoss = TrainingLoss[-1]
            bestLearningRateIdx = index
            Loss = copy.deepcopy(TrainingLoss)
            bestWeight = copy.deepcopy(Weights)
            bestBias = copy.deepcopy(Biases)

    print("Best learning rate", learnRate[bestLearningRateIdx], "loss:", Loss[-1])

    #Calculating training accuracy, validation loss and validation accuracy for best learning rate
    TrainingAccuracy = [None for ep in range(epochs)]
    ValidLoss = [None for ep in range(epochs)]
    ValidAccuracy = [None for ep in range(epochs)]
    TestAccuracy = [None for ep in range(epochs)]

    for ep in range(epochs):
        #accuracy
        accuracy = calculateClassificationAccuracy(X, Y, bestWeight[ep], bestBias[ep])

        #Validation Loss
        validationLoss = calculateCrossEntropyLoss(XValid, YValid, bestWeight[ep], bestBias[ep], l)

        TrainingAccuracy[ep]= sess.run(accuracy, feed_dict={X:trainData, Y:trainTarget})
        ValidLoss[ep] = sess.run(validationLoss, feed_dict={XValid:validData, YValid: validTarget})
        ValidAccuracy[ep] = sess.run(accuracy, feed_dict={X:validData, Y:validTarget})
        TestAccuracy[ep] = sess.run(accuracy, feed_dict={X:testData, Y:testTarget})

    plt.close('all')
    fig = plt.figure(1)
    plt.plot(range(epochs), Loss, c='r', label='training')
    plt.plot(range(epochs), ValidLoss, c='b', label='validation')
    plt.xlabel('the n-th epoch')
    plt.ylabel('cross-entropy loss')
    plt.legend()
    plt.title("cross-entropy loss vs number of epochs for learning rate of %f" % learnRate[bestLearningRateIdx])
    fig.savefig("part2_1_1_loss.png")
    fig = plt.figure(2)
    plt.plot(range(epochs), TrainingAccuracy, c='r', label='training')
    plt.plot(range(epochs), ValidAccuracy, c='b', label='validation')
    plt.xlabel('the n-th epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.title("accuracy vs number of epochs for learning rate of %f" % learnRate[bestLearningRateIdx])
    fig.savefig("part2_1_1_accuracy.png")
    print("Best Learning Rate: %f" % learnRate[bestLearningRateIdx])
    print("Classification accuracy on training data: %f" % TrainingAccuracy[-1])
    print("Classification accuracy on validation data: %f" % ValidAccuracy[-1])
    print("Classification accuracy on test data: %f" % TestAccuracy[-1])
