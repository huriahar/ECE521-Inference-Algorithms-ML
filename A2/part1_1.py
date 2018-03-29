from __future__ import print_function
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

sess = tf.InteractiveSession()

def loadData (fileName):
    with np.load(fileName) as data:
        Data, Target = data["images"], data["labels"]
        posClass = 2
        negClass = 9
        dataIndx = (Target == posClass) + (Target == negClass)
        Data = Data[dataIndx]/255.0
        Data = Data.reshape(-1,28*28)
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
    loss = tf.reduce_sum(tf.squared_difference(YHead, Y))
    N = X.get_shape().as_list()[0]
    loss = tf.divide(loss, tf.to_double(2*N))
    regularizer = tf.nn.l2_loss(w)
    loss = loss + lda*regularizer
    return loss

if __name__ == "__main__":
    trainData, trainTarget, validData, validTarget, testData, testTarget = loadData("notMNIST.npz")

    # part1.1
    batchSize = 500
    d = trainData.shape[1]      # 28*28 pixels/image = 784

    # Training and test data for each mini batch in each epoch
    XTrain = tf.placeholder(tf.float64, [batchSize, d])
    YTrain = tf.placeholder(tf.float64, [batchSize, 1])

    XValid = tf.placeholder(tf.float64, validData.shape)
    YValid = tf.placeholder(tf.float64, validTarget.shape)

    XTest = tf.placeholder(tf.float64, testData.shape)
    YTest = tf.placeholder(tf.float64, testTarget.shape)

    iteration = 20000.

    lda = 0.0                                           # lambda i.e. weight decay coefficient

    N = len(trainData)                                   # 3500
    losses = []
    
    init = tf.global_variables_initializer()
    sess.run(init)
    learnRate = [0.005, 0.001, 0.0001]
    iterPerEpoch = int(N / batchSize)                       # 7
    epochs = int(np.ceil(iteration/float(iterPerEpoch)))    # 2858
    plt.close('all')
    for index, lr in enumerate(learnRate):
        w = tf.Variable(tf.truncated_normal([d, 1], stddev=0.5, dtype=tf.float64), name="weights")
        b = tf.Variable(0.0, dtype=tf.float64, name="biases")
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        loss = calculateMSELoss(XTrain, YTrain, w, b, lda)
        fig = plt.figure(index*2 + 1)
        optimizer = tf.train.GradientDescentOptimizer(lr).minimize(loss)
        L = [None for ep in range(epochs)]
        for ep in range(epochs):
            for i in range(iterPerEpoch):
                XBatch = trainData[i*batchSize:(i+1)*batchSize]
                YBatch = trainTarget[i*batchSize:(i+1)*batchSize]
                feed = {XTrain:XBatch, YTrain:YBatch}
                _,  L[ep] = sess.run([optimizer, loss], feed_dict=feed)

        print("Minimum loss for learning rate", lr, "is:", L[-1])
        losses.append(L)
        plt.scatter(range(epochs), L, marker='|')
        plt.xlabel('the n-th epoch')
        plt.ylabel('loss')
        plt.title("MSE vs number of epochs for learning rate of %f" % lr)
        fig.savefig("part1_1_learnrate_%d.png"%index)
        fig = plt.figure(index * 2 + 2)
        plt.scatter(range(1500,epochs), L[1500:], marker='|')
        plt.xlabel('the n-th epoch')
        plt.ylabel('loss')
        plt.title("MSE vs number of epochs for learning rate of %f" % lr)
        fig.savefig("part1_1_learnrate_%d_zoomedin.png" % index)

    #####################
    fig = plt.figure((index+1)*2 + 1)
    plt.plot(range(epochs), losses[0], c='r', label='n = %f'%learnRate[0])
    plt.plot(range(epochs), losses[1], c='g', label='n = %f'%learnRate[1])
    plt.plot(range(epochs), losses[2], c='b', label='n = %f'%learnRate[2])
    plt.legend()
    plt.title("MSE vs number of epochs for different learning rates")
    plt.xlabel("Number of epochs")
    plt.ylabel("Men Squared Error")
    fig.savefig("part1_1_AllInOne.png")    
