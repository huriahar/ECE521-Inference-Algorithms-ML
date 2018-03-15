from __future__ import print_function
import numpy as np
import tensorflow as tf
import time
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


if __name__ == "__main__":
    trainData, trainTarget, validData, validTarget, testData, testTarget = loadData("notMNIST.npz")
    XValid = tf.placeholder(tf.float64, validData.shape)
    YValid = tf.placeholder(tf.float64, validTarget.shape)

    XTest = tf.placeholder(tf.float64, testData.shape)
    YTest = tf.placeholder(tf.float64, testTarget.shape)

    batchSize = 500
    d = 784
    N = len(trainData)


    XTrain = tf.placeholder(tf.float64, [batchSize, d])
    YTrain = tf.placeholder(tf.float64, [batchSize, 1])

    X = tf.placeholder(tf.float64, name="X")
    Y = tf.placeholder(tf.float64, name = "Y")

    l = 0.01 #lambda
    iteration = 5000.
    learnRate = [0.005, 0.001, 0.0001]
    iterPerEpoch = int(N / batchSize)
    epochs = int(np.ceil(iteration/float(iterPerEpoch)))
    print("Number of epochs=",epochs)
    plt.close('all')
    Loss = []
    LossV = []
    LossTest=[]
    AccuracyT = []
    AccuracyV = []
    AccuracyTest = []
    best_lr = learnRate[0]
    minimumLossValid = 1000000.0
    for index, lr in enumerate(learnRate):
        w = tf.Variable(tf.truncated_normal([d, 1], stddev=0.1, dtype=tf.float64), name="weights")
        b = tf.Variable(tf.truncated_normal([1], stddev=0.1, dtype=tf.float64), name="biases")
        #np.random.seed(521)

        #Training Cross Entropy Loss
        logits = (tf.matmul(XTrain,w) + b)
        loss_d = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=YTrain,logits=logits))
        loss_w = (l)*(tf.nn.l2_loss(w))
        loss = loss_d + loss_w
        optimizer = tf.train.GradientDescentOptimizer(lr).minimize(loss)

        #Validation Cross Entropy Loss
        logitsValid = (tf.matmul(XValid,w) + b)
        lossValid_d = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=YValid,logits=logitsValid))
        lossValid_w = (l)*(tf.nn.l2_loss(w))
        lossValid = lossValid_d + lossValid_w

        #Test Cross Entropy Loss
        logitsTest = (tf.matmul(XTest,w) + b)
        lossTest_d = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=YTest,logits=logitsTest))
        lossTest_w = (l)*(tf.nn.l2_loss(w))
        lossTest = lossTest_d + lossTest_w
        
        #Output Prediction
        YPred = tf.sigmoid((tf.matmul(X,w)+b))
        YClassfication = tf.cast(tf.greater(YPred , 0.5), tf.float64)
        YCorrect = tf.cast(tf.equal(YClassfication, Y),tf.float64)
        accuracy = tf.reduce_mean(tf.cast(YCorrect,tf.float64))*100

        fig = plt.figure(index*2+1)
     
        TrainingLoss = [None for ep in range(epochs)]
        TrainingAccuracy = [None for ep in range(epochs)]
        ValidLoss = [None for ep in range(epochs)]
        ValidAccuracy = [None for ep in range(epochs)]
        TestLoss = [None for ep in range(epochs)]
        TestAccuracy = [None for ep in range(epochs)]
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
                _,TrainingLoss[ep] = sess.run([optimizer,loss],feed_dict= {XTrain:XBatch,YTrain:YBatch})
            ValidLoss[ep] = sess.run(lossValid,feed_dict={XValid:validData, YValid:validTarget})
            TestLoss[ep] = sess.run(lossTest, feed_dict={XTest:testData,YTest:testTarget})
            TrainingAccuracy[ep]= sess.run(accuracy,feed_dict={X:trainData, Y:trainTarget})
            ValidAccuracy[ep]= sess.run(accuracy,feed_dict={X:validData, Y:validTarget})
            TestAccuracy[ep]= sess.run(accuracy,feed_dict={X:testData, Y:testTarget})

        # To Do: improve plots!!!!!!!
        plt.scatter(range(epochs), TrainingLoss, marker='.',)
        plt.scatter(range(epochs), TrainingAccuracy, marker='.',)
        plt.scatter(range(epochs), ValidLoss, marker='*',)
        plt.scatter(range(epochs), ValidAccuracy, marker='*',)
        plt.xlabel('the n-th epoch')
        plt.ylabel('loss/Accuracy')
        plt.title("MSE vs number of epoch for learning rate of %f" % lr)
        fig.savefig("part2_1_learnrate_%d.png"%index)
        fig = plt.figure(index * 2 + 2)
        plt.scatter(range(100,epochs), TrainingLoss[100:], marker='.', )
        plt.xlabel('the n-th epoch')
        plt.ylabel('loss')
        plt.title("MSE vs number of epoch for learning rate of %f" % lr)
        fig.savefig("part2_1_learnrate_%d_zoomedin.png" % index)
        
        #appending the losses and accuracies across epochs for training, validation and test
        Loss.append(TrainingLoss)
        LossV.append(ValidLoss)
        LossTest.append(TestLoss)
        AccuracyT.append(TrainingAccuracy)
        AccuracyV.append(ValidAccuracy)
        AccuracyTest.append(TestAccuracy)

    print("Validation Loss")
    print("Learning rate 0.005",LossV[0][epochs-1])
    print("Learning rate 0.001",LossV[1][epochs-1])
    print("Learning rate 0.0001",LossV[2][epochs-1])

    print("Training Accuracy")
    print("Lerning rate 0.005",AccuracyT[0][epochs-1])
    print("Learning rate 0.001",AccuracyT[1][epochs-1])
    print("Learning rate 0.0001",AccuracyT[2][epochs-1])

    print("Validation Accuracy")
    print("Lerning rate 0.005",AccuracyV[0][epochs-1])
    print("Learning rate 0.001",AccuracyV[1][epochs-1])
    print("Learning rate 0.0001",AccuracyV[2][epochs-1])
   
    print("Test Accuracy")
    print("Lerning rate 0.005",AccuracyTest[0][epochs-1])
    print("Learning rate 0.001",AccuracyTest[1][epochs-1])
    print("Learning rate 0.0001",AccuracyTest[2][epochs-1])

    #To Do: improve plots for all learning rates in one plot
    fig2 = plt.figure(7)
    plt.xlabel('the n-th epoch')
    plt.ylabel('loss')
    plt.scatter(range(epochs), Loss[0],marker='*', c='r')
    plt.scatter(range(epochs), Loss[1], marker='*', c='b')
    plt.scatter(range(epochs), Loss[2],marker='*', c='g')
    plt.scatter(range(epochs), LossV[0],marker='.', c='r')
    plt.scatter(range(epochs), LossV[1], marker='.', c='b')
    plt.scatter(range(epochs), LossV[2],marker='.', c='g')
    fig2.savefig("part2_1_all_learning_rates")

