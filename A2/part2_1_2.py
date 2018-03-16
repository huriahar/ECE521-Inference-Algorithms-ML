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

def calculateCrossEntropyLoss(x,y,weights,bias,lambdaParam):
    logits = (tf.matmul(x,weights) + bias)
    loss_d = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y,logits=logits))
    loss_w = (lambdaParam)*(tf.nn.l2_loss(weights))
    crossEntropyLoss = loss_d + loss_w
    return crossEntropyLoss

def calculateClassificationAccuracy(x,y,weights,bias):
    YPred = tf.sigmoid((tf.matmul(x,weights)+bias))
    YClassfication = tf.cast(tf.greater(YPred , 0.5), tf.float64)
    YCorrect = tf.cast(tf.equal(YClassfication, y),tf.float64)
    accuracy = tf.reduce_mean(tf.cast(YCorrect,tf.float64))*100
    return accuracy

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
    learnRate = 0.001
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

    TrainingLossSGD = [None for ep in range(epochs)]
    TrainingAccuracySGD = [None for ep in range(epochs)]
    TrainingLossAdam = [None for ep in range(epochs)]
    TrainingAccuracyAdam = [None for ep in range(epochs)]

    ValidLossSGD = [None for ep in range(epochs)]
    ValidAccuracySGD = [None for ep in range(epochs)]
    ValidLossAdam = [None for ep in range(epochs)]
    ValidAccuracyAdam= [None for ep in range(epochs)]

    TestLossSGD = [None for ep in range(epochs)]
    TestAccuracySGD = [None for ep in range(epochs)]
    TestLossAdam= [None for ep in range(epochs)]
    TestAccuracyAdam = [None for ep in range(epochs)]

    wSGD = tf.Variable(tf.truncated_normal([d, 1], stddev=0.1, dtype=tf.float64), name="weights")
    bSGD = tf.Variable(tf.truncated_normal([1], stddev=0.1, dtype=tf.float64), name="biases")

    wAD = tf.Variable(tf.truncated_normal([d, 1], stddev=0.1, dtype=tf.float64), name="weights")
    bAD = tf.Variable(tf.truncated_normal([1], stddev=0.1, dtype=tf.float64), name="biases")
   

    lossSGD = calculateCrossEntropyLoss(XTrain,YTrain,wSGD,bSGD,l)
    optimizerSGD = tf.train.GradientDescentOptimizer(learnRate).minimize(lossSGD)

    lossAD = calculateCrossEntropyLoss(XTrain,YTrain,wAD,bAD,l)
    optimizerAD = tf.train.AdamOptimizer(learnRate).minimize(lossAD)

    lossSGDValid = calculateCrossEntropyLoss(XValid,YValid,wSGD,bSGD,l)
    lossSGDTest = calculateCrossEntropyLoss(XTest,YTest,wSGD,bSGD,l)

    lossAdamValid = calculateCrossEntropyLoss(XValid,YValid,wAD,bAD,l)
    lossAdamTest = calculateCrossEntropyLoss(XTest,YTest,wAD,bAD,l)

    accuracySGD = calculateClassificationAccuracy(X,Y,wSGD,bSGD)
    accuracyAdam = calculateClassificationAccuracy(X,Y,wAD,bAD)

    fig = plt.figure(1)

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
            _,TrainingLossSGD[ep] = sess.run([optimizerSGD,lossSGD],feed_dict= {XTrain:XBatch,YTrain:YBatch})
            _,TrainingLossAdam[ep] = sess.run([optimizerAD,lossAD],feed_dict= {XTrain:XBatch,YTrain:YBatch})
        
        ValidLossSGD[ep] = sess.run(lossSGDValid,feed_dict={XValid:validData, YValid:validTarget})
        TestLossSGD[ep] = sess.run(lossSGDTest, feed_dict={XTest:testData,YTest:testTarget})
        TrainingAccuracySGD[ep]= sess.run(accuracySGD,feed_dict={X:trainData, Y:trainTarget})
        ValidAccuracySGD[ep]= sess.run(accuracySGD,feed_dict={X:validData, Y:validTarget})
        TestAccuracySGD[ep]= sess.run(accuracySGD,feed_dict={X:testData, Y:testTarget})

        ValidLossAdam[ep] = sess.run(lossAdamValid,feed_dict={XValid:validData, YValid:validTarget})
        TestLossAdam[ep] = sess.run(lossAdamTest, feed_dict={XTest:testData,YTest:testTarget})
        TrainingAccuracyAdam[ep]= sess.run(accuracyAdam,feed_dict={X:trainData, Y:trainTarget})
        ValidAccuracyAdam[ep]= sess.run(accuracyAdam,feed_dict={X:validData, Y:validTarget})
        TestAccuracyAdam[ep]= sess.run(accuracyAdam,feed_dict={X:testData, Y:testTarget})

    # To Do: improve plots!!!!!!!
    plt.scatter(range(epochs), TrainingLossSGD, marker='.', c= 'r', label ="SGD")
    plt.scatter(range(epochs), TrainingLossAdam, marker='*', c= 'b', label = "Adam")
    plt.legend()
    plt.xlabel('the n-th epoch')
    plt.ylabel('loss/Accuracy')
    plt.title("MSE vs number of epoch for learning rate of %f" % learnRate)
    fig.savefig("part2_1_2_learnrate.png")

    
    #appending the losses and accuracies across epochs for training, validation and test
    Loss.append(TrainingLossSGD)
    Loss.append(TrainingLossAdam)
    LossV.append(ValidLossSGD)
    LossV.append(ValidLossAdam)
    LossTest.append(TestLossSGD)
    LossTest.append(TestLossAdam)

    AccuracyT.append(TrainingAccuracySGD)
    AccuracyT.append(TrainingAccuracyAdam)
    AccuracyV.append(ValidAccuracySGD)
    AccuracyV.append(ValidAccuracyAdam)
    AccuracyTest.append(TestAccuracySGD)
    AccuracyTest.append(TestAccuracyAdam)


    print("SGD")
    print("Training Loss")
    print("Learning rate 0.001",Loss[0][epochs-1])

    print("Validation Loss")
    print("Learning rate 0.001",LossV[0][epochs-1])

    print("Test Loss")
    print("Learning rate 0.001",LossTest[0][epochs-1])

    print("Training Accuracy")
    print("Learning rate 0.001",AccuracyT[0][epochs-1])

    print("Validation Accuracy")
    print("Learning rate 0.001",AccuracyV[0][epochs-1])
   
    print("Test Accuracy")
    print("Learning rate 0.001",AccuracyTest[0][epochs-1])
  
    print("Adam")
    print("Training Loss")
    print("Learning rate 0.001",Loss[1][epochs-1])

    print("Validation Loss")
    print("Learning rate 0.001",LossV[1][epochs-1])

    print("Test Loss")
    print("Learning rate 0.001",LossTest[1][epochs-1])

    print("Training Accuracy")
    print("Learning rate 0.001",AccuracyT[1][epochs-1])

    print("Validation Accuracy")
    print("Learning rate 0.001",AccuracyV[1][epochs-1])
   
    print("Test Accuracy")
    print("Learning rate 0.001",AccuracyTest[1][epochs-1])

