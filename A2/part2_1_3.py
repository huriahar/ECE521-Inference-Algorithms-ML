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

def calculateSquaredErrorLoss(x,y,weights,bias,lambdaParam):
    YHead = tf.matmul(x,weights) + bias
    loss = tf.reduce_mean(tf.squared_difference(YHead, y))
    loss = tf.divide(loss,tf.to_double(2))
    regularizer = tf.nn.l2_loss(weights)
    SquaredErrorloss = loss + lambdaParam * regularizer
    return SquaredErrorloss

def calculateLogClassificationAccuracy(x,y,weights,bias):
    YPred = tf.sigmoid((tf.matmul(x,weights)+bias))
    YClassfication = tf.cast(tf.greater(YPred , 0.5), tf.float64)
    YCorrect = tf.cast(tf.equal(YClassfication, y),tf.float64)
    accuracy = tf.reduce_mean(tf.cast(YCorrect,tf.float64))*100
    return accuracy

def calculateLinearClassificationAccuracy(x,y,weights,bias):
    YPred = tf.matmul(x,weights)+bias
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

    l = 0.0 #lambda
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

    TrainingLossLogistic = [None for ep in range(epochs)]
    TrainingAccuracyLogistic = [None for ep in range(epochs)]
    TrainingLossLinear = [None for ep in range(epochs)]
    TrainingAccuracyLinear = [None for ep in range(epochs)]

    ValidLossLogistic = [None for ep in range(epochs)]
    ValidAccuracyLogistic = [None for ep in range(epochs)]
    ValidLossLinear = [None for ep in range(epochs)]
    ValidAccuracyLinear= [None for ep in range(epochs)]

    TestLossLogistic = [None for ep in range(epochs)]
    TestAccuracyLogistic = [None for ep in range(epochs)]
    TestLossLinear= [None for ep in range(epochs)]
    TestAccuracyLinear = [None for ep in range(epochs)]

    wLog = tf.Variable(tf.truncated_normal([d, 1], stddev=0.1, dtype=tf.float64), name="weights")
    bLog = tf.Variable(tf.truncated_normal([1], stddev=0.1, dtype=tf.float64), name="biases")

    wLin = tf.Variable(tf.truncated_normal([d, 1], stddev=0.1, dtype=tf.float64), name="weights")
    bLin = tf.Variable(tf.truncated_normal([1], stddev=0.1, dtype=tf.float64), name="biases")
   

    lossLog = calculateCrossEntropyLoss(XTrain,YTrain,wLog,bLog,l)
    optimizerLog = tf.train.AdamOptimizer(learnRate).minimize(lossLog)

    lossLin = calculateSquaredErrorLoss(XTrain,YTrain,wLin,bLin,l)
    optimizerLin = tf.train.AdamOptimizer(learnRate).minimize(lossLin)

    lossLogValid = calculateCrossEntropyLoss(XValid,YValid,wLog,bLog,l)
    lossLogTest = calculateCrossEntropyLoss(XTest,YTest,wLog,bLog,l)

    lossLinValid = calculateSquaredErrorLoss(XValid,YValid,wLin,bLin,l)
    lossLinTest = calculateSquaredErrorLoss(XTest,YTest,wLin,bLin,l)

    accuracyLog = calculateLogClassificationAccuracy(X,Y,wLog,bLog)
    accuracyLin = calculateLinearClassificationAccuracy(X,Y,wLin,bLin)

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
            _,TrainingLossLogistic[ep] = sess.run([optimizerLog,lossLog],feed_dict= {XTrain:XBatch,YTrain:YBatch})
            _,TrainingLossLinear[ep] = sess.run([optimizerLin,lossLin],feed_dict= {XTrain:XBatch,YTrain:YBatch})
        
        ValidLossLogistic[ep] = sess.run(lossLogValid,feed_dict={XValid:validData, YValid:validTarget})
        TestLossLogistic[ep] = sess.run(lossLogTest, feed_dict={XTest:testData,YTest:testTarget})
        TrainingAccuracyLogistic[ep]= sess.run(accuracyLog,feed_dict={X:trainData, Y:trainTarget})
        ValidAccuracyLogistic[ep]= sess.run(accuracyLog,feed_dict={X:validData, Y:validTarget})
        TestAccuracyLogistic[ep]= sess.run(accuracyLog,feed_dict={X:testData, Y:testTarget})

        ValidLossLinear[ep] = sess.run(lossLinValid,feed_dict={XValid:validData, YValid:validTarget})
        TestLossLinear[ep] = sess.run(lossLinTest, feed_dict={XTest:testData,YTest:testTarget})
        TrainingAccuracyLinear[ep]= sess.run(accuracyLin,feed_dict={X:trainData, Y:trainTarget})
        ValidAccuracyLinear[ep]= sess.run(accuracyLin,feed_dict={X:validData, Y:validTarget})
        TestAccuracyLinear[ep]= sess.run(accuracyLin,feed_dict={X:testData, Y:testTarget})

    # To Do: improve plots!!!!!!!
    plt.scatter(range(epochs), TrainingLossLogistic, marker='.', c= 'r', label ="Logistic Regression")
    plt.scatter(range(epochs), TrainingLossLinear, marker='*', c= 'b', label = "Linear Regression")
    plt.legend()
    plt.xlabel('the n-th epoch')
    plt.ylabel('loss/Accuracy')
    plt.title("Cross Entropy Loss/MSE vs number of epoch for learning rate of %f" % learnRate)
    fig.savefig("part2_1_3_learnrate_loss.png")

    fig = plt.figure(2)
    plt.scatter(range(epochs), TrainingAccuracyLogistic, marker='.', c='r', label='Logistic Regression')
    plt.scatter(range(epochs), TrainingAccuracyLinear, marker='.', c='b', label='Linear Regression')
    plt.xlabel('the n-th epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.title("Accuracy vs number of epochs for learning rate of %f" % learnRate)
    fig.savefig("part2_1_3_accuracy.png")
    
    #appending the losses and accuracies across epochs for training, validation and test
    Loss.append(TrainingLossLogistic)
    Loss.append(TrainingLossLinear)
    LossV.append(ValidLossLogistic)
    LossV.append(ValidLossLinear)
    LossTest.append(TestLossLogistic)
    LossTest.append(TestLossLinear)

    AccuracyT.append(TrainingAccuracyLogistic)
    AccuracyT.append(TrainingAccuracyLinear)
    AccuracyV.append(ValidAccuracyLogistic)
    AccuracyV.append(ValidAccuracyLinear)
    AccuracyTest.append(TestAccuracyLogistic)
    AccuracyTest.append(TestAccuracyLinear)

    #debugging
    print("Logistic")
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
  
    print("Linear")
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

    #dummy graph plots

    dummyPred = tf.cast(tf.linspace(0.,1.,num=100), tf.float64)
    dummyTarget = tf.zeros(100, tf.float64)

    crossEntropyLossDummy = tf.nn.sigmoid_cross_entropy_with_logits(labels=dummyTarget,logits=dummyPred)
    squaredDifferenceLossDummy = tf.squared_difference(dummyPred,dummyTarget)

    crossEntropyLoss = sess.run(crossEntropyLossDummy)
    squaredDifferenceLoss = sess.run(squaredDifferenceLossDummy)

    fig = plt.figure(3)
    plt.scatter(np.linspace(0.,1.,num=100), crossEntropyLoss, marker='.', c= 'r', label ="Cross Entropy Error Loss")
    plt.scatter(np.linspace(0.,1.,num=100), squaredDifferenceLoss, marker='*', c= 'b', label = "MSE Loss")
    plt.legend()
    plt.xlabel('prediction')
    plt.ylabel('loss')
    plt.title("MSE/CrossEntropyLoss vs prediction")
    fig.savefig("part2_1_3_learnrate_loss_dummy.png")