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
    learnRate = [0.005, 0.001, 0.0001]
    iterPerEpoch = int(N / batchSize)
    epochs = int(np.ceil(iteration/float(iterPerEpoch)))
    print("Number of epochs=",epochs)
    plt.close('all')
    
    #For training set
    Weights =[]
    Bias=[]
    Loss = []

    for index, lr in enumerate(learnRate):
        w = tf.Variable(tf.truncated_normal([d, 1], stddev=0.1, dtype=tf.float64), name="weights")
        b = tf.Variable(tf.truncated_normal([1], stddev=0.1, dtype=tf.float64), name="biases")
        TrainingLoss = [None for ep in range(epochs)]
        TrainingWeights =[None for ep in range(epochs)]
        TrainingBias = [None for ep in range(epochs)]
     
        loss = calculateCrossEntropyLoss(XTrain,YTrain,w,b,l)
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
                _,TrainingLoss[ep], TrainingWeights[ep],TrainingBias[ep]= sess.run([optimizer,loss,w,b],feed_dict= {XTrain:XBatch,YTrain:YBatch})
        
        Loss.append(TrainingLoss)
        Weights.append(TrainingWeights)
        Bias.append(TrainingBias)


    #Select the best learning rate based on training loss
    best_lr = learnRate[0]
    bestLrIndex = 0
    minimumTrainingLoss= float('inf')

    for index,lr in enumerate(learnRate):
        print(Loss[index][-1], lr)
        if(Loss[index][-1] <minimumTrainingLoss):
            minimumTrainingLoss = Loss[index][-1]
            best_lr = learnRate[index]
            bestLrIndex = index

    #Calculating training accuracy, validation loss and validation accuracy for best learning rate
    TrainingAccuracy = [None for ep in range(epochs)]
    ValidLoss = [None for ep in range(epochs)]
    ValidAccuracy = [None for ep in range(epochs)]
    TestAccuracy = [None for ep in range(epochs)]
    for ep in range(epochs):
        weightEp = Weights[bestLrIndex][ep]
        biasEp =Bias[bestLrIndex][ep]

        #accuracy
        accuracy = calculateClassificationAccuracy(X,Y,weightEp,biasEp)

        #Validation Loss
        validationLoss = calculateCrossEntropyLoss(XValid, YValid, weightEp, biasEp,l)

        TrainingAccuracy[ep]= sess.run(accuracy, feed_dict={X:trainData,Y:trainTarget})
        ValidLoss[ep] = sess.run(validationLoss,feed_dict={XValid: validData, YValid: validTarget})
        ValidAccuracy[ep] = sess.run(accuracy,feed_dict={X:validData, Y:validTarget})
        TestAccuracy[ep] = sess.run(accuracy,feed_dict={X:testData, Y:testTarget})


    plt.close('all')
    fig = plt.figure(1)
    plt.scatter(range(epochs), Loss[bestLrIndex], marker='.', c='r', label='training')
    plt.scatter(range(epochs), ValidLoss, marker='.', c='b', label='validation')
    plt.xlabel('the n-th epoch')
    plt.ylabel('cross-entropy loss')
    plt.legend()
    plt.title("cross-entropy loss vs number of epochs for learning rate of %f" % learnRate[bestLrIndex])
    fig.savefig("part2_1_1_loss.png")
    fig = plt.figure(2)
    plt.scatter(range(epochs), TrainingAccuracy, marker='.', c='r', label='training')
    plt.scatter(range(epochs), ValidAccuracy, marker='.', c='b', label='validation')
    plt.xlabel('the n-th epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.title("accuracy vs number of epochs for learning rate of %f" % learnRate[bestLrIndex])
    fig.savefig("part2_1_1_accuracy.png")
    print("Best Learning Rate: %f" % learnRate[bestLrIndex])
    print("Classification accuracy on training data: %f" % TrainingAccuracy[-1])
    print("Classification accuracy on validation data: %f" % ValidAccuracy[-1])
    print("Classification accuracy on test data: %f" % TestAccuracy[-1])

