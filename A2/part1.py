from __future__ import print_function
import numpy as np
import tensorflow as tf

sess = tf.InteractiveSession()

def loadData (fileName):
    with np.load(fileName) as data:
        print(fileName)
        Data, Target = data["images"], data["labels"]
        posClass = 2
        negClass = 9
        dataIndx = (Target == posClass) + (Target == negClass)
        Data = Data[dataIndx]/255.0
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

    XTrain = tf.placeholder(tf.float64, trainData.shape)
    YTrain = tf.placeholder(tf.int32, trainTarget.shape)

    XValid = tf.placeholder(tf.float64, validData.shape)
    YValid = tf.placeholder(tf.int32, validTarget.shape)

    XTest  = tf.placeholder(tf.float64, testData.shape)   
    YTest  = tf.placeholder(tf.int32, testTarget.shape)

    init = tf.global_variables_initializer()
    sess.run(init)

    