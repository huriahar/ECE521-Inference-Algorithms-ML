from __future__ import print_function
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

NUMVALID = 5

def loadData (fileName):
    with np.load(fileName) as data:
        Data, Target = data["images"], data["labels"]
        np.random.seed(521)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data = Data[randIndx]/255.0
        Data = Data.reshape(-1,28*28)
        Target = Target[randIndx].reshape(-1, 1)
        trainData, trainTarget = Data[:15000], Target[:15000]
        validData, validTarget = Data[15000:16000], Target[15000:16000]
        testData, testTarget = Data[16000:], Target[16000:]
        return trainData, trainTarget, validData, validTarget, testData, testTarget

def layerBuildingBlock (XLMinus1, numHiddenUnits):
    # Initialize W - Use Xavier initialization for weight matrix
    dimLMinus1 = XLMinus1.get_shape().as_list()[1]
    xavierStdDev = np.sqrt(3.0/(dimLMinus1 + numHiddenUnits))
    # Alternate method
    # W = tf.Variable(shape=[dimLMinus1, numHiddenUnits], dtype=tf.float64, initializer=tf.contrib.layers.xavier_initializer(), name="weights")
    W = tf.Variable(tf.truncated_normal(shape=[dimLMinus1, numHiddenUnits], dtype=tf.float64, seed=521, stddev=xavierStdDev, name="weights"))
    b = tf.Variable(tf.zeros(shape=[numHiddenUnits], dtype=tf.float64, name="bias"))
    return tf.matmul(XLMinus1, W) + b, W, b

def calculateCrossEntropyLoss(logits, weights, Y, numClass, lambdaparam):
    labels = tf.squeeze(tf.one_hot(Y, numClass,dtype=tf.float64))
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
    regularizer = tf.nn.l2_loss(weights)
    loss = loss + lambdaparam * regularizer
    return loss

def calculateClassificationAccuracy(predictedValues, actualValues):
    correctPrediction = tf.equal(tf.squeeze(actualValues), tf.argmax(predictedValues, 1, output_type=tf.int32))
    accuracy = tf.reduce_mean(tf.cast(correctPrediction, tf.float64))
    return accuracy*100

def calculateClassificationError(predictedValues,actualValues):
    accuracy = calculateClassificationAccuracy(predictedValues,actualValues)
    return 100.0-accuracy

# Returns true if the validationPartial list has values in ascending order
def validationIncreasing (validationPartial):
    result = all(validationPartial[i] <= validationPartial[i + 1] for i in range(len(validationPartial) - 1))
    return result

# def generateImage(weights):
#     weights = tf.unstack(weights,100)


if __name__ == '__main__':
    trainData, trainTarget, validData, validTarget, testData, testTarget = loadData("notMNIST.npz")

    d = trainData.shape[1]          # 28x28
    N = len(trainData)              # 15000 training examples

    batchSize = 500

    XNN = tf.placeholder(tf.float64, [None, d])
    YNN = tf.placeholder(tf.int32, [None, 1])
    KeepProbability = tf.placeholder(tf.float64)

    iteration = 6000.
    lda = 3e-4
    ldaDropOut = 0.0

    iterPerEpoch = int(N / batchSize)                       # 30
    epochs = int(np.ceil(iteration/float(iterPerEpoch)))    # 200

    finalCELossEpochs = epochs
    CELossEarlyStopped = False

    finalCELossEpochsDropOut = epochs
    CELossEarlyStoppedDropOut = False

    finalClErrorEpochs = epochs
    ClErrorEarlyStopped = False

    finalClErrorEpochsDropOut = epochs
    ClErrorEarlyStoppedDropOut = False


    learningRate = 0.005                                    # Chosen from part 1.2

    numHiddenUnits = 1000
    numClasses = 10

    
    trainingHiddenLayerWeights= [None for _ in range(epochs)]
    trainingLoss = [None for _ in range(epochs)]
    validationLoss = [None for _ in range(epochs)]
    testLoss = [None for _ in range(epochs)]
    trainingClassificationError = [None for _ in range(epochs)]
    validationClassificationError = [None for _ in range(epochs)]
    testClassificationError = [None for _ in range(epochs)]

    trainingHiddenLayerWeightsDropOut = [None for _ in range(epochs)]
    trainingLossDropOut = [None for _ in range(epochs)]
    validationLossDropOut = [None for _ in range(epochs)]
    testLossDropOut = [None for _ in range(epochs)]
    trainingClassificationErrorDropOut = [None for _ in range(epochs)]
    validationClassificationErrorDropOut = [None for _ in range(epochs)]
    testClassificationErrorDropOut = [None for _ in range(epochs)]

    plt.close()

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        hiddenLayerInput, WHidden, BHidden = layerBuildingBlock(XNN, numHiddenUnits)
        hiddenLayerOutput = tf.nn.relu(hiddenLayerInput)

        #DropOut 
        dropOutHiddenLayer = tf.nn.dropout(hiddenLayerOutput, KeepProbability)

        #Case: Dropout 
        outputLayerInputDropOut, WOutputDropOut, BOutputDropOut = layerBuildingBlock(dropOutHiddenLayer, numClasses) 
        outputLayerOutputDropOut = tf.nn.softmax(outputLayerInputDropOut)

        crossEntropyLossDropOut = calculateCrossEntropyLoss(outputLayerInputDropOut, WOutputDropOut, YNN, numClasses, ldaDropOut)
        optimizerDropOut = tf.train.AdamOptimizer(learningRate).minimize(crossEntropyLossDropOut)
        classificationErrorDropOut = calculateClassificationError(outputLayerOutputDropOut, YNN)
        
        #Case: No Dropout
        outputLayerInput, WOutput, BOutput = layerBuildingBlock(hiddenLayerOutput, numClasses) 
        outputLayerOutput = tf.nn.softmax(outputLayerInput)

        crossEntropyLoss = calculateCrossEntropyLoss(outputLayerInput, WOutput, YNN, numClasses, lda)
        optimizer = tf.train.AdamOptimizer(learningRate).minimize(crossEntropyLoss)
        classificationError = calculateClassificationError(outputLayerOutput, YNN)

        tf.global_variables_initializer().run()

        for epoch in range(epochs):
            for i in range(iterPerEpoch):
                XBatch = trainData[i*batchSize:(i+1)*batchSize]
                YBatch = trainTarget[i*batchSize:(i+1)*batchSize]
                feed = {XNN:XBatch, YNN:YBatch, KeepProbability: 0.5}
                _,trainingHiddenLayerWeights[epoch] = sess.run([optimizer,WHidden],feed_dict=feed)
                _,trainingHiddenLayerWeightsDropOut[epoch] = sess.run([optimizerDropOut,WHidden],feed_dict=feed)
            
            #Case: No Dropout
            trainingLoss[epoch], trainingClassificationError[epoch]= sess.run([crossEntropyLoss,classificationError], feed_dict= feed)
            feed = {XNN:validData, YNN:validTarget, KeepProbability:1.0}
            validationLoss[epoch], validationClassificationError[epoch] = sess.run([crossEntropyLoss, classificationError], feed_dict=feed)
            feed = {XNN:testData, YNN:testTarget, KeepProbability:1.0}
            testLoss[epoch], testClassificationError[epoch] = sess.run([crossEntropyLoss, classificationError], feed_dict=feed)

            #Case: Dropout
            trainingLossDropOut[epoch],trainingClassificationErrorDropOut[epoch] = sess.run([crossEntropyLossDropOut,classificationErrorDropOut], feed_dict= feed)
            feed = {XNN:validData, YNN:validTarget, KeepProbability:1.0}
            validationLossDropOut[epoch], validationClassificationErrorDropOut[epoch] = sess.run([crossEntropyLossDropOut, classificationErrorDropOut], feed_dict=feed)
            feed = {XNN:testData, YNN:testTarget, KeepProbability:1.0}
            testLossDropOut[epoch], testClassificationErrorDropOut[epoch] = sess.run([crossEntropyLossDropOut, classificationErrorDropOut], feed_dict=feed)


            if epoch >= NUMVALID:
                # Check if validation error has been continuosly increasing for the last 5 epochs
                # If so, early stop
                if not CELossEarlyStoppedDropOut and validationIncreasing(validationLossDropOut[epoch-NUMVALID:epoch]):
                    # Only epochs up till finalCELossEpochs should be considered
                    finalCELossEpochsDropOut = epoch - NUMVALID
                    CELossEarlyStoppedDropOut = True

                if not ClErrorEarlyStoppedDropOut and validationIncreasing(validationClassificationErrorDropOut[epoch-NUMVALID:epoch]):
                    # Only epochs up till finalClErrorEpochs should be considered
                    finalClErrorEpochsDropOut = epoch - NUMVALID
                    ClErrorEarlyStoppedDropOut = True

                if not CELossEarlyStopped and validationIncreasing(validationLoss[epoch-NUMVALID:epoch]):
                    # Only epochs up till finalCELossEpochs should be considered
                    finalCELossEpochs = epoch - NUMVALID
                    CELossEarlyStopped = True

                if not ClErrorEarlyStopped and validationIncreasing(validationClassificationError[epoch-NUMVALID:epoch]):
                    # Only epochs up till finalClErrorEpochs should be considered
                    finalClErrorEpochs = epoch - NUMVALID
                    ClErrorEarlyStopped = True

        print("Stopping point for ce loss dropout: %f", finalCELossEpochsDropOut)
        print("Stopping point for classification error dropout: %f", finalClErrorEpochsDropOut)
        print("Stopping point for ce loss: %f", finalCELossEpochs)
        print("Stopping point for classification error: %f", finalClErrorEpochs)

        #image generation
        #Dropout
        plt.close('all')
        indicesESD = [int((25.0*finalClErrorEpochsDropOut)/100.0),finalClErrorEpochsDropOut]
        print(indicesESD)
        imageIndex = 25
        for i in indicesESD:
            weights1 = np.array(trainingHiddenLayerWeightsDropOut[i][:,0:100])
            print(weights1.shape)
            weights1 = np.reshape(weights1,[28,28,100])

            figure, ax = plt.subplots(10,10,sharex='col',sharey='row')

            for i in range(10):
                for j in range(10):
                    ax[i,j].imshow(weights1[:,:,10*i+j],cmap=plt.cm.gray,interpolation="nearest")
                    ax[i,j].axis('off')
            plt.axis('off')
            plt.show()
            figure.savefig("DropOut_Weights_%f.png" % imageIndex)
            imageIndex *=4

        #No Dropout
        plt.close('all')
        imageIndex = 25
        indices = [int((25.0*finalClErrorEpochs)/100.0),finalClErrorEpochs]
        print(indices)
        for i in indices:
            weights1 = np.array(trainingHiddenLayerWeights[i][:,0:100])
            print(weights1.shape)
            weights1 = np.reshape(weights1,[28,28,100])

            figure, ax = plt.subplots(10,10,sharex='col',sharey='row')

            for i in range(10):
                for j in range(10):
                    ax[i,j].imshow(weights1[:,:,10*i+j],cmap=plt.cm.gray,interpolation="nearest")
                    ax[i,j].axis('off')
            plt.axis('off')
            plt.show()
            figure.savefig("No_DropOut_Weights_%f.png" % imageIndex)
            imageIndex *=4

       
        print("Dropout")
        print("Training Classification Error: %f", trainingClassificationErrorDropOut[epoch])
        print("Validation Classification Error: %f", validationClassificationErrorDropOut[epoch])
        print("Test Classification Error: %f", testClassificationErrorDropOut[epoch])
        print("-----------------------------------------------------------------------------------------")
        print("No Dropout")
        print("Training Classification Error: %f", trainingClassificationError[epoch])
        print("Validation Classification Error: %f", validationClassificationError[epoch])
        print("Test Classification Error: %f", testClassificationError[epoch])


        # Plot the training, validation and test classification error for best learning rate
        fig = plt.figure(0)
        plt.plot(range(epochs), trainingClassificationErrorDropOut, c='m', label='Training with Dropout')
        #plt.plot(range(epochs), trainingClassificationError, c='c', label='Training without Dropout')
        plt.legend()
        plt.title("Training Classification Error vs no. of epochs for learning rate: %f"%learningRate)
        plt.xlabel("Number of epochs")
        plt.ylabel("Classification Error (%)")
        fig.savefig("part_3_2_TrainingClassificationError_Dropout.png")

          # Plot the training, validation and test classification error for best learning rate
        fig = plt.figure(1)
        plt.plot(range(epochs), validationClassificationErrorDropOut, c='m', label='Validation with Dropout')
        #plt.plot(range(epochs), validationClassificationError, c='c', label='Validation without Dropout')
        plt.legend()
        plt.title("Validation Classification Error vs no. of epochs for learning rate: %f"%learningRate)
        plt.xlabel("Number of epochs")
        plt.ylabel("Classification Error (%)")
        fig.savefig("part_3_2_ValidationClassificationError_Dropout.png")

