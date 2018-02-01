from __future__ import print_function
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import part1

def calculateClassificationError(prediction, target):
    diff = tf.subtract(target, prediction)
    incorrectPrediction = tf.count_nonzero(diff)
    N = target.get_shape().as_list()[0]
    accuracy = tf.divide(tf.subtract(N, tf.cast(incorrectPrediction, tf.int32)), N)*100
    return incorrectPrediction, accuracy

def KNearestNeighbours (distances, k):
    top_k_vals, top_k_indices = tf.nn.top_k(tf.negative(distances), k=k)
    return top_k_indices

# Returns a prediction vector for X_test
def prediction(X_train, Y_train, X_test, K):
    # calculate the euclidean distances between test_input and test_train
    distances = part1.euclideanDistance(X_test, X_train)
    
    N1 = distances.get_shape().as_list()[0]
    # Get indices of the K nearest neighbours for the distances matrix
    topKNeighbours = KNearestNeighbours(distances, K)

    # Consolidate the best target labels for the K nearest neighbours for the input/training targets
    bestLabels = tf.gather(Y_train, topKNeighbours)

    #creating a list to save the predictions
    prediction = []

    # Note: for loop is required since the function tf.unique_with_counts accepts only 1-D tensor
    # input. Hence, we have to loop over each input datapoint and calculate its prediction
    for i in range(N1):

        #get the best label id(s) for input data i
        best_labels_ids = tf.gather(bestLabels, i)

        #count the frequency of each class and finally pick the class with the highest frequency
        #as the prediction value z
        values, indices, counts = tf.unique_with_counts(best_labels_ids)
        max_count_index = tf.argmax(counts)
        z = tf.gather(values, max_count_index)
        prediction.append(z)

    prediction = tf.stack(prediction)
    return prediction

def dataSegmentation(data_path, target_path, task):
    # task = 0 >>  select the name ID targets for face recognition task
    # task = 1 >>  select the gender ID targets for gender recognition task

    data = np.load(data_path)/255.0
    data = np.reshape(data, [-1,32*32])
    target = np.load(target_path)

    np.random.seed(45689)
    rnd_idx = np.arange(np.shape(data)[0])
    np.random.shuffle(rnd_idx)

    trBatch = int(0.8*len(rnd_idx))
    validBatch = int(0.1*len(rnd_idx))

    trainData, validData, testData = data[rnd_idx[1:trBatch],:], \
                                     data[rnd_idx[trBatch+1:trBatch+validBatch],:], \
                                     data[rnd_idx[trBatch+validBatch+1:-1],:]

    trainTarget, validTarget, testTarget = target[rnd_idx[1:trBatch],task], \
                                           target[rnd_idx[trBatch+1:trBatch+validBatch],task], \
                                           target[rnd_idx[trBatch+validBatch+1:-1],task]

    return trainData, validData, testData, trainTarget, validTarget, testTarget

if __name__ == "__main__":
    
    with tf.Session() as sess:

        task = 0 #0- celebrity classification, #1- Gender classification
        trainData, validData, testData, trainTarget, validTarget, testTarget = dataSegmentation('data.npy','target.npy', task)
        X_train = tf.placeholder(tf.float32, trainData.shape)
        X_valid = tf.placeholder(tf.float32, validData.shape)
        X_test = tf.placeholder(tf.float32, testData.shape)

        Y_train = tf.placeholder(tf.int32, trainTarget.shape)
        Y_valid = tf.placeholder(tf.int32, validTarget.shape)
        Y_test = tf.placeholder(tf.int32, testTarget.shape)

        #Prediction vectors for training, validation and test data and target
        init = tf.global_variables_initializer()
        sess.run(init)

        Ks = [1, 5, 10, 25, 50, 100, 200]
        for K in Ks:
            print("K=%d:"%K)
            PredictionTrain = prediction(X_train, Y_train, X_train, K)
            PredictionValid= prediction(X_train, Y_train, X_valid, K)
            PredictionTest = prediction(X_train, Y_train, X_test ,K)
            incorTrain, accuracyTrain = calculateClassificationError(PredictionTrain, Y_train)
            invorValid,  accuracyValid = calculateClassificationError(PredictionValid, Y_valid)
            incorTest, accuracyTest = calculateClassificationError(PredictionTest, Y_test)
            print("training accuracy: ", sess.run(accuracyTrain, {X_train: trainData, Y_train: trainTarget}))
            print("valid accuracy: ", sess.run(accuracyValid, {X_train: trainData, Y_train: trainTarget, X_valid: validData, Y_valid: validTarget}))
            print("test accuracy: ", sess.run(accuracyTest, {X_train: trainData, Y_train: trainTarget, X_test: testData, Y_test: testTarget}))
