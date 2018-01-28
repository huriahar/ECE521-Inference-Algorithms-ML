from __future__ import print_function
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import part1

def calculate_classification_error(prediction, target):
    # error !!!!!: doesn't calculate the error correctly.. i am running into problems comparing the 
    # original target vector and predicition vector
    index=0
    error = 0.0
    for ele in prediction:
        if index == 0:
            print(ele)
            print(target[index])
        if tf.equal(ele, target[index]):
            error += 1#./ target.shape[0]
        index+=1
    return error

def KNearestNeighbours (distances, k):
    top_k_vals, top_k_indices = tf.nn.top_k(tf.negative(distances), k=k)
    return top_k_indices

def prediction(X_train, Y_train, X_test, Y_test, K):
    #calculate the euclidean distances between test_input and test_train
    distances = part1.euclideanDistance(X_test, X_train)
    
    N1 = distances.get_shape().as_list()[0]

    #calculate the K nearest neighbours for the distances matrix
    top_k_neighbours = KNearestNeighbours(distances, K)

    # Consolidate the best target labels for the K nearest neighbours for the input/training targets
    best_labels = tf.gather(Y_train, top_k_neighbours)

    #just for debugging
    print('neighbure total')
    print(top_k_neighbours.get_shape().as_list())
    print('best label total')
    print(best_labels.get_shape().as_list())

    #creating a list to save the predictions
    prediction = []
    print(N1)

    # Note: for loop is required since the function tf.unique_with_counts accepts only 1-D tensor
    # input. Hence, we have to loop over each input datapoint and calculate its prediction
    for i in range(N1):

        #just for debugging
        #top_k_neighbours_i = tf.gather(top_k_neighbours, i)
        # if i == 0 or i == N1-1:
        #     print(top_k_neighbours_i.get_shape().as_list())

        #get the best label id(s) for input data i
        best_labels_ids = tf.gather(best_labels, i)

        #just for debugging
        # if i == 0 or i == N1-1:
        #     print(best_labels_ids.get_shape().as_list())
        # if K == 1:
        #      z = tf.squeeze(best_labels_ids)
        #      print(z.get_shape().as_list())
        #      #prediction.append(z)
        # else:
        #     pass

        #count the frequency of each class and finally pick the class with the highest frequency
        #as the prediction value z
        values, indices, counts = tf.unique_with_counts(best_labels_ids)
        max_count_index = tf.argmax(counts)
        z= tf.gather(values, max_count_index)

        #just for debugging
        #print(z.get_shape().as_list())
        
        prediction.append(z)
    
    #just for debugging
    print(len(prediction))

    return prediction
             

def data_segmentation(data_path,target_path,task):
    # task = 0 >>  select the name ID targets for face recognition task
    # task = 1 >>  select the gender ID targets for gender recognition task

    data = np.load(data_path)/255
    data = np.reshape(data, [-1,32*32])
    target = np.load(target_path)

    np.random.seed(45689)
    rnd_idx = np.arange(np.shape(data)[0])
    np.random.shuffle(rnd_idx)

    trBatch = int(0.8*len(rnd_idx))
    validBatch = int(0.1*len(rnd_idx))

    trainData, validData, testData = data[rnd_idx[1:trBatch],:], data[rnd_idx[trBatch+1:trBatch+validBatch],:],data[rnd_idx[trBatch+validBatch+1:-1],:]

    trainTarget, validTarget, testTarget = target[rnd_idx[1:trBatch],task], target[rnd_idx[trBatch+1:trBatch+validBatch],task],target[rnd_idx[trBatch+validBatch+1:-1],task]

    return trainData, validData, testData, trainTarget, validTarget, testTarget

if __name__ == "__main__":
    
    with tf.Session() as sess:

        task = 0  #0 - celebrity classification, #1- Gender classification
        K = 3
        trainData, validData, testData, trainTarget, validTarget, testTarget = data_segmentation('./data.npy','./target.npy',task)
        
        X_train = tf.placeholder(tf.int32, trainData.shape)
        X_valid = tf.placeholder(tf.int32, validData.shape)
        X_test = tf.placeholder(tf.int32, testData.shape)

        Y_train = tf.placeholder(tf.int32, trainTarget.shape)
        Y_valid = tf.placeholder(tf.int32, validTarget.shape)
        Y_test = tf.placeholder(tf.int32, testTarget.shape)

        #Prediction vectors for training, validation and test data and targets
        PredictionTrain = prediction(X_train, Y_train, X_train, Y_train,K)
        PredictionValid = prediction(X_train, Y_train, X_valid, Y_valid,K)
        PredictionTest = prediction(X_train, Y_train, X_test, Y_test,K)

        init = tf.global_variables_initializer()
        sess.run(init)

        # Just for Debugging
        print(trainData.shape)
        print(validData.shape)
        print(testData.shape)
        print(trainTarget.shape)
        print(validTarget.shape)
        print(testTarget.shape)

        #Predictions
        print("Train Prediction:", sess.run(PredictionTrain, {X_train: trainData, Y_train: trainTarget}))
        print("Valid Prediction:", sess.run(PredictionValid, {X_train: trainData, Y_train: trainTarget, X_valid: validData, Y_valid: validTarget}))
        print("Test Prediction:",  sess.run(PredictionTest, {X_train: trainData, Y_train: trainTarget, X_test: testData, Y_test: testTarget}))
        
        #original targets
        print("Train target:")
        print(trainTarget)
        print("Valid target:")
        print(validTarget)
        print("Test target:")
        print(testTarget)

        #function calls for calculating error
        #errorTrain = calculate_classification_error(PredictionTrain,trainTarget)
        #errorValid = calculate_classification_error(PredictionValid,validTarget)
        #errorTest = calculate_classification_error(PredictionTest,testTarget)
        

        #Just for Debugging
        # print("Training Error: ", errorTrain)
        #print("Validation Error:", errorValid)
        #print("Test Error:", errorTest)
        # print(trainTarget)
        #print(Prediction)
        #plt.close('all')
        #plt.plot(trainTarget)
        #plt.show()



    # Copied from Part 2: Maybe useful for Part 3.2 and 3.3
    # Ks = [1, 3, 5, 50]
    # with tf.Session() as sess:
    #     for K in Ks:
    #         print("K=%d:"%K)
    #         MSE_train = calculate_MSE(X_train, Y_train, X_train, Y_train, K)
    #         MSE_valid = calculate_MSE(X_train, Y_train, X_valid, Y_valid, K)
    #         MSE_test = calculate_MSE(X_train, Y_train, X_test, Y_test, K)
    #         init = tf.global_variables_initializer()
    #         sess.run(init)
    #         print("training MSE loss: ", sess.run(MSE_train, {X_train: trainData, Y_train: trainTarget}))
    #         print("valid MSE loss: ", sess.run(MSE_valid, {X_train: trainData, Y_train: trainTarget, X_valid: validData, Y_valid: validTarget}))
    #         print("test MSE loss: ", sess.run(MSE_test, {X_train: trainData, Y_train: trainTarget, X_test: testData, Y_test: testTarget}))
    # New_Data = np.linspace(0.0 , 11.0 , num =1000) [:, np.newaxis]
    # New_Target = np.sin( New_Data ) + 0.1 * np.power( New_Data , 2) + 0.5 * np.random.randn(1000 , 1)

    # X_new_test = tf.placeholder(tf.float32, [1000,1])
    # Deuc = part1.euclideanDistance(X_new_test, X_train)
    # Predictions = dict.fromkeys(Ks)
    # with tf.Session() as sess:
    #     for K in Ks:
    #         responsibility = KNearestNeighbours(Deuc, K)
    #         Y_head = tf.matmul(responsibility, Y_train)
    #         Y_head = tf.reduce_sum(Y_head, axis=1)
    #         init = tf.global_variables_initializer()
    #         sess.run(init)
    #         Predictions[K] = sess.run(Y_head, {X_train: trainData, X_new_test: New_Data, Y_train: trainTarget})
    # plt.close('all')
    # plt.scatter(New_Data, New_Target, marker='.', label='True Target')
    # plt.xlabel('input Data (X)')
    # plt.ylabel('predictions (Y head) and target values (Y)')
    # K_max = float(max(Ks))
    # for K in Ks:
    #     plt.plot(New_Data, Predictions[K], c=(np.random.uniform(0.0,1.0) , np.random.uniform(0.0,1.0) , np.random.uniform(0.0,1.0) ), label=("Predictions when K=%d" % K))
    # plt.legend()
    # plt.show()
        
