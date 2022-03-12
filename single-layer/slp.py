from helper_functions import *

def slp_test(biased_X_test, weight_matrix):

    """
    Computes and returns the outputs for the test data set

    Arguments:

    biased_X_test -- one sample of the test dataset

    Y_test_target -- actual output value of the above mentioned sample

    weight_matrix -- weight matrix associated between input and output layer

    
    """

    output = np.dot(biased_X_test, weight_matrix.T).astype('float32')
    return output


def slp(biased_X_train, Y_train_target, biased_X_test, Y_test_target, epochs: int, learning_rate: float):

    """
    Calculation is based on Input layer and Ouput layer only.

    Based on the values of hyperparameters like epochs and learning rate, the weights in the weight matrix are updated using the perceptron learning algorithm.

    """

    train_accuracy_matrix = list()
    test_accuracy_matrix = list()

    weight_matrix = initialize_weight_matrix(10, 784)

    for epoch in range(0, epochs):
        train_pred_list = list()
        test_pred_list = list()

        for row in range(0, biased_X_train.shape[0]):
            output = np.dot(biased_X_train[row], weight_matrix.T).astype('float32')
            Y_pred = np.where(output<=0, 0, 1)
            train_prediction = np.argmax(Y_pred)
            train_pred_list.append(train_prediction==np.argmax(Y_train_target[row]))

            if(row < biased_X_test.shape[0]):
                    test_output = slp_test(biased_X_test[row], weight_matrix)
                    test_prediction = np.argmax(test_output)
                    test_pred_list.append(test_prediction==np.argmax(Y_test_target[row]))
            if(train_prediction==np.argmax(Y_train_target[row])):
                continue
            else:
                dW = learning_rate * reshape_matrix(Y_train_target[row]-Y_pred, (10,1)) * reshape_matrix(biased_X_train[row],(1,785))
                weight_matrix = weight_matrix + dW

        test_accuracy = np.mean(test_pred_list)
        train_accuracy = np.mean(train_pred_list) 

        print("Epoch : ", epoch)
        print("Training accuracy : ", train_accuracy)
        print("Test accuracy : ", test_accuracy)

        train_accuracy_matrix.append([epoch, train_accuracy])
        test_accuracy_matrix.append([epoch, test_accuracy])

    return train_accuracy_matrix, test_accuracy_matrix




            

