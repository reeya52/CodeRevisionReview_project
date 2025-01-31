import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from helper_functions import *
from slp import slp

def plot_accuracy_graph(train_accuracy_matrix, test_accuracy_matrix):

    """
    This takes in the training accuracy and test accuracy as arguments and 
    plots the data onto an X-Y plane

    train_accuracy_matrix: all the training accuracies corresponding to each epoch

    test_accuracy_matirx: all the test accuracies corresponding to each epoch

    """

    train_epoch = train_accuracy_matrix[0]
    train_accuracy = train_accuracy_matrix[1]
    plt.plot(train_epoch, train_accuracy, label = "Training Accuracy", linestyle = 'dashed')

    test_epoch = test_accuracy_matrix[0]
    test_accuracy = test_accuracy_matrix[1]
    plt.plot(test_epoch, test_accuracy, label = "Test Accuracy", linestyle = 'dashed')

    plt.xlabel('epochs')
    plt.ylabel('accuracy')

    plt.title("epochs = 50 learning rate = 0.1")

    plt.legend()
    plt.show()

def main():

    """
    Read the data in the MNIST csv files using pandas read_csv and 
    convert each of them into numpy arrays

    Split both train and test numpy arrays to corresponding Y and X numpy arrays

    Normalize the X_train and X_test as the values range between 0 to 255

    add bias to the X_train and X_test

    One hot encode the Y_train and Y_test to make them compatible 
    to be compared to the output values of the neural network

    Train the model using the data based on different hyperparameters

    """

    data_train = pd.read_csv('MNIST_Dataset/mnist_train.csv')
    data_test= pd.read_csv('MNIST_Dataset/mnist_test.csv')

    data_train=np.array(data_train)
    data_test = np.array(data_test)

    Y_train = data_train[:,0]
    Y_test = data_test[:,0]

    X_train = data_train[:,1:]
    X_train = normalize_array(X_train)
    biased_X_train= add_bias(X_train)
    
    X_test = data_test[:,1:]
    X_test = normalize_array(X_test)
    biased_X_test = add_bias(X_test)

    Y_train_target = one_hot_encode_array(Y_train)
    Y_test_target = one_hot_encode_array(Y_test)
    
    train_accuracy_matrix, test_accuracy_matrix = slp(biased_X_train, Y_train_target, biased_X_test, Y_test_target, 50, 0.1)
    plot_accuracy_graph(np.array(train_accuracy_matrix).T,np.array(test_accuracy_matrix).T)


if __name__ == "__main__":
    main()