# Code Revision and Review Project
We are a group of 2 people, implementing clean coding techniques to an already existing
open-source code. The code we have chosen is [“MNIST-Digit-classification using
Perceptron Learning algorithm”](https://github.com/ajinsh/MNIST-Digit-classification) which is a 3 years old code written in Python. 

In this project, we have refactored the whole code with broadly the following objectives:

- Clean code 
- Meaningful Naming convention
- Comments
- Code Formatting
- Function based structure
- Project Files Structure

## MNIST-Digit-classification using Perceptron Learning algorithm
### Overview
MNIST data set is a collection of 70,000 monochrome images of numeric digits, where each image is of the resolution 28X28. The data set has 70000 rows, where each row represents an image(or you may call it sample) and 785 columns in which 784 columns represent the value of the corresponding pixel and the last one column represents the digit in the corresponding image. We use this data set for training perceptron models to classify the images.

Below is an example of how the samples look when represented as images:

![MNIST Dataset](https://knowm.org/wp-content/uploads/Screen-Shot-2015-08-14-at-2.44.57-PM.png)

### Perceptron Models

There are broadly two types of models that use perceptron learning algorithm, which we will implementing in this project and those are:

- Single Layer Perceptron Network
- Multi Layer Perceptron Network

**Single Layer Perceptron Network**

In this type of network there are only two layers considered. One is the input layer and the second layer is output layer. A weight matrix is initialised with random values in the range of [-0.05, 0.05]. These weights act as the weighted edges that connect the input layer to the output layer making it a fully connected graph. 

Now, based on the values of hyperparameters like *epochs* and *learning rate*, the weights in the weight matrix are updated using the perceptron learning algorithm. 

   **The perceptron Learning Algorithm:**

   - At each output compute 

       - $\sum_{i=0}^{n} w_i x_i$

   - The output with highest value is the prediction
   - If correct, do nothing and continue to the next sample. If incorrect, update the weights with the following formula

       - $w_i \leftarrow w_i +  \Delta w_i$

           where $\Delta w_i = \eta (t^k - y^k) x_i^k$

We will follow the above described algorithm for each sample and update the weights accordingly and this will be iteratively done for mentioned number of `epochs` times.

**Multi Layer Perceptron Network**

One of the main issues with single layer perceptron is it can only solve linearly separable problems, which is what led to the Multi Layer Perceptron Network that uses famous Back Propagation Algorithm.

In this type of network, there are multiple layers in between input and output layer which are called hidden layers. Each hidden layer has a fixed number of units called `hidden units`, this is a hyperparameter and hence can be fine tuned outside the algorithm. 

Now based on the number of hidden layers and hidden units in each hidden weight matrices are initialised with random values of weights in the range of [-0.05, 0.05]. 

In this project we have implemented a neural network that has one hidden layer between the input and output layer. 

   **The Back Propagation Algorithm**

   - Propogate the input forward

       - Input x to the network and compute the activation of *h<sub>j</sub>* of each hidden unit *j*

       - Compute the activation *o<sub>k</sub>* of each output unit *k*
    
   - Calculate Error Terms

       - For each output unit *k*, calculate error term $\delta_k$:

           - $\delta_k\leftarrow o_k (1-o_k)(t_k-o_k)$

       - For each hidden unit j, calculate error term $\delta_j$:

           - $\delta_j\leftarrow h_j (1-h_j)(\sum_{k\in output\,units} w_{kj} \delta_k)$
    
   - Update Weights

       - Hidden to Output Layer: For each weight $w_{kj}$

           - $w_{kj} \leftarrow w_{kj} + \Delta w_{kj}$, 
                                
                where $\Delta w_{kj} = \eta \delta_k h_j$

       - Input to Hidden Layer: For each weight $w_{ji}$

           - $w_{ji} \leftarrow w_{ji} + \Delta w_{ji}$

                where $\Delta w_{ji} = \eta\delta_j x_i$ 
                        
We will perform the above described algorithm iteratively until a stopping condition is satisfied, which in out case is the number of `epochs`. 

To avoid falling into a local minimum of the loss function, we also use a hyperparameter called momentum($\alpha$). 

The weight updates with momentum will go as follows:

   - $\Delta w_{kj}^t = \eta \delta_k h_j + \alpha \Delta w_{kj}^{t-1}$ (hidden to output)

   - $\Delta w_{ji}^t = \eta \delta_j x_i + \alpha \Delta w_{ji}^{t-1}$ (input to hidden)

## Instructions for running the code

 - First unzip the `MNIST_Dataset.zip` file in each folder(`single_layer` and `multi_layer`).

 - Run the `main.py` file in each folder on your local machine. 

 - You may change the hyperparameters like `epochs`, `learning_rate`, `momentum`, `hidden_units` according to your need in the `main.py` file.