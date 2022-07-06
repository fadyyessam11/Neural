#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from sklearn import metrics


# In[2]:


# load MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()


def grid_Image(img , row , col ):
    x , y = img.shape
    assert x % row == 0, "28 is Not Divisible by This Number !".format(x, row)
    assert y % col == 0, "28 is Not Divisible by This Number !".format(y, col)
    return (img.reshape ( x //row, row, -1, col)
               .swapaxes(1,2)
               .reshape(-1, row, col))
print(grid_Image(x_test[1] , 7 , 7 ).shape)
grid_Image(x_test[1] , 7 , 7 )

def centroid_function(img):
    feature_vector = []
    for grid in grid_Image(img , 7 , 7 ) :     
        Xc = 0 
        Yc = 0 
        sum = 0
        for i, x in np.ndenumerate(grid):
          sum+= x 
          Xc += x * i[0]
          Yc += x * i[1]  
        if sum != 0 :
            feature_vector.append( Xc/ sum )
            feature_vector.append(Yc/ sum )
        else :
             feature_vector.append(0)
             feature_vector.append(0)  
    return np.array(feature_vector)


# In[3]:


# train_features = [centroid_function(img)  for img in x_train  ]
# print("Feature Extracted From Training Data")
# train_features = np.array(train_features)
# train_features.shape

# test_features = [centroid_function(img)  for img in x_test  ]
# print("Feature Extracted From Test Data")
# test_features = np.array(test_features)
# test_features.shape


# In[4]:


class Layer:
    def init(self):
        self.input = None
        self.output = None

    # Output Y of a layer for a given input X
    def forward_propagation(self, input):
        raise NotImplementedError

    # dE/dX for a given dE/dY and update parameters if any
    def backward_propagation(self, output_error, learning_rate):
        raise NotImplementedError


# In[5]:


# inherit from base class Layer
class FCLayer(Layer):
    # input_size = number of input neurons
    # output_size = number of output neurons
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5

    # Output for a given Input
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    # dE/dW, dE/dB for a given output_error=dE/dY. Returns input_error=dE/dX.
    def backward_propagation(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)
        # dBias = output_error

        # Update Parameters
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return input_error


# In[6]:


# inherit from base class Layer
class ActivationLayer(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    # Activated input
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    # Returns input_error=dE/dX for a given output_error=dE/dY.
    def backward_propagation(self, output_error, learning_rate):
        return self.activation_prime(self.input) * output_error


# In[7]:


# Activation Function 
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative
def sigmoid_prime(x):
    return np.exp(-x) / (1 + np.exp(-x))**2


# In[8]:


# loss function
def mse(y_true, y_pred):
    return np.mean(np.power(y_true-y_pred, 2));

# Derivative
def mse_prime(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.size;


# In[11]:


class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None

    # Add Layer to network
    def add(self, layer):
        self.layers.append(layer)

    # set loss to use
    def use(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    # Predict Output for given Input
    def predict(self, input_data):
        samples = len(input_data)
        result = []

        # Run Network over all Samples
        for i in range(samples):
            
            # forward propagation
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)

        return result

    # Train the Network
    def fit(self, x_train, y_train, epochs, learning_rate):
        
        samples = len(x_train)

        # training loop
        for i in range(epochs):
            err = 0
            for j in range(samples):
                
                # forward propagation
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                # Compute loss
                err += self.loss(y_train[j], output)

                # Backward Propagation
                error = self.loss_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)

            # Average Errors on all samples
            err /= samples
            print('epoch %d/%d   error=%f' % (i+1, epochs, err))
     
            
    def accuracy(self, X, y):
        P = self.predict(X)
        return sum(np.equal(P, np.argmax(y, axis=0))) / y.shape[1]*100
      
    
    def testing(self,x_test, y_test):
        # test the model on the training dataset
        total_correct = 0
        for n in range(len(x_test)):
            y = y_test[n]
            x = x_test[n][:]
            prediction = np.argmax(self.forward_propagation(x,y)['f_X'])
            if (prediction == y):
                total_correct += 1
        print('Accuarcy Test: ',total_correct/len(x_test))
        return total_correct/np.float(len(x_test))


# In[30]:


# load MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# Reshape and Normalize input data
x_train = x_train.reshape(x_train.shape[0], 1, 28*28)
x_train = x_train.astype('float32')
x_train /= 255

# Encode output which is a number in range [0,9] into a vector of size 10
# e.g. number 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
y_train = np_utils.to_categorical(y_train)


x_test = x_test.reshape(x_test.shape[0], 1, 28*28)
x_test = x_test.astype('float32')
x_test /= 255
y_test = np_utils.to_categorical(y_test)

# Network
net = Network()
net.add(FCLayer(28*28, 100))   # input_shape=(1, 28*28) ; output_shape=(1, 100)
net.add(ActivationLayer(sigmoid, sigmoid_prime))

net.add(FCLayer(100, 50))      # input_shape=(1, 100) ; output_shape=(1, 50)
net.add(ActivationLayer(sigmoid, sigmoid_prime))

net.add(FCLayer(50, 10))       # input_shape=(1, 50) ; output_shape=(1, 10)
net.add(ActivationLayer(sigmoid, sigmoid_prime))

# Train on 1000 samples
net.use(mse, mse_prime)
net.fit(x_train[0:1000], y_train[0:1000], epochs=150, learning_rate=0.1)

# Test on 50 samples
out = net.predict(x_test[0:50])


# net.accuracy(x_test[0:5], y_train[0:1000])


print("\n")
print("Predicted Values : ")
print(out, end="\n")
print("True Values : ")
print(y_test[0:5])


# In[31]:


ratio = sum([np.argmax(y) == np.argmax(net.predict(x)) for x, y in zip(x_test, y_test)]) / len(x_test)
error = sum([mse(y, net.predict(x)) for x, y in zip(x_test, y_test)]) / len(x_test)

print('Ratio: %.2f' % ratio)
print('MSE: %.4f' % error)
print(net.accuracy(x_test, y_test))


# In[14]:


predictions = net.predict(x_test)
print(predictions)


# In[15]:


predictions = np.argmax(predictions, axis=1)
print(predictions)


# In[16]:


accuracy = metrics.accuracy_score(y_test, predictions)
accuracy


# In[ ]:




