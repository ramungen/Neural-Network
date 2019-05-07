import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import activation_funcs
from activation_funcs import *

class neural_network:

    def __init__(self, network_structure, activation_functions):
        ## assert that everything is okay 
        assert(len(network_structure) == len(activation_functions))
        assert(len(network_structure))
        assert(len(activation_functions))

        self.structure = network_structure
        self.activation_funcs = activation_functions
        self.length = len(network_structure)
        self.params = None

    ## initialize all the weights and biases to random values/zeros 
    def initialize_parameters(self, X):
        params = {}

        n_prev = X.shape[0]

        for i in range(self.length):     
            n_cur = self.structure[i]

            ## use He initialization 
            params['W' + str(i + 1)] = np.random.randn(n_cur, n_prev) * np.sqrt(2/n_prev)
            params['b' + str(i + 1)] = np.zeros((n_cur, 1)) 
            n_prev = n_cur

        return params

    def forward_propagate(self, X):
        cache = {}
        cache['A0'] = X
        cache['Z0'] = 0

        for i in range(self.length):

            W = self.params['W' + str(i + 1)]
            b = self.params['b' + str(i + 1)]

            A_prev = cache['A' + str(i)]
            Z = np.dot(W, A_prev) + b
            ## change later to an arbitrary activation function
            A_curr = self.activate(Z, i)
            #A_curr = sigmoid(Z) if (i == self.length - 1) else np.tanh(Z) 
            
            cache['Z' + str(i + 1)] = Z
            cache['A' + str(i + 1)] = A_curr


        return cache

    ## change later !!!
    def activate(self, Z, i):
        assert(self.activation_funcs[i] in {'sigmoid', 'tanh', 'relu', 'softmax'})

        if(self.activation_funcs[i] == 'sigmoid'):
            return sigmoid(Z)
        elif (self.activation_funcs[i] == 'tanh'):
            return np.tanh(Z)
        elif (self.activation_funcs[i] == 'relu'):
            return ReLU(Z)
        elif (self.activation_funcs[i] == 'softmax'):
            return softmax(Z)
        else:
            print('activation function {} is not supported, using relu as a default'.format(self.activation_funcs[i]))
            return ReLU(Z)

        #compute the cost
    def compute_cost(self, Y_hat, Y):
        assert(Y_hat.shape == Y.shape)
        ## avoid infinity
        loss = - Y * np.log(Y_hat)
        #loss = - (Y * np.log(Y_hat + 0.001) + (1 - Y) * np.log(1 - Y_hat + 0.001))
        #cost = np.sum(loss) / m
        cost = np.mean(loss)
        return(cost)
    

    def backward_propagate(self, Y, X, cache):
        derivs = {}

        m = X.shape[1]
        dZ = cache['A' + str(self.length)] - Y

        for i in range(self.length, 0, -1):

            A_prev = cache['A' + str(i - 1)]
            Z = cache['Z' + str(i - 1)]

            W = self.params['W' + str(i)]

            dW = np.dot(dZ, A_prev.T) / m
            db = np.sum(dZ, axis = 1, keepdims = True) / m
            if(i != 1):
                #dZ = np.dot(W.T, dZ) * tanh_deriv(Z)
                ## -2 not -1 because index starts at 0
                dZ = np.dot(W.T, dZ) * self.activation_deriv(Z, i - 2)
            derivs['dW' + str(i)] = dW
            derivs['db' + str(i)] = db

        return derivs


    ## softmax derivative not supported yet
    def activation_deriv(self, Z, i):
        assert(self.activation_funcs[i] in {'sigmoid', 'tanh', 'relu', 'softmax'})

        if(self.activation_funcs[i] == 'sigmoid'):
            return sigmoid_deriv(Z)
        if (self.activation_funcs[i] == 'tanh'):
            return tanh_deriv(Z)
        elif (self.activation_funcs[i] == 'relu'):
            return ReLU_deriv(Z)
        elif (self.activation_funcs[i] == 'softmax'):
            return softmax_deriv(Z)
        else:
            print('activation function {} is not supported, using relu as a default'.format(self.activation_funcs[i]))
            return ReLU_deriv(Z)


    ## parameter update step in momentum
    def __update_parameters_momentum(self, derivs, derivs_M1, learning_rate, beta,current_step, bias_correction = True):
        for i in range(self.length):
            derivs_M1['dW' + str(i + 1)] = beta * derivs_M1['dW' + str(i + 1)] + (1-beta) * derivs['dW' + str(i+1)]
            derivs_M1['db' + str(i + 1)] = beta * derivs_M1['db' + str(i + 1)] + (1-beta) * derivs['db' + str(i+1)]

            adjusted_learning_rate = learning_rate

            if bias_correction:
                adjusted_learning_rate = learning_rate / (1 - beta ** current_step) 

            self.params['W' + str(i + 1)] -= adjusted_learning_rate * derivs_M1['dW' + str(i+1)]
            self.params['b' + str(i + 1)] -= adjusted_learning_rate * derivs_M1['db' + str(i+1)]

            

    ## parameter update step in RMSprop
    def __update_parameters_RMSprop(self, derivs, derivs_M2, learning_rate, beta2, current_step, bias_correction = True):
        for i in range(self.length):
            epsilon = 1e-8

            ## derivs_M2 <- second moment estimates of dW and db
            derivs_M2['dW' + str(i+1)] = beta2 * derivs_M2['dW' + str(i+1)] + (1-beta2) * (derivs['dW' + str(i+1)] ** 2)
            derivs_M2['db' + str(i+1)] = beta2 * derivs_M2['db' + str(i+1)] + (1-beta2) * (derivs['db' + str(i+1)] ** 2)

            ## add a small number to denominator for numerical stability
            dW_RMS_corrected = derivs['dW' + str(i+1)] / np.sqrt(derivs_M2['dW' + str(i+1)] + epsilon**2)
            db_RMS_corrected = derivs['db' + str(i+1)] / np.sqrt(derivs_M2['db' + str(i+1)] + epsilon**2)

            adjusted_learning_rate = learning_rate
            if bias_correction:
                # because 1/(1/a) = a
                adjusted_learning_rate = learning_rate * np.sqrt((1 - beta2 ** current_step))
            
            self.params['W' + str(i + 1)] -= adjusted_learning_rate * dW_RMS_corrected
            self.params['b' + str(i + 1)] -= adjusted_learning_rate * db_RMS_corrected


    def __update_parameters_adam(self, derivs, derivs_M1, derivs_M2, learning_rate, beta1, beta2, current_step, bias_correction = True):

        epsilon = 1e-8
        for i in range(self.length):
    
            derivs_M1['dW' + str(i + 1)] = beta1 * derivs_M1['dW' + str(i + 1)] + (1-beta1) * derivs['dW' + str(i+1)]
            derivs_M1['db' + str(i + 1)] = beta1 * derivs_M1['db' + str(i + 1)] + (1-beta1) * derivs['db' + str(i+1)]
            
            ## dW2_WMA <- second moment estimate of dW
            ## db2_WMA <- second moment estimate of db
            derivs_M2['dW' + str(i+1)] = beta2 * derivs_M2['dW' + str(i+1)] + (1-beta2) * (derivs['dW' + str(i+1)] ** 2)
            derivs_M2['db' + str(i+1)] = beta2 * derivs_M2['db' + str(i+1)] + (1-beta2) * (derivs['db' + str(i+1)] ** 2)

            if bias_correction:
                dW_WMA_unbiased = derivs_M1['dW' + str(i + 1)] / (1 - beta1 ** current_step) 
                db_WMA_unbiased = derivs_M1['db' + str(i + 1)] / (1 - beta1 ** current_step)

                dW2_WMA_unbiased = derivs_M2['dW' + str(i+1)] / (1 - beta2 ** current_step)
                db2_WMA_unbiased = derivs_M2['db' + str(i+1)] / (1 - beta2 ** current_step)
            else:
                dW_WMA_unbiased = derivs_M1['dW' + str(i + 1)] 
                db_WMA_unbiased = derivs_M1['db' + str(i + 1)]

                dW2_WMA_unbiased = derivs_M2['dW' + str(i+1)]
                db2_WMA_unbiased = derivs_M2['db' + str(i+1)]


            ## add a small number to denominator for numerical stability
            
            self.params['W' + str(i + 1)] -= (learning_rate * dW_WMA_unbiased / np.sqrt(dW2_WMA_unbiased + epsilon**2) )
            self.params['b' + str(i + 1)] -= (learning_rate * db_WMA_unbiased / np.sqrt(db2_WMA_unbiased + epsilon**2) )

    ## update the parameters
    def __update_parameters_stochastic(self, derivs, learning_rate):

        for i in range(self.length):
            self.params['W' + str(i + 1)] -= learning_rate * derivs['dW' + str(i + 1)]
            self.params['b' + str(i + 1)] -= learning_rate * derivs['db' + str(i + 1)]


    def __initialize_adam(self):
    
        derivs_M1 = {}
        derivs_M2 = {}
        for i in range(self.length):
            derivs_M1['dW' + str(i+1)] = np.zeros(self.params['W' + str(i+1)].shape)
            derivs_M1['db' + str(i+1)] = np.zeros(self.params['b' + str(i+1)].shape)

            derivs_M2['dW' + str(i+1)] = np.zeros(self.params['W' + str(i+1)].shape)
            derivs_M2['db' + str(i+1)] = np.zeros(self.params['b' + str(i+1)].shape)

        return derivs_M1, derivs_M2

    def __initialize_momentum(self):
    
        derivs_M1 = {}
        for i in range(self.length):
            derivs_M1['dW' + str(i+1)] = np.zeros(self.params['W' + str(i+1)].shape)
            derivs_M1['db' + str(i+1)] = np.zeros(self.params['b' + str(i+1)].shape)

        return derivs_M1

    def __initialize_RMSprop(self):

        derivs_M2 = {}
        for i in range(self.length):
            derivs_M2['dW' + str(i+1)] = np.zeros(self.params['W' + str(i+1)].shape)
            derivs_M2['db' + str(i+1)] = np.zeros(self.params['b' + str(i+1)].shape)

        return derivs_M2

    def __perform_gradient_descent(self, algorithm, learning_rate, data, labels, num_epochs, 
    minibatch_size, bias_correction = None, beta1 = None, beta2 = None, print_cost = False, plot_cost = True):
        assert(self.params == None)
        assert(labels.shape[0] == self.structure[-1])
        self.params = self.initialize_parameters(data)
        ## only those algorithms are supported as of now
        assert(algorithm in {'stochastic_GD', 'Adam', 'RMSprop', 'momentum'})
        costs = []

        if algorithm == 'Adam':
            derivs_M1, derivs_M2 = self.__initialize_adam()
        elif algorithm == 'RMSprop':
            derivs_M2 = self.__initialize_RMSprop()
        elif algorithm == 'momentum':
            derivs_M1 = self.__initialize_momentum()

        current_step = 0
        for epoch in range(1, num_epochs+1):
            
            minibatches = self.__get_shuffled_minibatches(data, labels, minibatch_size)

            for minibatch in minibatches:

                
                current_step += 1
                ## get the minibatches 
                (X_batch, Y_batch) = minibatch

                cache = self.forward_propagate(X_batch)

                Y_hat_batch = cache['A' + str(self.length)]
                
                cost = self.compute_cost(Y_hat_batch, Y_batch)

                derivs = self.backward_propagate(Y_batch, X_batch, cache)
                if algorithm == 'stochastic_GD':
                    self.__update_parameters_stochastic(derivs, learning_rate)
                elif algorithm == 'Adam':
                    self.__update_parameters_adam(derivs, derivs_M1, derivs_M2, 
                    learning_rate, beta1, beta2, current_step, bias_correction)
                elif algorithm == 'RMSprop':
                    self.__update_parameters_RMSprop(derivs, derivs_M2, learning_rate, 
                    beta2, current_step, bias_correction)
                elif algorithm == 'momentum':
                    self.__update_parameters_momentum(derivs, derivs_M1, learning_rate, beta1, current_step, bias_correction)


            if print_cost:
                    costs.append(cost)
                    print('cost after {} epochs: {}'.format(epoch, cost))

        # plot the costs
        plt.plot(costs)
        plt.ylabel('cost')
        plt.xlabel('# of iterations')
        plt.show()
        return self.params

    # after the learning phase
    def predict(self, data):
        cache = self.forward_propagate(data)
        Y_hat = cache['A' + str(self.length)]

        ## choose the number with highest probability
        predictions = np.asmatrix(np.argmax(Y_hat, axis = 0)).T
        return predictions

    # after the learning phase
    def prediction_accuracy(self, data, correct):
        predictions = self.predict(data)
        correct_percentage = np.round_((np.mean(predictions == correct)) * 100, decimals = 2)
        print('correctly identified {}% digits'.format(correct_percentage))


    ## permutes the dataset and divides it into minibatches
    def __get_shuffled_minibatches(self, X, Y, minibatch_size):
        m = X.shape[1]
        mini_batches = []

        permutation = list(np.random.permutation(m))
        X_permuted = X[:, permutation]
        Y_permuted = Y[:, permutation]


        num_batches = m // minibatch_size

        for i in range(num_batches):
            X_batch = X_permuted[:, minibatch_size * i: minibatch_size * (i+1)]
            Y_batch = Y_permuted[:, minibatch_size * i: minibatch_size * (i+1)]
            mini_batches.append((X_batch, Y_batch))


        if X.shape[1] % minibatch_size != 0:
            X_batch_last = X_permuted[:, num_batches * minibatch_size:]
            Y_batch_last = Y_permuted[:, num_batches * minibatch_size:]
            mini_batches.append((X_batch_last, Y_batch_last))

        return mini_batches


    def train_stochastic_GD(self, data, labels, learning_rate = 0.1, num_epochs = 10, minibatch_size = 64, 
    print_cost = True, plot_cost = True):
        assert(labels.shape[0] == self.structure[-1])

        return self.__perform_gradient_descent(algorithm = 'stochastic_GD', data = data, labels = labels, 
        learning_rate = learning_rate, num_epochs = num_epochs, minibatch_size = minibatch_size, 
        print_cost = print_cost, plot_cost = plot_cost)


    def train_adam(self, data, labels, learning_rate = 0.1, num_epochs = 10, minibatch_size = 64, beta1 = 0.9, beta2 = 0.999, 
    bias_correction = True, print_cost = True, plot_cost = True):
        assert(0 < beta1 < 1)
        assert(0 < beta2 < 1)
        assert(labels.shape[0] == self.structure[-1])

        return self.__perform_gradient_descent(algorithm = 'Adam', data = data, labels = labels, 
        learning_rate = learning_rate, num_epochs = num_epochs, bias_correction = bias_correction,
        minibatch_size = minibatch_size, print_cost = print_cost, plot_cost = plot_cost, beta1 = beta1, beta2 = beta2)

    
    def train_RMSprop(self, data, labels, learning_rate = 0.1, num_epochs = 10, minibatch_size = 64, beta2 = 0.999,
    bias_correction = True, print_cost = True, plot_cost = True):
        assert(labels.shape[0] == self.structure[-1])
        assert(0 < beta2 < 1)

        return self.__perform_gradient_descent(algorithm = 'RMSprop', data = data, labels = labels, 
        learning_rate = learning_rate, num_epochs = num_epochs, bias_correction = bias_correction,
        minibatch_size = minibatch_size, print_cost = print_cost, plot_cost = plot_cost, beta2 = beta2)

    def train_momentum(self, data, labels, learning_rate = 0.1, num_epochs = 10, minibatch_size = 64, 
    beta = 0.9, bias_correction = True, print_cost = True, plot_cost = True):
        assert(labels.shape[0] == self.structure[-1])
        assert(0 < beta < 1)

        return self.__perform_gradient_descent(algorithm = 'momentum', data = data, labels = labels, 
        learning_rate = learning_rate, num_epochs = num_epochs, bias_correction = bias_correction,
        minibatch_size = minibatch_size, print_cost = print_cost, plot_cost = plot_cost, beta1  = beta)