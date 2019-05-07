import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

def sigmoid(Z):
    return 1/(1 + np.exp(-Z))

def softmax(Z):
    return np.exp(Z)/(np.sum(np.exp(Z), axis = 0))

def sigmoid_deriv(Z):
    return sigmoid(Z) * (1 - sigmoid(Z))

def softmax_deriv(Z):
    return 1.0/(1 + np.exp(-Z))

def ReLU(Z):
    return np.maximum(Z, 0)

def ReLU_deriv(Z):
    deriv = Z
    deriv[deriv<=0] = 0
    deriv[deriv>0] = 1
    return deriv

def tanh_deriv(Z):
    return (1 - np.tanh(Z) ** 2)

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
            params['b' + str(i + 1)] = 0 
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

        m = Y.shape[1]
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


    ## update the parameters
    def update_parameters(self, derivs, learning_rate = 0.01):

        for i in range(self.length):
            self.params['W' + str(i + 1)] -= learning_rate * derivs['dW' + str(i + 1)]
            self.params['b' + str(i + 1)] -= learning_rate * derivs['db' + str(i + 1)]
        


    ## train the neural network
    def train(self, correct, data, learning_rate = 1.0, iterations = 300, print_cost = False):

        assert(self.params == None)
        assert(correct.shape[0] == self.structure[-1])

        self.params = self.initialize_parameters(data)
        Y_hat = None
        costs = []
        #learning phase 
        for i in range(iterations):
            cache = self.forward_propagate(data)

            Y_hat = cache['A' + str(self.length)]

            cost = self.compute_cost(Y_hat, correct)

            derivs = self.backward_propagate(correct, data, cache)

            self.update_parameters(derivs, learning_rate)

            costs.append(cost)
            
            if(print_cost):
                print('cost after {} iterations: {}'.format(i, cost))


        if(print_cost):
            cost = self.compute_cost(Y_hat, correct) 
            print(cost)

        ## plot the costs
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

    def get_shuffled_minibatches(self, X, Y, minibatch_size):
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


        


    def minibatch_gradient_descent(self, correct, data, num_epochs = 100, learning_rate = 0.1, minibatch_size = 64, learning_rate_decay = True, print_cost = False):
        assert(self.params == None)
        assert(correct.shape[0] == self.structure[-1])
        self.params = self.initialize_parameters(data)

        costs = []
        early_stop = False
        lr_original = learning_rate
        
        ## for cost tracking
        current_iteration = 0
        for epoch in range(num_epochs):
            #### learning rate decay 
            #learning_rate = lr_original/(1 + 10*epoch)
            #param = 1/20
            #learning_rate = param**(epoch) * lr_original
            ###
            minibatches = self.get_shuffled_minibatches(data, correct, minibatch_size)

            for minibatch in minibatches:

                
                current_iteration += 1
                ## get the minibatches 
                (X_batch, Y_batch) = minibatch

                cache = self.forward_propagate(X_batch)

                Y_hat_batch = cache['A' + str(self.length)]
                
                cost = self.compute_cost(Y_hat_batch, Y_batch)

                #self.process_minibatch(Y_hat[minibatch_size * batch: (minibatch_size + 1) * batch, :], data)
                derivs = self.backward_propagate(Y_batch, X_batch, cache)
                self.update_parameters(derivs, learning_rate)

                

                #if(cost < 0.006):
                    #early_stop = True
                   # break
            

            if(print_cost):
                    costs.append(cost)
                    print('cost after {} epochs: {}'.format(epoch, cost))

            #if(early_stop):
                #break

        # plot the costs
        plt.plot(costs)
        plt.ylabel('cost')
        plt.xlabel('# of iterations')
        plt.show()
        return self.params
            