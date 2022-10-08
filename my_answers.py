""" This module contain class that helps to implement forward and backward pass using numpy

This file can be imported as module and contains following functions:
 * train function -  that controls both forwardpass, backwardPass and updating of weights.
 * forward_pass_train - it performs a forwardpass
 * backpropagation - it performs backpropagation for a minibatch
 * run function - it performs a forward pass just for inference
"""

import numpy as np

class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.velocity1 = 0
        self.velocity2 = 0
        #NB changed momentum from 0.7 to 0.5
        self.momentum = 0.5

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5,(self.input_nodes, self.hidden_nodes))
        
        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5, (self.hidden_nodes, self.output_nodes))
        self.lr = learning_rate
        
        #### TODO: Set self.activation_function to your implemented sigmoid function ####
        #
        # Note: in Python, you can define a function with a lambda expression,
        # as shown below.
        self.activation_function = lambda x : 1/(1+np.exp(-x))  # Replace 0 with your sigmoid calculation.
        
        ### If the lambda code above is not something you're familiar with,
        # You can uncomment out the following three lines and put your 
        # implementation there instead.
        #
        #def sigmoid(x):
        #    return 0  # Replace 0 with your sigmoid calculation here
        #self.activation_function = sigmoid
                    

    def train(self, features, targets):
        ''' Train the network on batch of features and targets. 
        
            Arguments
            ---------
            
            features: 2D array, each row is one data record, each column is a feature
            targets: 1D array of target values
        
        '''
        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
        for X, y in zip(features, targets):
            #NB Commented this
            X=X[:,None] #this is required..
            
            #print('X shape:',X.shape)
            final_outputs, hidden_outputs = self.forward_pass_train(X)  # Implement the forward pass function below
            # Implement the backproagation function below
            delta_weights_i_h, delta_weights_h_o = self.backpropagation(final_outputs, hidden_outputs, X, y, 
                                                                        delta_weights_i_h, delta_weights_h_o)
            
            #NB Changed the indentation to include the following statement under for loop.
        self.update_weights(delta_weights_i_h, delta_weights_h_o, n_records)


    def forward_pass_train(self, X):
        ''' Implement forward pass here 
         
            Arguments
            ---------
            X: features batch
        '''
        #### Implement the forward pass here ####
        ### Forward pass ###
        # TODO: Hidden layer - Replace these values with your calculations.
        
        #NB changed the following line
        hidden_inputs = self.weights_input_to_hidden.T.dot(X) # signals into hidden layer
        # hidden_inputs = np.dot(X, self.weights_input_to_hidden) # signals into hidden layer
        
        hidden_outputs = self.activation_function(hidden_inputs) # signals from hidden layer
        
        #print('hidden_outputs',hidden_outputs.shape)

        # TODO: Output layer - Replace these values with your calculations.
        #NB changed the following line
        final_inputs = self.weights_hidden_to_output.T.dot(hidden_outputs) # signals into final output layer
        # final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output) # signals into final output layer
        
        final_outputs = final_inputs # signals from final output layer
        #print('final_outputs',final_outputs.shape)
        
        return final_outputs, hidden_outputs

    def backpropagation(self, final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o):
        ''' Implement backpropagation
         
            Arguments
            ---------
            final_outputs: output from forward pass
            y: target (i.e. label) batch
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers
        '''
        #### Implement the backward pass here ####
        ### Backward pass ###

        # TODO: Output error - Replace this value with your calculations.
        
        #NB changed the following line
        error = final_outputs-y # Output layer error is the difference between desired target and actual output.
        # error = y-final_outputs # Output layer error is the difference between desired target and actual output.
        
        #print('error',error.shape)
        
        # TODO: Calculate the hidden layer's contribution to the error
        #NB changed the following line
        hidden_error = self.weights_hidden_to_output.dot(error)*hidden_outputs*(1-hidden_outputs)
        # hidden_error = np.dot(self.weights_hidden_to_output, error)*hidden_outputs*(1-hidden_outputs)
        
        #print('hidden_error',hidden_error.shape)
        
        # TODO: Backpropagated error terms - Replace these values with your calculations.
        #output_error_term = error
        
        #hidden_error_term = hidden_error * hidden_outputs * (1 - hidden_outputs)
        
        
        # Weight step (input to hidden)
        #NB changed the following line
        delta_weights_i_h += hidden_error.dot(X.T).T
        # delta_weights_i_h += hidden_error*(X[:, None])
        
        #print('delta_weights_i_h',delta_weights_i_h.shape)
        # Weight step (hidden to output)
        #NB changed the following line
        delta_weights_h_o += error.dot(hidden_outputs.T).T
        # delta_weights_h_o += error*(hidden_outputs[:, None])
        #print('delta_weights_h_o',delta_weights_h_o.shape)
        return delta_weights_i_h, delta_weights_h_o

    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):
        ''' Update weights on gradient descent step
         
            Arguments
            ---------
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers
            n_records: number of records
        '''
        #print('self.weights_hidden_to_output',self.weights_hidden_to_output)
        self.weights_hidden_to_output -= (self.lr/n_records)*delta_weights_h_o # update hidden-to-output weights with gradient descent step
        self.weights_input_to_hidden -= (self.lr/n_records)*delta_weights_i_h # update input-to-hidden weights with gradient descent step
        #print('weights_hidden_to_output',self.weights_hidden_to_output.T)
        #print('weights_input_to_hidden',self.weights_input_to_hidden.T)
        #print('(self.lr/n_records)*delta_weights_h_o',(self.lr/n_records)*delta_weights_h_o)
        #print('(self.lr/n_records)*delta_weights_i_h',(self.lr/n_records)*delta_weights_i_h)
        #print(self.lr)

        #code for using Momentum with learning rate...
        #NB changed the following
        #self.velocity1 = self.momentum*self.velocity1 - (self.lr/n_records)*delta_weights_h_o
        # self.velocity1 = (self.lr/n_records)*delta_weights_h_o - self.momentum*self.velocity1
        # self.weights_hidden_to_output += self.velocity1

        #NB changed the following
        #self.velocity2 = self.momentum*self.velocity2 - (self.lr/n_records)*delta_weights_i_h
        # self.velocity2 = (self.lr/n_records)*delta_weights_i_h - self.momentum*self.velocity2
        # self.weights_input_to_hidden += self.velocity2
        

    def run(self, features):
        ''' Run a forward pass through the network with input features 
        
            Arguments
            ---------
            features: 1D array of feature values
        '''
        # import pdb;pdb.set_trace()
        #### Implement the forward pass here ####
        # TODO: Hidden layer - replace these values with the appropriate calculations.
        #NB changed the following
        hidden_inputs = self.weights_input_to_hidden.T.dot(features.T) # signals into hidden layer
        # hidden_inputs = np.dot(features, self.weights_input_to_hidden) # signals into hidden layer
        
        hidden_outputs = self.activation_function(hidden_inputs) # signals from hidden layer
        # pdb.set_trace()
        # TODO: Output layer - Replace these values with the appropriate calculations.
         #NB changed the following
        final_inputs = self.weights_hidden_to_output.T.dot(hidden_outputs) # signals into final output layer
        # final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output) # signals into final output layer
        
        final_outputs = final_inputs # signals from final output layer 
        
        return final_outputs


#########################################################
# Set your hyperparameters here
##########################################################
#NB Changed the hyperparameters, iterations from 500 to 1200, learning_rate from 0.0045 to 0.09, hidden_nodes
# from 12 to 5
iterations = 9000
learning_rate = 0.2
hidden_nodes = 6
output_nodes = 1
