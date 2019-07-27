#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 13:56:23 2019
NN framework as a class
Declare all hyper parameters in class definition, shared by the other functions
@author: nkolli
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 20:13:42 2019

Framework for NN, fully connected layers followed by linear activation for output
Output activation should be a linear function
Functions are written and tested separately where possible

@author: Naveen Kolli
"""
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time
import copy

# functions to implement the model
import ex_fwd_prop
import ex_init_layer_weights
import ex_update_parameters
import ex_back_prop
import ex_compute_cost

def main():
    
    myNN = Deep_NN_frmwrk("catsDogs")
    print(myNN.kfolds)
    print(myNN.localfolder)
    
    starttime = time.time()
    print(time.ctime())

#    myNN.train_data()

    ### Test data
    success = myNN.test_data((myNN.img_w, myNN.img_h))

    endtime = time.time()
    print('Time {0}, num_epochs {1}, success {2}'.format((endtime-starttime), myNN.num_epochs, success))

## Global variables are defined in the class definition
class Deep_NN_frmwrk:
    
    def __init__(self, name):
        self.name = name
        # self.localfolder = '/Users/administrator/Documents/Python/dim_128_mono';
        self.localfolder = 'gdrive/My Drive/';
        self.train_data_file = 'X_train.dat'
        self.train_labels_file = 'Y_train.dat'
        self.test_data_file = 'X_test.dat'
        self.test_labels_file = 'Y_test.dat'

        
        # Define input feature length - number of pixels in the images here
        self.img_h = 32
        self.img_w = 32
        self.img_d = 1
    
        dim_1 = self.img_h * self.img_w * self.img_d
        self.layers_dims = [dim_1, 50, 11, 5, 1] #  5-layer model
        self.nn_parameters = {}
    
        ### Train data
        self.numtrainfiles = 1
        self.kfolds = 10
        self.numbatches = 2 
        self.learning_rate = 0.005
        self.learning_rate_start = 0.005
        self.epochs = 10000 # per data set
        self.print_cost = 1
        self.use_saved_params = 0
        self.L2_lambd = 0.8
        
        # To hold runtime stuff
        self.costs = []                         # keep track of cost
        self.train_successes = []
        self.xval_successes = []

        # Adam optimizer
        self.beta_mom = 0.9
        self.beta_rms = 0.99
        self.mom_prev = {}
        self.rms_prev = {}
        self.epsilon = 1e-8
        
        # keep track of status
        self.starttime = time.time()
        self.curr_epoch = 0
        self.n_iter = 0

##### Function
    # Adam optimized update of weights        
    def update_params_Adam(self, grads, batch_size):

        # print('Size of mom_prev is ..' + str(len(self.mom_prev)))
        # batch_size = 1
        if (len(self.mom_prev)==0):
            for i in range(1, int(len(self.nn_parameters.keys())/2)+1):
                self.mom_prev['VdW'+str(i)] = np.zeros((self.nn_parameters['W'+str(i)].shape))
                self.mom_prev['Vdb'+str(i)] = np.zeros((self.nn_parameters['b'+str(i)].shape))
                self.rms_prev['SdW'+str(i)] = np.zeros((self.nn_parameters['W'+str(i)].shape))
                self.rms_prev['Sdb'+str(i)] = np.zeros((self.nn_parameters['b'+str(i)].shape))

        for i in range(1, int(len(self.nn_parameters.keys())/2)+1):
            
            self.mom_prev['VdW'+str(i)] = (self.beta_mom * self.mom_prev['VdW'+str(i)]) + ((1 - self.beta_mom) * grads['dW'+str(i)]/batch_size)
            tmp_VdW = np.divide(self.mom_prev['VdW'+str(i)], (1 - np.power(self.beta_mom, self.n_iter)))
            self.mom_prev['Vdb'+str(i)] = (self.beta_mom * self.mom_prev['Vdb'+str(i)]) + ((1 - self.beta_mom) * grads['db'+str(i)]/batch_size)
            tmp_Vdb = np.divide(self.mom_prev['Vdb'+str(i)], (1 - np.power(self.beta_mom, self.n_iter)))

            self.rms_prev['SdW'+str(i)] = (self.beta_rms * self.rms_prev['SdW'+str(i)]) + ((1 - self.beta_rms) * np.square(grads['dW'+str(i)]/batch_size))
            tmp_SdW = np.divide(self.rms_prev['SdW'+str(i)], (1 - np.power(self.beta_rms, self.n_iter)))
            self.rms_prev['Sdb'+str(i)] = (self.beta_rms * self.rms_prev['Sdb'+str(i)]) + ((1 - self.beta_rms) * np.square(grads['db'+str(i)]/batch_size))
            tmp_Sdb = np.divide(self.rms_prev['Sdb'+str(i)], (1 - np.power(self.beta_rms, self.n_iter)))                
    
            self.nn_parameters['W'+str(i)] = self.nn_parameters['W'+str(i)] - (self.learning_rate * np.divide(tmp_VdW, (np.sqrt(tmp_SdW) + self.epsilon)))
            self.nn_parameters['b'+str(i)] = self.nn_parameters['b'+str(i)] - (self.learning_rate * np.divide(tmp_Vdb, (np.sqrt(tmp_Sdb) + self.epsilon)))

##### Function
    # Adam with Weight decay, use without L2 regularization        
    def update_params_Adam_Wdecay(self, grads):

        if (len(self.mom_prev)==0):
            for i in range(1, int(len(self.nn_parameters.keys())/2)+1):
                self.mom_prev['VdW'+str(i)] = np.zeros((self.nn_parameters['W'+str(i)].shape))
                self.mom_prev['Vdb'+str(i)] = np.zeros((self.nn_parameters['b'+str(i)].shape))
                self.rms_prev['SdW'+str(i)] = np.zeros((self.nn_parameters['W'+str(i)].shape))
                self.rms_prev['Sdb'+str(i)] = np.zeros((self.nn_parameters['b'+str(i)].shape))

        for i in range(1, int(len(self.nn_parameters.keys())/2)+1):
            
            self.mom_prev['VdW'+str(i)] = (self.beta_mom * self.mom_prev['VdW'+str(i)]) + ((1 - self.beta_mom) * grads['dW'+str(i)])
            tmp_VdW = np.divide(self.mom_prev['VdW'+str(i)], (1 - np.power(self.beta_mom, self.n_iter)))
            self.mom_prev['Vdb'+str(i)] = (self.beta_mom * self.mom_prev['Vdb'+str(i)]) + ((1 - self.beta_mom) * grads['db'+str(i)])
            tmp_Vdb = np.divide(self.mom_prev['Vdb'+str(i)], (1 - np.power(self.beta_mom, self.n_iter)))

            self.rms_prev['SdW'+str(i)] = (self.beta_rms * self.rms_prev['SdW'+str(i)]) + ((1 - self.beta_rms) * np.square(grads['dW'+str(i)]))
            tmp_SdW = np.divide(self.rms_prev['SdW'+str(i)], (1 - np.power(self.beta_rms, self.n_iter)))
            self.rms_prev['Sdb'+str(i)] = (self.beta_rms * self.rms_prev['Sdb'+str(i)]) + ((1 - self.beta_rms) * np.square(grads['db'+str(i)]))
            tmp_Sdb = np.divide(self.rms_prev['Sdb'+str(i)], (1 - np.power(self.beta_rms, self.n_iter)))                

            W_Wdecay = self.learning_rate * self.nn_parameters['W'+str(i)] * self.L2_lambd

            self.nn_parameters['W'+str(i)] = self.nn_parameters['W'+str(i)] - (self.learning_rate * np.divide(tmp_VdW, (np.sqrt(tmp_SdW) + self.epsilon))) - W_Wdecay
            self.nn_parameters['b'+str(i)] = self.nn_parameters['b'+str(i)] - (self.learning_rate * np.divide(tmp_Vdb, (np.sqrt(tmp_Sdb) + self.epsilon)))
            
##### Function
    # Load trained weights from disk    
    def loadSavedParams(self):
    
        num_layers = len(self.layers_dims)
        
        ### Load parameters from training +'.dat'
        for i in range(num_layers-1):
            self.nn_parameters['W'+str(i+1)] = np.loadtxt(self.localfolder+'Out_W'+str(i+1)+'.dat')
            if (len(self.nn_parameters['W'+str(i+1)].shape) < 2):
                self.nn_parameters['W'+str(i+1)] = np.reshape(self.nn_parameters['W'+str(i+1)], (1, len(self.nn_parameters['W'+str(i+1)])))
                
            self.nn_parameters['b'+str(i+1)] = np.loadtxt(self.localfolder+'Out_b'+str(i+1)+'.dat')
            if (len(self.nn_parameters['b'+str(i+1)].shape) < 1):
                self.nn_parameters['b'+str(i+1)] = np.reshape(self.nn_parameters['b'+str(i+1)], (1, 1))
            else:
                self.nn_parameters['b'+str(i+1)] = np.reshape(self.nn_parameters['b'+str(i+1)], (len(self.nn_parameters['b'+str(i+1)]), 1))            
        
##### Function
    # Save trained weights    
    def saveParams(self):
        # Save parameters from training
        for i in range(1,int(len(self.nn_parameters.keys())/2)+1):
            np.savetxt(self.localfolder+'Out_W'+str(i)+'.dat', self.nn_parameters['W'+str(i)])
            np.savetxt(self.localfolder+'Out_b'+str(i)+'.dat', self.nn_parameters['b'+str(i)])
                
##### Function
    # test function to view image data            
    def showImageData(X, Y, t_size):
        
    #    # To view the images, make sure labeling is correct
        imgi = int(input('Enter an index..'))
        while (imgi != 99):
            print('Index chosen is...', imgi)
            print('Label is...', Y[imgi])
            reimg = Image.fromarray(np.reshape(X[:,imgi],(t_size)))
    #        t_size = (320,320)
    #        reimg = reimg.resize(t_size, Image.ANTIALIAS)
            plt.imshow(reimg)
            imgi = int(input('Enter an index..'))

##### Function
    # Update learning rate based on epoch number        
    def decayLearningRate(self):
      
      self.learning_rate = self.learning_rate_start/(np.power(10, self.curr_epoch/self.epochs))

##### Function
    # K-fold xval, separate into train and x_val
    # If no kfold just use 10% as xval    
    def splitKfold(self, X, Y, j, m):
    
        if (self.kfolds == 1):
            xval = int(m/10)
            x_train = X[:,0:m-xval]
            y_train = Y[0:m-xval]    
            x_xval = X[:,m-xval+1:]
            y_xval = Y[m-xval+1:]
        else:
            xval = int(m/self.kfolds)
            tmp_range = range(j*xval, (j+1)*xval)
            x_xval = X[:, tmp_range]
            y_xval = Y[tmp_range]
            # splits X and Y
            if (j==0):
                tmp_range = range((j+1)*xval+1, m)
                x_train = X[:, tmp_range]
                y_train = Y[tmp_range]
            else:
                tmp_range1 = range(0, j*xval)
                tmp_range2 = range((j+1)*xval+1, m)
                x_train = np.concatenate((X[:, tmp_range1], X[:, tmp_range2]), axis=1)
                y_train = np.concatenate((Y[tmp_range1], Y[tmp_range2]), axis=0)
                
    #    print(x_xval.shape, y_xval.shape, x_train.shape, y_train.shape)

        return x_xval, y_xval, x_train, y_train

##### Function
    # Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
    # With Kfold cross validation
    def L_layer_model_Kfold(self, X, Y):#lr was 0.009
            
        m = X.shape[1]
        
        # Loop (gradient descent)
        for i in range(0, self.kfolds): 
    
            X_xval, Y_xval, X_train, Y_train = self.splitKfold(X, Y, i, m)

            # Batch training, numbatches is 1 if no batch separation
            batch_size = int(X_train.shape[1]/self.numbatches)

            for k in range(0, self.numbatches):
    
                self.n_iter += 1
                
                X_batch = X_train[:,(k*batch_size):((k+1)*batch_size-1)]
                Y_batch = Y_train[(k*batch_size):((k+1)*batch_size-1)]

                # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.            
                AL, caches = L_model_forward(X_batch, self.nn_parameters)

                # Compute cost
                # cost, train_success = compute_cost(AL, Y_batch)
                # Compute cost with L2 reg
                cost, train_success = compute_L2_reg_cost(AL, Y_batch, self.nn_parameters, self.L2_lambd)

                # Backward propagation
                # grads = L_model_backward(AL, Y_batch, caches)
                # Backward propagation with L2 reg
                grads = L_model_backward_L2_reg(AL, Y_batch, caches, self.L2_lambd)

                # Update parameters.
                # self.nn_parameters = update_parameters(self.nn_parameters, grads, self.learning_rate)
                # self.update_params_Adam(grads, 1)
                self.update_params_Adam_Wdecay(grads)                
                    
            # Print the cost every nth training cycle in epoch
            if self.print_cost and i % 4 == 0:
                endtime = time.time()
                print ("Cost after run %i, iteration %i: %f, Time: %f" %(self.curr_epoch, i, cost, (endtime-self.starttime)))
                self.costs.append(cost)
                self.train_successes.append(train_success)
                self.costs = self.costs[-1000:]
                self.train_successes = self.train_successes[-1000:]

                # Run xval set and compute cost.
                AL_xval, xval = L_model_forward(X_xval, self.nn_parameters)        
#                xval_cost, xval_success = compute_cost(AL_xval, Y_xval)
                xval_cost, xval_success = compute_L2_reg_cost(AL_xval, Y_xval, self.nn_parameters, self.L2_lambd)
                self.xval_successes.append(xval_success)
                
                # plot the costs
                plt.plot(np.squeeze(self.train_successes), 'b')
                plt.plot(np.squeeze(self.xval_successes), 'k')            
                plt.ylabel('Success %')
                plt.xlabel('iterations (per tens)')
                plt.title("Learning rate =" + str(self.learning_rate))
                plt.draw()
                plt.pause(1)
                                    
        return train_success
      
##### Function
    # Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.    
    def L_layer_model(self, X, Y):#lr was 0.009
    #    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
        
        # Batch training, numbatches is 1 if no batch separation
        batch_size = int(X.shape[1]/self.numbatches)

        for k in range(0, self.numbatches):

            self.n_iter += 1

            X_batch = X[:,(k*batch_size):((k+1)*batch_size-1)]
            Y_batch = Y[(k*batch_size):((k+1)*batch_size-1)]

            # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.            
            AL, caches = L_model_forward(X_batch, self.nn_parameters)

            # Compute cost
            # cost, train_success = compute_cost(AL, Y_batch)
            # Compute cost with L2 reg
            cost, train_success = compute_L2_reg_cost(AL, Y_batch, self.nn_parameters, self.L2_lambd)

            # Backward propagation
            # grads = L_model_backward(AL, Y_batch, caches)
            # Backward propagation with L2 reg
            grads = L_model_backward_L2_reg(AL, Y_batch, caches, self.L2_lambd)

            # Update parameters.
            # self.nn_parameters = update_parameters(self.nn_parameters, grads, self.learning_rate)
            # self.update_params_Adam(grads, 1)
            self.update_params_Adam_Wdecay(grads)


            # Print the cost every nth training cycle in epoch
            if self.print_cost and k % 4 == 0:
                endtime = time.time()
                print ("Cost after run %i, iteration %i: %f, Time: %f" %(self.curr_epoch, k, cost, (endtime-self.starttime)))
                self.costs.append(cost)
                self.train_successes.append(train_success)

                # plot the costs
                plt.plot(np.squeeze(self.train_successes), 'b')
                plt.ylabel('Success %')
                plt.xlabel('iterations (per tens)')
                plt.title("Learning rate =" + str(self.learning_rate))
                plt.draw()
                plt.pause(1)
    
                
        return train_success

##### Function
    # Train data based on class variables      
    def train_data(self):
        
        print('Layer dims...'+str(self.layers_dims) + '...' + str(time.ctime()))

         # Parameters initialization        
        if (self.use_saved_params):
            self.loadSavedParams()
        else:
            self.nn_parameters = initialize_parameters_deep(self.layers_dims)
                    
        for k in range(self.epochs):
          
          self.curr_epoch = k+1
          
          for i in range(self.numtrainfiles):

              if (self.numtrainfiles) > 1:
                  X_filename = str.split(self.train_data_file,'.')[0] + '_' + str(i+1) + '.' + str.split(self.train_data_file,'.')[1]
                  Y_filename = str.split(self.train_labels_file,'.')[0] + '_' + str(i+1) + '.' + str.split(self.train_labels_file,'.')[1]
                  # Load dataset 
                  X_train = np.loadtxt(self.localfolder + X_filename)
                  Y_train = np.loadtxt(self.localfolder + Y_filename)

                  # Number of examples
                  m = X_train.shape[1]
                  print('Data loaded..' + str(time.ctime()) + '...' + str(X_train.shape))

                  # Shuffle the data
                  marr = np.arange(m)
                  np.random.shuffle(marr)
                  X_train = X_train[:,marr]
                  Y_train = Y_train[marr]

                  X_train = X_train/255       
                  
              else:
                  X_filename = self.train_data_file
                  Y_filename = self.train_labels_file 
                  
                  if (k == 0):
                    # Load dataset 
                    X_train = np.loadtxt(self.localfolder + X_filename)
                    Y_train = np.loadtxt(self.localfolder + Y_filename)
                    # Number of examples
                    m = X_train.shape[1]
                    print('Data loaded..' + str(time.ctime()) + '...' + str(X_train.shape))

                    # Shuffle the data
                    marr = np.arange(m)
                    np.random.shuffle(marr)
                    X_train = X_train[:,marr]
                    Y_train = Y_train[marr]

                    X_train = X_train/255       


              
              train_success = self.L_layer_model_Kfold(X_train, Y_train)
              #train_success = self.L_layer_model(X_train, Y_train)              
          
          # Decay learning rate every epoch
          self.decayLearningRate()

          # Save params every 10 epochs
          if (k + 1) % 10 == 0:
            self.saveParams()
          
          # Save success and cost every epoch
          np.savetxt(self.localfolder + 'train_success.dat', self.train_successes)
          np.savetxt(self.localfolder + 'train_costs.dat', self.costs)



        print('Optimization done..'+str(train_success))        
        self.saveParams()
    
##### Function
    # Test data based on saved parameters and class variables            
    def test_data(self, t_size):
        
        # Load test data
        X_test = np.loadtxt(self.localfolder + self.test_data_file)
        Y_test = np.loadtxt(self.localfolder + self.test_labels_file)
        
        # Number of examples
        m = X_test.shape[1]
        
        # Shuffle the data
        marr = np.arange(m)
        np.random.shuffle(marr)
        X_test = X_test[:,marr]
        Y_test = Y_test[marr]
        
        X_test = X_test/255
        
        self.loadSavedParams()
        
        ### Run forward prop to get the output activations    
        AL, caches = L_model_forward(X_test, self.nn_parameters)
        
        ### Convert AL to binary calls
        calls = (AL >= 0.5)
        
        ## Calculate success percentage
        success = 1 - np.count_nonzero(Y_test - calls)/m
    
        # Show images from test data and the classification results    
#        self.showImageData(X_test, calls, (t_size))    
        
        return success

    
if __name__ == "__main__":
    main()
 
    
