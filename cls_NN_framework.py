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
## Global variables are defined in the class definition
class Deep_NN_frmwrk:
    
    def __init__(self, name):
        self.name = name
        # self.localfolder = '/Users/administrator/Documents/Python/dim_128_mono';
        self.localfolder = 'gdrive/My Drive/';
        self.train_data_file = 'X_K_mono_train_P3.dat'
        self.train_labels_file = 'Y_K_mono_train_P3.dat'
        self.test_data_file = 'X_K_mono_test.dat'
        self.test_labels_file = 'Y_K_mono_test.dat'

        
        # Define input feature length - number of pixels in the images here
        self.img_h = 128
        self.img_w = 128
        self.img_d = 1
    
        dim_1 = self.img_h * self.img_w * self.img_d
        self.layers_dims = [dim_1, 25, 13, 5, 1] #  5-layer model
    
        ### Train data
        self.numtrainfiles = 3
        self.kfolds = 20
        self.numbatches = 15 
        self.learning_rate = 0.01
        self.learning_rate_start = 0.01
        self.epochs = 100 # per data set
        self.print_cost = 1
        self.use_saved_params = 0
        self.L2_lambd = 0.8
        
        # To hold runtime stuff
        self.costs = []                         # keep track of cost
        self.train_successes = []
        self.xval_successes = []

        # Adam optimizer
        self.beta_mom = 0.9
        self.beta_rms = 0.999
        self.mom_prev = {}
        self.rms_prev = {}
        self.epsilon = 1e-8
        
        # keep track of status
        self.starttime = time.time()
        self.curr_epoch = 0

    def update_params_Adam(self, parameters, grads):

        for i in range(1, int(len(parameters.keys())/2)+1):
            if (len(self.mom_prev)==0):
                self.mom_prev['VdW'+str(i)] = np.zeros((parameters['W'+str(i)].shape))
                self.mom_prev['Vdb'+str(i)] = np.zeros((parameters['b'+str(i)].shape))
                self.rms_prev['SdW'+str(i)] = np.zeros((parameters['W'+str(i)].shape))
                self.rms_prev['Sdb'+str(i)] = np.zeros((parameters['b'+str(i)].shape))
            
            self.mom_prev['VdW'+str(i)] = (self.beta_mom * self.mom_prev['VdW'+str(i)]) + ((1 - self.beta_mom) * grads['dW'+str(i)])
            tmp_VdW = np.divide(self.mom_prev['VdW'+str(i)], (1 - np.power(self.beta_mom, self.curr_epoch)))
            self.mom_prev['Vdb'+str(i)] = (self.beta_mom * self.mom_prev['Vdb'+str(i)]) + ((1 - self.beta_mom) * grads['db'+str(i)])
            tmp_Vdb = np.divide(self.mom_prev['Vdb'+str(i)], (1 - np.power(self.beta_mom, self.curr_epoch)))

            self.rms_prev['SdW'+str(i)] = (self.beta_rms * self.rms_prev['SdW'+str(i)]) + ((1 - self.beta_rms) * np.square(grads['dW'+str(i)]))
            tmp_SdW = np.divide(self.mom_rms['SdW'+str(i)], (1 - np.power(self.beta_rms, self.curr_epoch)))
            self.rms_prev['Sdb'+str(i)] = (self.beta_rms * self.rms_prev['Sdb'+str(i)]) + ((1 - self.beta_rms) * np.square(grads['db'+str(i)]))
            tmp_Sdb = np.divide(self.mom_rms['Sdb'+str(i)], (1 - np.power(self.beta_rms, self.curr_epoch)))                
    
            parameters['W'+str(i)] = parameters['W'+str(i)] - (self.learning_rate * np.divide(tmp_VdW, np.sqrt(tmp_SdW) + self.epsilon))
            parameters['b'+str(i)] = parameters['b'+str(i)] - (self.learning_rate * np.divide(tmp_Vdb, np.sqrt(tmp_Sdb) + self.epsilon))
            
        return parameters
    
    def loadSavedParams(self):
    
        num_layers = len(self.layers_dims)
        
        ### Load parameters from training +'.dat'
        parameters = {}
        for i in range(num_layers-1):
            parameters['W'+str(i+1)] = np.loadtxt(self.localfolder+'Out_W'+str(i+1)+'.dat')
            if (len(parameters['W'+str(i+1)].shape) < 2):
                parameters['W'+str(i+1)] = np.reshape(parameters['W'+str(i+1)], (1, len(parameters['W'+str(i+1)])))
                
            parameters['b'+str(i+1)] = np.loadtxt(self.localfolder+'Out_b'+str(i+1)+'.dat')
            if (len(parameters['b'+str(i+1)].shape) < 1):
                parameters['b'+str(i+1)] = np.reshape(parameters['b'+str(i+1)], (1, 1))
            else:
                parameters['b'+str(i+1)] = np.reshape(parameters['b'+str(i+1)], (len(parameters['b'+str(i+1)]), 1))            
        
        return parameters
    
    def saveParams(self, params):
        # Save parameters from training
        for i in range(1,int(len(params.keys())/2)+1):
            np.savetxt(self.localfolder+'Out_W'+str(i)+'.dat', params['W'+str(i)])
            np.savetxt(self.localfolder+'Out_b'+str(i)+'.dat', params['b'+str(i)])
                
            
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
    
    def decayLearningRate(self):
      
      self.learning_rate = self.learning_rate_start/(np.power(10, self.curr_epoch/self.epochs))
               
    def splitKfold(self, X, Y, j, m):
    
        # K-fold xval, separate into train and x_val
        # If no kfold just use 10% as xval
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
    
    def L_layer_model_Kfold(self, X, Y, parameters):#lr was 0.009
    #    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
            
        m = X.shape[1]
        newparams = copy.deepcopy(parameters)
        
        # Loop (gradient descent)
        for i in range(0, self.kfolds): 
    
            X_xval, Y_xval, X_train, Y_train = self.splitKfold(X, Y, i, m)

            # Batch training, numbatches is 1 if no batch separation
            batch_size = int(X_train.shape[1]/self.numbatches)

            for k in range(0, self.numbatches):

                X_batch = X_train[:,(k*batch_size):((k+1)*batch_size-1)]
                Y_batch = Y_train[(k*batch_size):((k+1)*batch_size-1)]

                # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.            
                AL, caches = ex_fwd_prop.L_model_forward(X_batch, newparams)

                # Compute cost
#                cost, train_success = compute_cost(AL, Y_batch)
                # Compute cost with L2 reg
                cost, train_success = ex_compute_cost.compute_L2_reg_cost(AL, Y_batch, newparams, self.L2_lambd)

                # Backward propagation
#                grads = L_model_backward(AL, Y_batch, caches)
                # Backward propagation with L2 reg
                grads = ex_back_prop.L_model_backward_L2_reg(AL, Y_batch, caches, self.L2_lambd)

                # Update parameters.
                newparams = ex_update_parameters.update_parameters(newparams, grads, self.learning_rate)
                    
            # Print the cost every nth training cycle in epoch
            if self.print_cost and i % 4 == 0:
                endtime = time.time()
                print ("Cost after run %i, iteration %i: %f, Time: %f" %(self.curr_epoch, i, cost, (endtime-self.starttime)))
                self.costs.append(cost)
                self.train_successes.append(train_success)

                # Run xval set and compute cost.
                AL_xval, xval = ex_fwd_prop.L_model_forward(X_xval, newparams)        
#                xval_cost, xval_success = compute_cost(AL_xval, Y_xval)
                xval_cost, xval_success = ex_compute_cost.compute_L2_reg_cost(AL_xval, Y_xval, newparams, self.L2_lambd)
                self.xval_successes.append(xval_success)
                
                # plot the costs
                plt.plot(np.squeeze(self.train_successes), 'b')
                plt.plot(np.squeeze(self.xval_successes), 'k')            
                plt.ylabel('Success %')
                plt.xlabel('iterations (per tens)')
                plt.title("Learning rate =" + str(self.learning_rate))
                plt.draw()
                plt.pause(1)
                                    
        return newparams, train_success

    def L_layer_model(self, X, Y, parameters):#lr was 0.009
    #    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
        newparams = copy.deepcopy(parameters)
        
        # Batch training, numbatches is 1 if no batch separation
        batch_size = int(X.shape[1]/self.numbatches)

        for k in range(0, self.numbatches):

            X_batch = X[:,(k*batch_size):((k+1)*batch_size-1)]
            Y_batch = Y[(k*batch_size):((k+1)*batch_size-1)]

            # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.            
            AL, caches = ex_fwd_prop.L_model_forward(X_batch, newparams)

            # Compute cost
#                cost, train_success = compute_cost(AL, Y_batch)
            # Compute cost with L2 reg
            cost, train_success = ex_compute_cost.compute_L2_reg_cost(AL, Y_batch, newparams, self.L2_lambd)

            # Backward propagation
#                grads = L_model_backward(AL, Y_batch, caches)
            # Backward propagation with L2 reg
            grads = ex_back_prop.L_model_backward_L2_reg(AL, Y_batch, caches, self.L2_lambd)

            # Update parameters.
            newparams = ex_update_parameters.update_parameters(newparams, grads, self.learning_rate)

            # Print the cost every nth training cycle in epoch
            if self.print_cost and k % 10 == 0:
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
    
                
        return newparams, train_success
          
    def train_data(self):
        
        print('Layer dims...'+str(self.layers_dims) + '...' + str(time.ctime()))

         # Parameters initialization        
        newparams = {}
        if (self.use_saved_params):
            parameters = self.loadSavedParams()
        else:
            parameters = ex_init_layer_weights.initialize_parameters_deep(self.layers_dims)
                    
        for k in range(self.epochs):
          
          self.curr_epoch = k
          
          for i in range(self.numtrainfiles):

              if (self.numtrainfiles) > 1:
                  X_filename = str.split(self.train_data_file,'.')[0] + '_' + str(i+1) + '.' + str.split(self.train_data_file,'.')[1]
                  Y_filename = str.split(self.train_labels_file,'.')[0] + '_' + str(i+1) + '.' + str.split(self.train_labels_file,'.')[1]
              else:
                  X_filename = self.train_data_file
                  Y_filename = self.train_labels_file 

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
              
              # After the first run, start using the derived params
              if k > 0 or i > 0:
                  parameters = newparams

              newparams, train_success = self.L_layer_model_Kfold(X_train, Y_train, parameters)
          
          # Decay learning rate every epoch
          self.decayLearningRate()

          # Save params every 10 epochs
          if (k + 1) % 10 == 0:
            self.saveParams(newparams)
          
          # Save success and cost every epoch
          np.savetxt(self.localfolder + 'train_success.dat', self.train_successes)
          np.savetxt(self.localfolder + 'train_costs.dat', self.costs)



        print('Optimization done..'+str(train_success))        
        self.saveParams(newparams)
    
            
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
        
        parameters = self.loadSavedParams()
        
        ### Run forward prop to get the output activations    
        AL, caches = ex_fwd_prop.L_model_forward(X_test, parameters)
        
        ### Convert AL to binary calls
        calls = (AL >= 0.5)
        
        ## Calculate success percentage
        success = 1 - np.count_nonzero(Y_test - calls)/m
    
        # Show images from test data and the classification results    
#        self.showImageData(X_test, calls, (t_size))    
        
        return success

    
if __name__ == "__main__":
    main()
 
    
