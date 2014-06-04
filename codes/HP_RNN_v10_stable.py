# -*- coding: cp936 -*-
# Horizontal product + Recurrent network to functionally simulate
# Ventral and dorsal pathways in the visual process of the brain.

#
# Author: J Zhong
# Date: 01-24-2013
# zhong@informatik.uni-hamburg.de

# See also: Zhong, J., Weber, C., & Wermter, S. (2012). Learning Features and Predictive
# Transformation Encoding Based on a Horizontal Product Model. Artificial Neural
# Networks and Machine Learning–ICANN 2012, 539-546.

"""
First, download all files into one directory:
- HP_RNN_distributed.py  (this file)
- retina.py     (for reading the image data patches)
- KTimage.py    (for exporting the weights and activations as pgm files for display)
- look.tcl      (to display those pgm files; look.tcl may be in any directory)

 To visualise the weight and layer activation,
 ./look.tcl a w 0 1  (left mouse click reloads the newest weights and activation files).
 Remember to create a directory /tmp/coco  for the weights and activations files.

 You will see
 1 . Activations in one hidden layer are constrained to be the same as previous time-step (ventral-like representation)
 2 . Activations in another hidden layer represent dorsal-like pathway

"""

# VARIATION OF Code from Chapter 3 of Machine Learning: An Algorithmic Perspective
# by Stephen Marsland (http://seat.massey.ac.nz/personal/s.r.marsland/MLBook.html)
# Stephen Marsland, 2008

# -------------------------------------------------------------------
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# -------------------------------------------------------------------


from numpy import *
import numpy
import random
import retina as R
import KTimage as KT



class MLP:
    """ A Multi-Layer Perceptron"""

    def __init__(self,size, nObject, beta=1, outtype='linear'):
        """ Constructor """
        # Set up network size
        

        self.size = size

        self.size_in_h = self.size
        self.size_in_w = self.size
        self.size_out_h = self.size
        self.size_out_w = self.size

        self.visual_step = 100
        self.nLayer = nObject
        self.nhidden1 = self.nLayer # layer 1 represents object identity. Number of units should be the same as layers
        self.nhidden2 = 4 * self.size**2 # layer 2 represents object movement info. Number of units equals to size of the grid in each layer multiplying 4 directions.
        self.beta = beta

        self.__forward_alpha_v = 0.70 # when doing forward transformation function, y = alpha*y + (1-alpha)*input*w
        self.__forward_alpha_d = 0.0

        
        self.__a1 = 2.5*numpy.ones((self.nhidden1))
        self.__b1 = -5.0*numpy.ones((self.nhidden1))
        self.__mu1 = 0.20

        self.__a2 = 2.5*numpy.ones((self.nhidden2))
        self.__b2 = -5.0*numpy.ones((self.nhidden2))
        self.__mu2 = 0.0025
        
        self.__eta_a = 0.0001
        self.__eta_b = 0.00001

        
        
        self.outtype = outtype
        self.nSampleInLoop = self.size + 1
        self.dataIteration = self.size_in_h * self.size_in_w # starting point of one block, equals to the size of block, i.e. width * height

        self.ndata = self.dataIteration * self.nSampleInLoop * 4 # number of the training data set in one layer (one object)
        print "Generating data with number", self.ndata, 'in one layer'
        
        self.inputs, self.targets = self.__genLargerData() # generate data for training
        
        print 'The complete training data includes', self.ndata*self.nLayer, 'samples.'
        print 'Size of each layer:', self.size**2
        
        self.nin = shape(self.inputs)[1] * shape(self.inputs)[2]
        self.nout = shape(self.targets)[1] * shape(self.targets)[2]
        #self.new_inputs = self.inputs
        self.new_inputs = numpy.reshape(self.inputs,(self.ndata*self.nLayer, self.nLayer*self.size_in_h*self.size_in_w))
        self.targets = numpy.reshape(self.targets, (self.ndata*self.nLayer, self.nLayer*self.size_in_h*self.size_in_w))

        
        
        KT.exporttiles(self.new_inputs + 1, self.size_in_h*self.nLayer, self.size_in_w, basedir+"obs_T_0.pgm", self.nLayer, self.ndata)
        print 'Training data saved...'


        # Initialise network
        self.weights_in_hidden1 = (numpy.random.rand(self.nhidden1,self.nin)-0.5)*2/sqrt(self.nin) #weights between input and hidden1
        self.weights_in_hidden2 = (numpy.random.rand(self.nhidden2,self.nin)-0.5)*2/sqrt(self.nin) #weights between input and hidden2
        
        self.weights_hidden1_out = (numpy.random.rand(self.nout,self.nhidden1)-0.5)*2/sqrt(self.nhidden1) #weights between hidden1 and output
        self.weights_hidden2_out = (numpy.random.rand(self.nout,self.nhidden2)-0.5)*2/sqrt(self.nhidden2) #weights between hidden2 and output

        self.bias_hidden1_out = -0.1*numpy.ones((1))
        self.bias_hidden2_out = -0.1*numpy.ones((1))
        self.weights_bias1_out = (numpy.random.rand(self.nout,1)-0.5)*2
        self.weights_bias2_out = (numpy.random.rand(self.nout,1)-0.5)*2

        self.weights_hidden1_hidden1 = (numpy.random.rand(self.nhidden1,self.nhidden1)-0.5)*2/sqrt(self.nhidden1) #weights within hidden1
        self.weights_hidden2_hidden2 = (numpy.random.rand(self.nhidden2,self.nhidden2)-0.5)*2/sqrt(self.nhidden2) #weights within hidden2

        self.weights_in_hidden1_delay = (numpy.random.rand(self.nhidden1,self.nin)-0.5)*2/sqrt(self.nin)
        self.weights_in_hidden2_delay = (numpy.random.rand(self.nhidden2,self.nin)-0.5)*2/sqrt(self.nin)

   
            
        
            
    def __genLargerData(self):

         #Data preparation (__genLargerData()):
         #The objects (number defined by 'nObject') is visualised as a layer.
         #The object movement is in four directions.
         #Starting of movement covers the whole grid world.

        
        OutputData = numpy.zeros((self.ndata*self.nLayer, self.nLayer, self.size_out_h*self.size_out_w)) # whole dataset 
        InputData = numpy.zeros((self.ndata*self.nLayer, self.nLayer, self.size_in_h*self.size_in_w))

        
        LayerOutputData = numpy.zeros((self.ndata, self.size_out_h*self.size_out_w)) # training set in one layer
        LayerInputData = numpy.zeros((self.ndata, self.size_in_h*self.size_in_w))
        Input_x = 0
        Input_y = 0
        
        for data in range(0, self.dataIteration):
            
            for x in range(0,self.nSampleInLoop):
                
                tempInputMatrix = numpy.zeros((self.size_in_h,self.size_in_w))
                tempOutputMatrix = numpy.zeros((self.size_out_h,self.size_out_w))
                tempInputMatrix[(Input_y+data)%self.size_in_h][(Input_x+x)%self.size_in_w] = 1
                tempOutputMatrix[(Input_y+data)%self.size_out_h][(Input_x+x+1)%self.size_out_w] = 1
                LayerInputData[data*self.nSampleInLoop+x,:]  = numpy.reshape(tempInputMatrix,(1,self.size_in_h*self.size_in_w))
                LayerOutputData[data*self.nSampleInLoop+x,:] = numpy.reshape(tempOutputMatrix,(1,self.size_out_h*self.size_out_w))
                tempInputMatrix = numpy.zeros((self.size_in_h,self.size_in_w))
                tempOutputMatrix = numpy.zeros((self.size_out_h,self.size_out_w))
                tempInputMatrix[(Input_x + x)%self.size_in_h][(Input_y+data)%self.size_in_w] = 1
                tempOutputMatrix[(Input_x + x + 1)%self.size_out_h][(Input_y+data)%self.size_out_w] = 1
                LayerInputData[self.ndata/2 + data*self.nSampleInLoop+x,:]  = numpy.reshape(tempInputMatrix,(1,self.size_in_h*self.size_in_w))
                LayerOutputData[self.ndata/2 + data*self.nSampleInLoop+x,:] = numpy.reshape(tempOutputMatrix,(1,self.size_out_h*self.size_out_w))
            if data%self.size == self.size - 1:
                Input_x += 1
        
        Input_x = 0
        for data in range(0,self.dataIteration):
            
            for x in range(0,self.nSampleInLoop):
                tempInputMatrix = numpy.zeros((self.size_in_h,self.size_in_w))
                tempOutputMatrix = numpy.zeros((self.size_out_h,self.size_out_w))
                tempInputMatrix[(Input_y+data)%self.size_in_h][(Input_x-x)%self.size_in_w] = 1
                tempOutputMatrix[(Input_y+data)%5][(Input_x-x-1)%5] = 1
                LayerInputData[self.ndata/4+data*self.nSampleInLoop+x, :]  = numpy.reshape(tempInputMatrix,(1,25))
                LayerOutputData[self.ndata/4+data*self.nSampleInLoop+x, :] = numpy.reshape(tempOutputMatrix,(1,25))
                tempInputMatrix = numpy.zeros((self.size_in_h,self.size_in_w))
                tempOutputMatrix = numpy.zeros((self.size_out_h,self.size_out_w))
                tempInputMatrix[(Input_x - x)%self.size_in_h][(Input_y+data)%self.size_in_w] = 1
                tempOutputMatrix[(Input_x - x - 1)%5][(Input_y+data)%5] = 1
                LayerInputData[self.ndata*3/4 + data*self.nSampleInLoop+x,:]  = numpy.reshape(tempInputMatrix,(1,25))
                LayerOutputData[self.ndata*3/4 + data*self.nSampleInLoop+x,:] = numpy.reshape(tempOutputMatrix,(1,25))
            if data%self.size == self.size - 1:
                Input_x += 1
        
        for layer in range(0, self.nLayer):
            for self.niteration in range(0,self.ndata):
                
                InputData[self.ndata*layer + self.niteration,layer,:] = LayerInputData[self.niteration,:]
                OutputData[self.ndata*layer + self.niteration,layer,:] = LayerOutputData[self.niteration, :]

        return InputData, OutputData  
    
    
    def mlptrain(self,eta_bias,eta,MaxIterations):
        """ Train the thing """
        
        
        
        updatew_in_hidden1_delay = zeros((shape(self.weights_in_hidden1_delay))) 
        updatew_in_hidden2_delay = zeros((shape(self.weights_in_hidden2_delay)))
        updatew_in_hidden1 = zeros((shape(self.weights_in_hidden1))) 
        updatew_in_hidden2 = zeros((shape(self.weights_in_hidden2)))
        updatew_hidden1_out = zeros((shape(self.weights_hidden1_out)))
        updatew_hidden2_out = zeros((shape(self.weights_hidden2_out)))
        updatew_hidden1_hidden1 = zeros((shape(self.weights_hidden1_hidden1)))
        updatew_hidden2_hidden2 = zeros((shape(self.weights_hidden2_hidden2)))

        
        error = 0.0
        for n in range(MaxIterations):
            old_error = error
            error = 0.0
            self.hidden1 = zeros((self.ndata*self.nLayer,self.nhidden1))
            self.hidden2 = zeros((self.ndata*self.nLayer,self.nhidden2))
            self.hidden1_y = zeros((self.ndata*self.nLayer, self.nhidden1))
            self.hidden2_y = zeros((self.ndata*self.nLayer, self.nhidden2))
            self.hidden1_z = zeros((self.ndata*self.nLayer, self.nhidden1))
            self.hidden2_z = zeros((self.ndata*self.nLayer, self.nhidden2))
            #hidden layer states of all samples by one iteration
            self.hidden1[0] = 0.5 * ones(self.nhidden1)
            self.hidden2[0] = 0.5 * ones(self.nhidden2)

            outputs_sav = numpy.zeros((self.ndata*self.nLayer, self.nLayer*self.size_in_h*self.size_in_w))
            horProduct1_sav = numpy.zeros((self.ndata*self.nLayer, self.nLayer*self.size_in_h*self.size_in_w))
            horProduct2_sav = numpy.zeros((self.ndata*self.nLayer, self.nLayer*self.size_in_h*self.size_in_w))
            
            for self.iter in range(self.ndata * self.nLayer):
                
                currenttar = self.targets[self.iter] #current target

                

                if self.iter%(self.nSampleInLoop) == 0:
                    self.outputs = self.mlpfwd(self.new_inputs[self.iter], numpy.zeros((self.nin)))
                else:
                    self.outputs = self.mlpfwd(self.new_inputs[self.iter], self.new_inputs[self.iter - 1])

               
                
                outputs_sav[self.iter,:] = self.outputs
                
                horProduct1_sav[self.iter,:] = self.horProduct1
                horProduct2_sav[self.iter,:] = self.horProduct2

                

                if  self.iter%(self.nSampleInLoop) > 0:
                    error += 0.5*sum((currenttar-self.outputs)**2)

                # Different types of output neurons
                    if self.outtype == 'linear':
                        deltao = (currenttar-self.outputs)
                    elif self.outtype == 'logistic':
                        deltao = (currenttar-self.outputs)*self.outputs*(1.0-self.outputs)
                    elif self.outtype == 'softmax':
                        deltao = (currenttar-self.outputs)
                    else:
                        print "error"

                # error in hidden node of current time-step
                    
                    deltah0_1  = self.hidden1[self.iter] * (1.0-(self.hidden1[self.iter]))*(dot(transpose(self.weights_hidden1_out),deltao*self.horProduct2))*(self.__a1)
                    deltah0_2  = self.hidden2[self.iter] * (1.0-(self.hidden2[self.iter]))*(dot(transpose(self.weights_hidden2_out),deltao*self.horProduct1))*(self.__a2)
                                             
                if (self.iter%self.nLayer > 1):
                    # error in hidden node of previous time-step, caused by recurrent
                    deltah2_2 = self.hidden2[self.iter-1]*(1.0-self.hidden2[self.iter-1])*(dot(transpose(self.weights_hidden2_hidden2),deltah0_2))
                    deltah1_1 = self.hidden1[self.iter-1]*(1.0-self.hidden1[self.iter-1])*(dot(transpose(self.weights_hidden1_hidden1),deltah0_1))
                
                    
                # update of weight between hidden layer and input (current and time-delay), learning only after movement starts
                if self.iter%(self.nSampleInLoop) > 0:
                    updatew_in_hidden1 = eta*(outer(deltah0_1,self.new_inputs[self.iter]))
                    updatew_in_hidden2 = eta*(outer(deltah0_2,self.new_inputs[self.iter]))

                    updatew_in_hidden1_delay = eta*(outer(deltah0_1,self.new_inputs[self.iter - 1]))
                    updatew_in_hidden2_delay = eta*(outer(deltah0_2,self.new_inputs[self.iter - 1]))
                
                if (self.iter%(self.nSampleInLoop) > 1):
                    updatew_in_hidden2 += eta*(outer(deltah2_2,self.new_inputs[self.iter - 1]))
                
                # update of weight between hidden layer and output, learning only after movement starts
                if self.iter%(self.nSampleInLoop) > 0:
                    
                    updatew_hidden1_out = eta*(outer(deltao*self.horProduct2,self.hidden1[self.iter]*self.__a1))
                    updatew_hidden2_out = eta*(outer(deltao*self.horProduct1,self.hidden2[self.iter]*self.__a2))
                    
                    updatew_bias1_out = eta_bias*(outer(dot(updatew_hidden1_out, transpose(self.hidden1[self.iter])) , self.bias_hidden1_out))
                    updatew_bias2_out = eta_bias*(outer(dot(updatew_hidden2_out, transpose(self.hidden2[self.iter])) , self.bias_hidden2_out))
                    
                # update within recurrent weights    
                if (self.iter%(self.nSampleInLoop) > 1):
                    updatew_hidden2_hidden2 = eta*(outer(deltah0_2,self.hidden2[self.iter-1]))
                    updatew_hidden1_hidden1 = eta*(outer(deltah0_1,self.hidden1[self.iter-1]))
                    
                if (self.iter%(self.nSampleInLoop) > 2):
                    updatew_hidden2_hidden2 += eta*(outer(deltah2_2,self.hidden2[self.iter-2]))
                    updatew_hidden1_hidden1 += eta*(outer(deltah1_1,self.hidden1[self.iter-2]))


                if (self.iter%(self.nSampleInLoop) > 0):
                    
                    self.weights_in_hidden1_delay += updatew_in_hidden1_delay
                    self.weights_in_hidden2_delay += updatew_in_hidden2_delay

                
                if self.iter%(self.nSampleInLoop) > 0:
                    self.weights_in_hidden1 += updatew_in_hidden1
                    self.weights_in_hidden2 += updatew_in_hidden2
                    self.weights_hidden1_out += updatew_hidden1_out
                    self.weights_hidden2_out += updatew_hidden2_out
                    self.weights_bias1_out += updatew_bias1_out
                    self.weights_bias2_out += updatew_bias2_out
                    self.weights_hidden2_hidden2 += updatew_hidden2_hidden2
                    self.weights_hidden1_hidden1 += updatew_hidden1_hidden1

                    self.weights_in_hidden1_delay = numpy.clip(self.weights_in_hidden1_delay, 0.0, numpy.inf)
                    self.weights_in_hidden2_delay = numpy.clip(self.weights_in_hidden2_delay, 0.0, numpy.inf)
                    self.weights_in_hidden1 = numpy.clip(self.weights_in_hidden1, 0.0, numpy.inf)
                    self.weights_in_hidden2 = numpy.clip(self.weights_in_hidden2, 0.0, numpy.inf)
                    self.weights_hidden1_out = numpy.clip(self.weights_hidden1_out, 0.0, numpy.inf)
                    self.weights_hidden2_out = numpy.clip(self.weights_hidden2_out, 0.0, numpy.inf)

                    # ----- Update of intrinsic plasticity ----- (Eq. 9 and 10)

                    inv_mu1 = 1.0/self.__mu1
                    yz_1 = self.hidden1_y[self.iter]*self.hidden1_z[self.iter]
                    dA_1 = 1.0/self.__a1 + self.hidden1_y[self.iter] - 2*yz_1 - inv_mu1*yz_1 + inv_mu1*yz_1*self.hidden1_z[self.iter]
                    dB_1 = 1.0 - 2*self.hidden1_z[self.iter] - inv_mu1*self.hidden1_z[self.iter] + inv_mu1*self.hidden1_z[self.iter]*self.hidden1_z[self.iter]
                    self.__a1 += self.__eta_a * dA_1
                    self.__b1 += self.__eta_b * dB_1

                    inv_mu2 = 1.0/self.__mu2
                    yz_2 = self.hidden2_y[self.iter]*self.hidden2_z[self.iter]
                    dA_2 = 1.0/self.__a2 + self.hidden2_y[self.iter] - 2*yz_2 - inv_mu2*yz_2 + inv_mu2*yz_2*self.hidden2_z[self.iter]
                    dB_2 = 1.0 - 2*self.hidden2_z[self.iter] - inv_mu2*self.hidden2_z[self.iter] + inv_mu2*self.hidden2_z[self.iter]*self.hidden2_z[self.iter]
                    self.__a2 += self.__eta_a * dA_2
                    self.__b2 += self.__eta_b * dB_2 
                    # ----- End of update of intrinsic plasticity -----
                                
            # normalization
            for i in range(self.nhidden1):
                self.weights_in_hidden1_delay[i] /= numpy.linalg.norm(self.weights_in_hidden1_delay[i])
                self.weights_in_hidden1[i] /= numpy.linalg.norm(self.weights_in_hidden1[i])
                self.weights_hidden1_hidden1[i] /= numpy.linalg.norm(self.weights_hidden1_hidden1[i])
            for i in range(self.nhidden2):
                self.weights_in_hidden2_delay[i] /= numpy.linalg.norm(self.weights_in_hidden2_delay[i])
                self.weights_in_hidden2[i] /= numpy.linalg.norm(self.weights_in_hidden2[i])
                self.weights_hidden2_hidden2[i] /= numpy.linalg.norm(self.weights_hidden2_hidden2[i])
            
            self.true_w_hidden1_out = self.weights_hidden1_out[:, 0:self.nhidden1]
            self.true_w_hidden2_out = self.weights_hidden2_out[:, 0:self.nhidden2]
            


            if n%self.visual_step == 0:
                
                KT.exporttiles(self.hidden1, self.ndata*self.nLayer, self.nLayer, basedir+"obs_H_1.pgm")
                KT.exporttiles(self.hidden2, self.ndata*self.nLayer, self.nhidden2, basedir+"obs_G_1.pgm")
                KT.exporttiles(self.hidden1_z, self.ndata*self.nLayer, self.nLayer, basedir+"obs_H_3.pgm")
                KT.exporttiles(self.hidden2_z, self.ndata*self.nLayer, self.nhidden2, basedir+"obs_G_3.pgm")
                KT.exporttiles(self.__a1, 1, self.nhidden1, basedir+"obs_A_1.pgm")
                KT.exporttiles(self.__b1, 1, self.nhidden1, basedir+"obs_B_1.pgm")
                KT.exporttiles(self.__a2, 1, self.nhidden2, basedir+"obs_C_1.pgm")
                KT.exporttiles(self.__b2, 1, self.nhidden2, basedir+"obs_D_1.pgm")
                KT.exporttiles(self.weights_bias1_out, self.size_in_h*self.nLayer, self.size_in_w, basedir+"obs_E_3.pgm")
                KT.exporttiles(self.weights_bias2_out, self.size_in_h*self.nLayer, self.size_in_w, basedir+"obs_F_3.pgm")
                KT.exporttiles(self.weights_in_hidden1, self.size_in_h*self.nLayer, self.size_in_w, basedir+"obs_W_1_0.pgm", self.nhidden1, 1)
                KT.exporttiles(self.weights_in_hidden2, self.size_in_h*self.nLayer, self.size_in_w, basedir+"obs_V_1_0.pgm", self.nhidden2, 1)
                KT.exporttiles(self.weights_in_hidden1_delay, self.size_in_h*self.nLayer, self.size_in_w, basedir+"obs_W_0_0.pgm", self.nhidden1, 1)
                KT.exporttiles(self.weights_in_hidden2_delay, self.size_in_h*self.nLayer, self.size_in_w, basedir+"obs_V_0_0.pgm", self.nhidden2, 1)
                
                KT.exporttiles(transpose(self.true_w_hidden1_out)+0.5, self.size_in_h*self.nLayer, self.size_in_w, basedir+"obs_W_2_1.pgm", self.nhidden1, 1)
                KT.exporttiles(transpose(self.true_w_hidden2_out)+0.5, self.size_in_h *self.nLayer, self.size_in_w, basedir+"obs_V_2_1.pgm", self.nhidden2, 1)
                
                KT.exporttiles(self.weights_hidden2_hidden2, self.nhidden2, self.nhidden2, basedir+"obs_V_1_1.pgm")
                KT.exporttiles(self.weights_hidden1_hidden1, self.nhidden1, self.nhidden1, basedir+"obs_W_1_1.pgm")
                
                KT.exporttiles(outputs_sav + 0.5, self.size_in_h*self.nLayer, self.size_in_w, basedir+"obs_O_2.pgm", self.nLayer, self.ndata)
                KT.exporttiles(horProduct1_sav + 1.0, self.size_in_h*self.nLayer, self.size_in_w, basedir+"obs_M_2.pgm", self.nLayer, self.ndata)
                KT.exporttiles(horProduct2_sav + 1.0, self.size_in_h*self.nLayer, self.size_in_w, basedir+"obs_N_2.pgm", self.nLayer, self.ndata)
            
                print 'iteration', n, 'error', error
                if abs(error-old_error)<1e-8:
                    print 'no more improvements during training, existing..'
                    break
                if (error - old_error) > 1e2 and (old_error != 0):
                    print error, old_error
                    print 'error increasing, existing..'
                    break


    def mlpfwd(self,inputs, previous_inputs):
        """ Run the network forward """
        # 1 . Hidden layer input includes weighted sum of input, recurrent connection of hidden layer itself, and the delayed input (Eq. 2 and 3)
        # 2.  Different from the paper, here we only use logistic function but not soft-max as transfer function (only Eq. 4)

        # for learning only after movement
        if  self.iter%(self.nSampleInLoop)  > 0 :
            hidden1_old = self.hidden1[self.iter-1]
            hidden2_old = self.hidden2[self.iter-1]
        else:
            hidden1_old = numpy.zeros((self.nhidden1)) #self.hidden1[0]
            hidden2_old = numpy.zeros((self.nhidden2))

        if  self.iter%(self.nLayer) == 1:  # keep ventral  layer  update when a new object comes
            
            self.hidden1_y[self.iter] = self.__forward_alpha_v * self.hidden1_y[self.iter-1] + (1-self.__forward_alpha_v) * (dot(self.weights_in_hidden1,inputs) + dot(self.weights_in_hidden1_delay, previous_inputs)) + dot(self.weights_hidden1_hidden1,hidden1_old)
            tmp = self.__a1 * self.hidden1_y[self.iter] + self.__b1
            self.hidden1[self.iter] = 1.0 / (1.0 + numpy.exp(-tmp)) 
        elif self.iter%(self.nLayer) == 0: # clear the memory when a new object comes
            self.hidden1[self.iter] = numpy.zeros((self.nhidden1))
        else:  # keep layer ventral when the object is the same
            self.hidden1_y[self.iter] = self.hidden1_y[self.iter - 1]
            self.hidden1[self.iter] = self.hidden1[self.iter - 1]

        if self.iter%self.nLayer == 0:  # clear memory when a new object comes in dorsal layer too
            hidden2_old = numpy.zeros((self.nhidden2))
        

        self.hidden2_y[self.iter] = self.__forward_alpha_d * self.hidden2_y[self.iter-1] + (1-self.__forward_alpha_d) * (dot(self.weights_in_hidden2,inputs) + dot(self.weights_in_hidden2_delay, previous_inputs)) + dot(self.weights_hidden2_hidden2,hidden2_old)
        tmp = self.__a2 * self.hidden2_y[self.iter] + self.__b2
        self.hidden2[self.iter] = 1.0 / (1.0 + numpy.exp(-tmp))
        
        

        self.horProduct1 = dot(self.weights_hidden1_out,self.hidden1[self.iter]) + dot(self.weights_bias1_out, self.bias_hidden1_out)
        self.horProduct2 = dot(self.weights_hidden2_out,self.hidden2[self.iter]) + dot(self.weights_bias2_out, self.bias_hidden2_out)
        self.horProduct1 = numpy.clip(self.horProduct1, 0.0, numpy.inf) 
        self.horProduct2 = numpy.clip(self.horProduct2, 0.0, numpy.inf)
        
        outputs = self.horProduct1*self.horProduct2

        # Different types of output neurons
        if self.outtype == 'linear':
            return outputs
        elif self.outtype == 'logistic':
            return 1.0/(1.0+exp(-self.beta*outputs))
        elif self.outtype == 'softmax':
            normaliser = sum(exp(outputs))
            return exp(outputs)/normaliser
        else:
            print "error"




basedir = "/tmp/coco/"




if __name__ == "__main__":

    nIteration = 1000 # maximum iteration
    
    nObject = 4 
    
    beta = 1.0
    
    outtype = 'linear'
    eta = 0.03
    eta_bias = 0.02
    
    
    size = 5 # edge length of the grid
  
    mlp = MLP(size, nObject, beta, outtype)
    mlp.mlptrain(eta_bias, eta, nIteration)
    print 'training complete!'

    
