# MLP network for award-driven continuous motor action 
#
# Author: J Zhong
# Date: 01-18-2013
# zhong@informatik.uni-hamburg.de
#
# Two parts, both based on MLP model, share the same two input units, but use different hidden layers.
# 1. In order to avoid the angle adjustment, we use two separate units as actor outputs representing x and y axis
# 2. Output is normalised. Only the angle from the invernt tangent is used to move the agent.
# 3. The actor MLP learnining is guided by critic value
# 


# Please refer to: First CACLA implementation as described at:
# http://homepages.cwi.nl/~hasselt/rl_algs/Cacla.html

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


from pylab import *

import numpy
import random
import KTimage as KT
import math


class world_model_RL:
    def __init__(self, reward_x, reward_y):
        # init input position
        self.reward_x = reward_x
        self.reward_y = reward_y
        self.start_x = 0.0
        self.start_y = 0.1
        self.sel_x = self.start_x
        self.sel_y = self.start_y

        self.StartPosCount = 10

      
    def newinit(self):
        

        if self.start_x < 0.9:
            self.start_x += 1.0/self.StartPosCount
        else:
            self.start_x = 0.1
            if self.start_y < 0.9:
                self.start_y += 1.0/self.StartPosCount
            else:
                self.start_y = 0.1
        
            
        self.sel_x = self.start_x
        self.sel_y = self.start_y
        
    def act(self, act): #act is [x,y], normalised
        # position world reaction
        
        self.sel_x +=  act[0]*0.06
        self.sel_y +=  act[1]*0.06 
        

        # position boundary conditions
        if  self.sel_x < 0.0 or self.sel_x > 1.0:
            self.sel_x -= act[0]*0.06
            self.sel_y -=  act[1]*0.06
        
        if  self.sel_y < 0.0 or self.sel_y > 1.0:
            self.sel_x -= act[0]*0.06
            self.sel_y -=  act[1]*0.06
        
       
        
    def reward(self):  # reward area is 0.2*0.2 area
        if  self.sel_x >= self.reward_x - 0.1 and self.sel_x <= self.reward_x + 0.1 and self.sel_y >= self.reward_y - 0.1 and self.sel_y <= self.reward_y + 0.1:
            return 1.0
        else:
            return 0.0
        
    def sense(self):
        
        return self.sel_x, self.sel_y

    def rand_winner (self, h, sigma):
        rand = random.normalvariate(h, sigma)
        if  rand < 0.0:
            rand += 2.0 * math.pi
        elif rand >= 2.0 * math.pi:
            rand -= 2.0 * math.pi

        return rand
        
    
      

class RNN:
    """ A Multi-Layer RNN"""

    def __init__(self, Perception, outtype='linear'):
        """ Constructor """
        # Set up network size
        self.__perception = Perception
        self.eta = 0.10 # learning rate
        self.sigma = 0.2*math.pi # sigma for random action selection.
        self.beta = 1.0
        self.beta1 = 1.0
        self.sm_beta = 90.0
        self.eps = 0.10
        
        self.nin = 2
        self.nout = 2
        self.ncritic = 1
        self.nhidden = 5 #TODO: determine the size of hidden layer
        print 'number of hidden units', self.nhidden
        

        self.outtype = outtype
    
        # Initialise network
        self.weights11 = (1*numpy.random.rand(self.nhidden,self.nin+1)-0.5)*2/sqrt(self.nin+1) #weights between input and hidden1 (action)
        self.weights12 = (1*numpy.random.rand(self.nhidden,self.nin+1)-0.5)*2/sqrt(self.nin+1) #weights between input and hidden2 (critic)
        self.weights2 = (1*numpy.random.rand(self.nout,self.nhidden+1)-0.5)*2/sqrt(self.nhidden+1)  #weights between hidden and output
        self.weights4 = (1*numpy.random.rand(self.ncritic,self.nhidden+1)-0.5)*2/sqrt(self.nhidden+1) # weights between critic and hidden
        


    def train(self,iteration):
        gamma = 0.9
        
        """ Train the thing """    
        # Add the inputs that match the bias node
        self.maxIteration = iteration
        
        
        self.error_action_sav = numpy.zeros((2,self.maxIteration/10))
        self.error_val_sav = numpy.zeros((self.maxIteration/10))
        self.error_total_action_sav = numpy.zeros((2,self.maxIteration/10))
        self.error_total_val_sav = numpy.zeros((self.maxIteration/10))
        self.average_duration_sav = numpy.zeros((self.maxIteration))
        
        for iteration in range(0,self.maxIteration):

            error = 0.0
            error_val = 0.0
            total_duration = 0
            
            updateTimes = 0

            self.visualisation()
            
            for StartPos in range(0, self.__perception.StartPosCount**2):
                self.__perception.newinit()
                
                reward = 0.0
                duration = 0
            
                self.hidden = numpy.zeros((self.nhidden))
            
                landmark_info = numpy.array(self.__perception.sense()) # 2 continuous inputs

            
                inputs_bias = numpy.concatenate((landmark_info,-1*numpy.ones((1))),axis=0)
                
                # ------- sigmoidal funcion for hidden layer 1 -------
                self.hidden1_y = dot(self.weights11,inputs_bias)
                self.hidden1 = 1.0 / (1.0 + numpy.exp(-self.beta1*self.hidden1_y))
                self.hiddenplusone1 = numpy.concatenate((self.hidden1,-1*numpy.ones((1))),axis=0) # add a bias unit
                # ------- end of sigmoidal funcion for hidden layer 1 -------
                h_out = self.mlpfwd()
                h = self.normalised(h_out)

                h_angle = math.atan2(h[1], h[0])
                action_angle = self.__perception.rand_winner(h_angle,self.sigma)

                action = numpy.zeros((2))
                action[0] = math.cos(action_angle)
                action[1] = math.sin(action_angle)
                action = self.normalised(action)

                # ------- sigmoidal funcion for hidden layer 2 -------
                self.hidden2_y = dot(self.weights12,inputs_bias)
                self.hidden2 = 1.0 / (1.0 + numpy.exp(-self.beta*self.hidden2_y))
                self.hiddenplusone2 = numpy.concatenate((self.hidden2,-1*numpy.ones((1))),axis=0) # add a bias unit
                # ------- end of sigmoidal funcion for hidden layer 2 -------
                val = self.valfwd()
                
                r = self.__perception.reward()   # read reward

                
                
                while (r != 1.0) and duration < 1000:
                
                    duration += 1
                    total_duration += 1
                    updatew11 = numpy.zeros((numpy.shape(self.weights11)))
                    updatew12 = numpy.zeros((numpy.shape(self.weights12)))
                    updatew2 = numpy.zeros((numpy.shape(self.weights2)))
                    updatew4 = numpy.zeros((numpy.shape(self.weights4)))
                    self.__perception.act(action)
                    landmark_info_tic = numpy.array(self.__perception.sense()) # 2 continuous inputs
                    r = self.__perception.reward()   # read reward
                    inputs_bias_tic = numpy.concatenate((landmark_info_tic,-1*numpy.ones((1))),axis=0)
                    
                    KT.exporttiles(inputs_bias_tic, self.nin+1, 1, basedir+"obs_S_0.pgm")

                    # ------- sigmoidal funcion for hidden layer 1 -------
                    self.hidden1_y = dot(self.weights11,inputs_bias_tic)
                    self.hidden1 = 1.0 / (1.0 + numpy.exp(-self.beta1*self.hidden1_y))
                    self.hiddenplusone1 = numpy.concatenate((self.hidden1,-1*numpy.ones((1))),axis=0) # add a bias unit
                    # ------- end of sigmoidal funcion for hidden layer 1 -------
                

                    h_out = self.mlpfwd()
                    h = self.normalised(h_out)

                    h_angle = math.atan2(h[1], h[0])
                    action_tic_angle = self.__perception.rand_winner(h_angle,self.sigma)
                    
                    action_tic = numpy.zeros((2))
                    action_tic[0] = math.cos(action_tic_angle)
                    action_tic[1] = math.sin(action_tic_angle)
                    action_tic = self.normalised(action_tic)

                
                    

                    # ------- sigmoidal funcion for hidden layer 2 -------
                    self.hidden2_y = dot(self.weights12,inputs_bias_tic)
                    self.hidden2 = 1.0 / (1.0 + numpy.exp(-self.beta*self.hidden2_y))
                    self.hiddenplusone2 = numpy.concatenate((self.hidden2,-1*numpy.ones((1))),axis=0) # add a bias unit
                    # ------- end of sigmoidal funcion for hidden layer 2 -------

                    

                    if self.__perception.sel_x > 0.1 and self.__perception.sel_y > 0.1 and self.__perception.sel_x < 0.3 and self.__perception.sel_y < 0.3:
                        KT.exporttiles(self.hidden1, 1, self.nhidden, basedir+"obs_S_1.pgm")
                        KT.exporttiles(self.hidden2, 1, self.nhidden, basedir+"obs_S_2.pgm")

                    if self.__perception.sel_x > 0.6 and self.__perception.sel_y > 0.6 and self.__perception.sel_x < 0.7 and self.__perception.sel_y < 0.7:
                        KT.exporttiles(self.hidden1, 1, self.nhidden, basedir+"obs_A_1.pgm")
                        KT.exporttiles(self.hidden2, 1, self.nhidden, basedir+"obs_A_2.pgm")
                
                    val_tic = self.valfwd()
                    
                    # ----- here are the training process--------#

                    if  r == 1.0:                                   # reward achieved
                        target = r 
                    else:                                           # because critic weights now converge.
                        target = gamma * val_tic                    # gamma = 0.9
                                                                    # prediction error; 
                    deltao = (target-val)
                
                    
                    error_val += abs(deltao)
                
                    
                    
                    
                    updatew4 = self.eps * (outer(deltao,self.hiddenplusone2))
                    deltah0 = self.hiddenplusone2*(1-self.hiddenplusone2)*(dot(transpose(self.weights4),deltao))
                    updatew12 = self.eta * (outer(deltah0[:-1],inputs_bias_tic))
                    

                    self.weights12 += updatew12
                    self.weights4 += updatew4
                       
                    if gamma * val_tic > val or r == 1.0:
                        updateTimes += 1
                        error += abs(action  - h)
                        deltao2 =  (action-h) / numpy.linalg.norm(action-h)
                        deltah0  = self.hiddenplusone1 * (1-self.hiddenplusone1) * dot(transpose(self.weights2),deltao2) 
                        updatew11 = self.eta*(outer(deltah0[:-1],inputs_bias_tic))
                        
                        self.weights11 += updatew11
                        updatew2 = self.eta * (outer(deltao2, self.hiddenplusone1))
                        self.weights2 += updatew2
                        
                    ##-------------end update when the critic are higher-----------##
                    
                    
                    landmark_info = landmark_info_tic
                
                    action = action_tic
                    val = val_tic
               

            if (iteration%1 == 0):
                print "iteration:", iteration
                print "Error in val:",  error_val, "average per move:", error_val/float(total_duration+1)
                print "Error in action:", error, "average per move:", error/float(updateTimes+1)
                print "Total duration:", total_duration
                print "Average duration", total_duration / (self.__perception.StartPosCount**2)
                print "Update Times:", updateTimes
            

            self.average_duration_sav[iteration] = total_duration / (self.__perception.StartPosCount**2)


    def ploterror(self):
        t = range(0, self.maxIteration)
        plot(t, self.average_duration_sav)
        show()
        t = range(0, self.maxIteration, 10)
        plot(t, self.average_duration_sav)
        show()
        plot(t, self.error_action_sav[0,:])
        plot(t, self.error_action_sav[1,:])
        show()
        plot(t, self.error_val_sav)
        show()
        plot(t, self.error_total_action_sav[0,:])
        plot(t, self.error_total_action_sav[1,:])
        show()
        plot(t, self.error_total_val_sav)
        show()

    def visualisation(self):
        KT.exporttiles(self.weights11[:,0:-1], self.nhidden, self.nin, basedir+"obs_V_1_0.pgm")
        KT.exporttiles(self.weights12[:,0:-1], self.nhidden, self.nin, basedir+"obs_W_1_0.pgm")
        KT.exporttiles(self.weights2[:,0:-1], self.nhidden, self.nout, basedir+"obs_V_2_1.pgm")
        KT.exporttiles(self.weights4[:,0:-1], self.nhidden, self.ncritic, basedir+"obs_W_2_1.pgm")
        print 'visualisation updated!!'
        

        
    def mlpfwd(self):
        """ Run the network forward """
        
        outputs = dot(self.weights2,self.hiddenplusone1)
        return outputs
        
        
        
    def valfwd(self):
        """ Run the network forward """
        outputs = dot(self.weights4,self.hiddenplusone2)
        return outputs
        
    
    def normalised(self, h):
        
        outputs = h / numpy.linalg.norm(h)
        return outputs


    def test_trajectory(self):

        Count = 100.0 # we need higher reslution for plotting the action

        self.action_sav = numpy.zeros((Count, Count))

        for x in range(int(Count)):
            for y in range(int(Count)):
        
            
                landmark_info = numpy.array([(x+1)/Count, (y+1)/Count]) # 2 continuous inputs
                inputs_bias = numpy.concatenate((landmark_info,-1*numpy.ones((1))),axis=0)
                
                # ------- sigmoidal funcion for hidden layer 1 -------
                self.hidden1_y = dot(self.weights11,inputs_bias)
                self.hidden1 = 1.0 / (1.0 + numpy.exp(-self.beta1*self.hidden1_y))
                self.hiddenplusone1 = numpy.concatenate((self.hidden1,-1*numpy.ones((1))),axis=0) # add a bia s unit
                # ------- end of sigmoidal funcion for hidden layer 1 -------

                h_out = self.mlpfwd()
                h = self.normalised(h_out)

                h_angle = math.atan2(h[1], h[0])

                self.action_sav[x,y] = h_angle
                

        

basedir = "/tmp/coco/"

if __name__ == "__main__":
    Perception = world_model_RL(0.55, 0.85)
    Prediction = RNN(Perception, 'linear')
    Prediction.train(1500)

    Prediction.test_trajectory()
    pcolor(Prediction.action_sav, cmap=cm.RdBu, vmax=abs(Prediction.action_sav).max(), vmin=-abs(Prediction.action_sav).max())
    colorbar()
    show()
    #Prediction.predict()



    
    

    
