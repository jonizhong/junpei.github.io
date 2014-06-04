#!/usr/bin/env python
# $Author$
# $LastChangedDate$
# $Rev$
'''

Ver 2
Author: J Zhong
Date: 01-24-2013
zhong@informatik.uni-hamburg.de
'''
'''
   IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
 
   By downloading, copying, installing or using the software you agree to 
   this license. If you do not agree to this license, do not download, install,
   copy or use the software.
 
 
                            License Agreement
                     For   RNN Elman SRN with PB	
 
  Copyright (C) 2011

  Jens Kleesiek <kleesiek@informatik.uni-hamburg.de>

  all rights reserved.

  Third party copyrights are property of their respective owners.
 
  Redistribution and use in source and binary forms, with or without 
  modification, are permitted provided that the following conditions are met:
 
    1) Redistribution's of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.
 
    2) Redistribution's in binary form must reproduce the above copyright 
    notice, this list of conditions and the following disclaimer in the 
    documentation and/or other materials provided with the distribution.

    3) All advertising materials mentioning features or use of this software
    must display the following acknowledgement: "This product includes 
    software developed by Jens Kleesiek"

    4) The name of the copyright holders may not be used to endorse or promote
    products derived from this software without specific prior written 
    permission.

 THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS "AS IS" AND ANY 
 EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED 
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE 
 DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY 
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES 
 (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; 
 LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND 
 ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT 
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS 
 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''
import logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s [%(levelname)s](%(funcName)s):' \
                        '%(message)s')#,
#                    filename='info.log',
#                    filemode = 'a') 
logger = logging.getLogger()

import platform,sys
import matplotlib
if platform.uname()[0] == 'Darwin':
    matplotlib.use('MacOSX')
    logger.info('Running on Mac')
elif platform.uname()[0] == 'Linux':
    matplotlib.use('Agg')
    logger.info('Running on Linux')
elif platform.uname()[0] == "Windows":
    matplotlib.use('Agg')
    logger.info('Running on Windows')
else:
    logger.error('Unknown host system')
import matplotlib.pyplot as P
import numpy as N
import data as D
import retina as R
import KTimage as KT

logger.info('RNN Elman SRN with PB')

class RNNPB(object):
    def __init__(self, nin, nhid, nhh, ncont, nout, npb, lrate, momentum, logger, 
                 signalLength, 
                 convergenceThreshold=0.000001, plot=True, recurrent=True):
        """ elman neuronal network """
        self.recurrent = recurrent
        """ data """
        self.timeSteps = signalLength
        
        """ Parameters """
        self.ni = nin
        self.nh1 = nhid + 1
        self.nh2 = 2
        self.nhh = nhh + 1
        
        self.no = nout
        self.np1 = npb
        self.np2 = npb
        
        self.eta = lrate # 0.001 # learning rate 
        
        self.eta_max = 100.0*lrate   # update rate for 'where'
        self.eta_min = lrate * 0.001 # update rate for 'what'
        
        
        

        #self.eta_ih_1 = self.eta
        #self.eta_ho_1 = self.eta
        #self.eta_ch_1 = self.eta
        #self.eta_ph_1 = self.eta
        #self.eta_ih_2 = self.eta
        #self.eta_ho_2 = self.eta
        #self.eta_ch_2 = self.eta
        #self.eta_ph_2 = self.eta
        
        '''
        '''

        #adaptve learning rate
        #self.eta_hh_1 = N.ones((self.nh1,self.nhh))*self.eta
        self.eta_ih_1 = N.ones((self.nh1,self.ni)) * self.eta
        self.eta_ho_1 = N.ones((self.no,self.nh1)) * self.eta
        self.eta_ch_1 = N.ones((self.nh1,self.nh1)) * self.eta
        self.eta_ph_1 = N.ones((self.nh1,self.np1)) * self.eta
        self.eta_ih_2 = N.ones((self.nh2,self.ni)) * self.eta_min
        self.eta_ho_2 = N.ones((self.no,self.nh2)) * self.eta_min
        self.eta_ch_2 = N.ones((self.nh2,self.nh2)) * self.eta_min
        self.eta_ph_2 = N.ones((self.nh2,self.np2)) * self.eta
        
        
        self.negative_fac_eta = 0.999999
        self.positive_fac_eta = 1.000001
        
        if self.recurrent:
            self.eta_hc = lrate
        
        self.gamma_factor = lrate * 0.005 #0.05# learning rate PB constant
        self.gamma_1 = 0.00005
        self.gamma_2 = 0.00005
        self.momentum = momentum
        self.convergence = convergenceThreshold

        
        """ Weights """
        scale = 1.0
        self.Wih_1 = (N.random.rand(self.nh1,self.ni)-0.5)*2/N.sqrt(self.ni) 
        self.Wih_2 = (N.random.rand(self.nh2,self.ni)-0.5)*2/N.sqrt(self.ni)
        self.Who_1 = (N.random.rand(self.no,self.nh1)-0.5)*2/N.sqrt(self.nh1)
        self.Who_2 = (N.random.rand(self.no,self.nh2)-0.5)*2/N.sqrt(self.nh2)
        # self.Whh_1 = (N.random.rand(self.nh1,self.nhh)-0.5)*2/N.sqrt(self.nhh)
        self.Wch_1 = (N.random.rand(self.nh1,self.nh1)-0.5)*2/N.sqrt(self.nh1)
        self.Wch_2 = (N.random.rand(self.nh2,self.nh2)-0.5)*2/N.sqrt(self.nh2)
        if self.recurrent:
            self.Whc_1 = (N.random.rand(self.nh1,self.nh1)-0.5*2)/N.sqrt(self.nh1)
            self.Whc_2 = (N.random.rand(self.nh2,self.nh2)-0.5*2)/N.sqrt(self.nh2)
        else:
            self.Whc_1 = N.eye(self.nh1)
            self.Whc_2 = N.eye(self.nh2)
        self.Wph_1 = N.random.uniform(-scale,scale,(self.nh1,self.np1))/(self.np1)
        self.Wph_2 = N.random.uniform(-scale,scale,(self.nh2,self.np2))/(self.np2)
        self.dWih_1 = N.zeros((self.nh1,self.ni))
        self.dWih_2 = N.zeros((self.nh2,self.ni))
        self.dWho_1 = N.zeros((self.no,self.nh1))
        self.dWho_2 = N.zeros((self.no,self.nh2))
        self.dWch_1 = N.zeros((self.nh1,self.nh1))
        self.dWch_2 = N.zeros((self.nh2,self.nh2))
        self.dWhc_1 = N.zeros((self.nh1,self.nh1))
        self.dWhc_2 = N.zeros((self.nh2,self.nh2))
        self.dWph_1 = N.zeros((self.nh1,self.np1))
        self.dWph_2 = N.zeros((self.nh2,self.np2))
        self.dWhh_1 = N.zeros((self.nh1,self.nhh))
        # delta weights backup - used for adpative learning rate
        self.dWih_1_old = N.zeros((self.nh1,self.ni))
        self.dWih_2_old = N.zeros((self.nh2,self.ni))
        self.dWho_1_old = N.zeros((self.no,self.nh1))
        self.dWho_2_old = N.zeros((self.no,self.nh2))
        self.dWch_1_old = N.zeros((self.nh1,self.nh1))
        self.dWch_2_old = N.zeros((self.nh2,self.nh2))
        self.dWhc_1_old = N.zeros((self.nh1,self.nh1))
        self.dWhc_2_old = N.zeros((self.nh2,self.nh2))
        self.dWph_1_old = N.zeros((self.nh1,self.np1))
        self.dWph_2_old = N.zeros((self.nh2,self.np2))
        self.dWhh_1_old = N.zeros((self.nh1,self.nhh))
        """ Activities """
        self.act_i = N.zeros((self.timeSteps,self.ni))
        self.act_p_1 = N.zeros(self.np1)
        self.act_p_2 = N.zeros(self.np2)
        self.rho_1 = N.zeros(self.np1)
        self.rho_2 = N.zeros(self.np2)
        self.act_h_1 = N.zeros((self.timeSteps,self.nh1))
        self.act_h_2 = N.zeros((self.timeSteps,self.nh2))
        self.act_hh_1 = N.zeros((self.timeSteps,self.nhh))
        self.act_c_1 = N.zeros((self.timeSteps,self.nh1))
        self.act_c_2 = N.zeros((self.timeSteps,self.nh2))
        self.act_c_1[0] = 0.5 * N.ones(self.nh1) # initial context
        self.act_c_1[0][-1] = -1.0
        self.act_c_2[0] = 0.5 * N.ones(self.nh2) # initial context
        self.act_c_2[0][-1] = -1.0
        self.act_o = N.zeros((self.timeSteps,self.no))

        self.z_1 = N.zeros((self.timeSteps, self.no))
        self.z_2 = N.zeros((self.timeSteps, self.no))
        """ Utils """
        self.logger = logger
        """ Plotting """
        self.plot = plot
        if self.plot == True:
            self.initPlotWeights()
            

                
        
    def resetActivities(self,t):
        """ Activities """
        self.timeSteps = t
        self.act_i = N.zeros((self.timeSteps,self.ni))
        self.act_h_1 = N.zeros((self.timeSteps,self.nh1))
        self.act_h_2 = N.zeros((self.timeSteps,self.nh2))
        self.act_hh_1 = N.zeros((self.timeSteps,self.nhh))
        self.act_c_1 = N.zeros((self.timeSteps,self.nh1))
        self.act_c_2 = N.zeros((self.timeSteps,self.nh2))
        self.act_c_1[0] = 0.5 * N.ones(self.nh1) # initial context
        self.act_c_1[0][-1] = -1.0
        self.act_c_2[0] = 0.5 * N.ones(self.nh2) # initial context
        self.act_c_2[0][-1] = -1.0
        self.act_o = N.zeros((self.timeSteps,self.no))
    def saveWeights(self, ext=''):
        N.save(str(ext)+'Wih.npy', self.Wih)
        N.save(str(ext)+'Who.npy', self.Who)
        N.save(str(ext)+'Wch.npy', self.Wch)
        if self.recurrent:
            N.save(str(ext)+'Whc.npy', self.Whc)
        N.save(str(ext)+'Wph.npy', self.Wph)
    def loadWeights(self):
        self.Wih = N.load('Wih.npy')
        self.Who = N.load('Who.npy')
        self.Wch = N.load('Wch.npy')
        if self.recurrent:
            self.Whc = N.load('Whc.npy')
        self.Wph = N.load('Wph.npy')
    def sigmoid(self, y, transFunc='logistic'):
        if transFunc == 'logistic':
            return  1.0/( 1.0 + N.exp(-y))
        elif transFunc == 'tanh':
            return N.tanh(y)
        elif transFunc == 'tanhOpt':
            return 1.7159*N.tanh(2.0/3.0 * y)
        elif transFunc == 'linear':
            return y
        else:
            self.logger.error("unknown transfer function")
    def dSigmoid(self, y, transFunc='logistic'): # different derivative functions
        if transFunc == 'logistic':
            return y * (1.0 - y)
        elif transFunc == 'tanh':
            return 1.0 - y**2
        elif transFunc == 'tanhOpt':
            return 2.0/3.0*1.7159 - 2.0/3.0*y**2
        elif transFunc == 'linear':
            return 1.0
        else:
            self.logger.error("unknown transferfunction")
    def forwardTS(self, signalIn, signalOut, beta, color, transFunc='logistic'):
        error = 0.
        for t in range(N.shape(signalIn)[0]):
            if t == 0:
                self.act_i[t] = signalIn[t]
            else:
                self.act_i[t] = (1.0-beta) * self.act_o[t-1,] + \
                    beta * signalIn[t]
                if N.any(N.nonzero(self.act_i[t] < -1.5)) or \
                        N.any(N.nonzero(self.act_i[t] > 1.5)):
                    self.logger.info("input out of range")
                    #self.act_i[t] = N.clip(self.act_i[t],0.,1.)
            y_h_1 = N.dot(self.Wih_1,self.act_i[t]) + \
                 N.dot(self.Wch_1,self.act_c_1[t])  + \
                 N.dot(self.Wph_1,self.act_p_1) # PB
            y_h_2 = N.dot(self.Wih_2,self.act_i[t]) + \
                 N.dot(self.Wch_2,self.act_c_2[t]) # + \
                #N.dot(self.Wph_2,self.act_p_2)
            self.act_h_1[t] = self.sigmoid(y_h_1,'logistic')
            self.act_h_1[t][-1] = -1.0# set bias neuron
                        
            #self.act_h_2[t] = self.sigmoid(y_h_2,'logistic')    
            #if color == 0:
            #    self.act_h_2[t, 0] = 1.0            
            #else:
            #    self.act_h_2[t,0] = -1.0
            
                
            self.act_h_2[t] = self.sigmoid(y_h_2,'logistic')
            self.act_h_2[t][-1] = -1.0 # set bias neuron
            
           
            #print self.act_h_2[t]
            # copy hidden layer to context layer
            if t < self.timeSteps-1:
                 c_1 = N.dot(self.Whc_1,self.act_h_1[t])
                 self.act_c_1[t+1] = self.sigmoid(c_1,'linear')
                 c_2 = N.dot(self.Whc_2,self.act_h_2[t])
                 self.act_c_2[t+1] = self.sigmoid(c_2,'linear')
            #output layer
            self.z_1[t] = N.dot(self.Who_1,self.act_h_1[t])
            self.z_2[t] = N.dot(self.Who_2,self.act_h_2[t])            # compute error
            self.act_o[t] = self.sigmoid(N.multiply(self.z_1[t],self.z_2[t]),'linear')
            #self.act_o[t] = self.sigmoid(self.z_1[t],'linear')
            error += (signalOut[t]-self.act_o[t])**2
#N.sqrt(signalOut[t]**2+self.act_o[t]**2)
        error /= float(self.timeSteps)
        return error
    def backwardTSrec(self, signalIn, signalOut, transFunc='logistic'):
        # initialize delta's
        delta_o = N.zeros((self.timeSteps,self.no))
        delta_h = N.zeros((self.timeSteps,self.nh))
        delta_c = N.zeros((self.timeSteps,self.nh))
        delta_p = N.zeros((self.timeSteps,self.np))

        for t in range(self.timeSteps-1,-1,-1): # going backward in time ...
            # compute deltas
            # error term output units
            delta_o[t] = (signalOut[t] - self.act_o[t])
            delta_o[t] *=self.dSigmoid(self.act_o[t],transFunc)#'linear')
            # error term hidden units
            delta_h[t] = N.dot(N.transpose(self.Who),delta_o[t])
            
            if t < self.timeSteps-1:
                '''
                FYI: delta_c[t+1], i.e. T+1!!! is used, because we need 
                the value of the previous time step (remember: we are 
                currently going back in time, thus +1) that has not been 
                updated, yet!
                '''
                delta_h[t] += N.dot(N.transpose(self.Whc),delta_c[t+1])
            else:
                delta_h[t] += N.dot(N.transpose(self.Whc),N.zeros(self.nh))
            
            delta_h[t] *= self.dSigmoid(self.act_h[t],transFunc)
            # error term parametric bias
            delta_p[t] = N.dot(N.transpose(self.Wph),delta_h[t])
            delta_p[t] *= self.dSigmoid(self.act_p, transFunc)
            # error term context units
            delta_c[t] = N.dot(N.transpose(self.Wch),delta_h[t])
            delta_c[t] *= self.dSigmoid(self.act_c[t],transFunc)
        tmpdEdRho = N.sum(delta_p,axis=0)
        self.rho += self.etaRho*tmpdEdRho
        self.act_p = self.sigmoid(self.rho,transFunc)
        
   
    def backwardTS(self, signalIn, signalOut, transFunc='logistic'):
        # initialize delta's
        delta_o = N.zeros((self.timeSteps,self.no))
        delta_o_1  = N.zeros((self.timeSteps,self.no))
        delta_o_2  = N.zeros((self.timeSteps,self.no))
        delta_h_1 = N.zeros((self.timeSteps,self.nh1))
        delta_h_2 = N.zeros((self.timeSteps,self.nh2))
        delta_c_1 = N.zeros((self.timeSteps,self.nh1))
        delta_c_2 = N.zeros((self.timeSteps,self.nh2))
        delta_p_1 = N.zeros((self.timeSteps,self.np1))
        delta_p_2 = N.zeros((self.timeSteps,self.np2))
        delta_hh_1 = N.zeros((self.timeSteps,self.nhh))
        # and dW's
        dWih_1 = N.zeros((self.nh1,self.ni))
        dWih_2 = N.zeros((self.nh2,self.ni))
        dWho_1 = N.zeros((self.no,self.nh1))
        dWho_2 = N.zeros((self.no,self.nh2))
        dWhh_1 = N.zeros((self.nh1,self.nhh))
        dWch_1 = N.zeros((self.nh1,self.nh1))
        dWch_2 = N.zeros((self.nh2,self.nh2))
        if self.recurrent:
            dWhc_1 = N.zeros((self.nh1,self.nh1))
            dWhc_2 = N.zeros((self.nh2,self.nh2))
        dWph_1 = N.zeros((self.nh1,self.np1))
        dWph_2 = N.zeros((self.nh2,self.np2))

        for t in range(self.timeSteps-1,-1,-1): # going backward in time ...
        
            # print dWch_1
            # compute deltas
            # error term output units
            #print t
            delta_o[t] = (signalOut[t] - self.act_o[t])
            #print signalOut[t]
            #print delta_o[t]
           # print 'signalOut', signalOut[t]
           # print 'act_o', self.act_o[t]
            
            #print signalOut[t]
            #print self.act_o[t]
            #print "delta_o", delta_o[t]
            delta_o[t] *=self.dSigmoid(self.act_o[t],'linear')
            
            delta_o_1[t] = delta_o[t] * self.z_2[t]
            delta_o_2[t] = delta_o[t] * self.z_1[t]
            # --- error term hidden units 1 ---
            
            delta_h_1[t] = N.dot(N.transpose(self.Who_1),delta_o_1[t])
            #delta_h_1[t] = N.dot(N.transpose(self.Who_1),delta_o[t])
            
            if t < self.timeSteps-1:
                delta_h_1[t] += N.dot(N.transpose(self.Whc_1),delta_c_1[t+1])
            else:
                delta_h_1[t] += N.dot(N.transpose(self.Whc_1),N.zeros(self.nh1))
            
            delta_h_1[t] *= self.dSigmoid(self.act_h_1[t],'logistic')
            # --- error term hidden units 2 ---
            delta_h_2[t] = N.dot(N.transpose(self.Who_2),delta_o_2[t])
            #delta_h_2[t] = N.dot(N.transpose(self.Who_2),delta_o[t])
            
            #delta_hh_1[t] = N.dot(N.transpose(self.Whh_1),delta_h_1[t])
            #delta_hh_1[t] *= self.dSigmoid(self.act_hh_1[t],'linear')
            
            if t < self.timeSteps-1:
                delta_h_2[t] += N.dot(N.transpose(self.Whc_2),delta_c_2[t+1])
            else:
                delta_h_2[t] += N.dot(N.transpose(self.Whc_2),N.zeros(self.nh2))
            
            
            delta_h_2[t] *= self.dSigmoid(self.act_h_2[t],'logistic')
            #print "delta_c_2", delta_c_2
            #print "delta_h_1", delta_h_1
            #print "delta_h_2", delta_h_2
            # error term parametric bias 1
            delta_p_1[t] = N.dot(N.transpose(self.Wph_1),delta_h_1[t])
            delta_p_1[t] *= self.dSigmoid(self.act_p_1, 'tanhOpt')
            # error term parametric bias 2
            delta_p_2[t] = N.dot(N.transpose(self.Wph_2),delta_h_2[t])
            delta_p_2[t] *= self.dSigmoid(self.act_p_2, 'tanhOpt')
            # --- error term context units 1 ---
            delta_c_1[t] = N.dot(N.transpose(self.Wch_1),delta_h_1[t])
            delta_c_1[t] *= self.dSigmoid(self.act_c_1[t],'linear')
            # --- error term context units 2 ---
            delta_c_2[t] = N.dot(N.transpose(self.Wch_2),delta_h_2[t])
            delta_c_2[t] *= self.dSigmoid(self.act_c_2[t],'linear')
            # compute weight change at time t
            # hidden to output weights
            tmpD = N.transpose((N.kron(N.ones((self.nh1,1)),
                                       delta_o_1[t])).reshape(self.nh1,self.no))
            tmpAct = N.transpose((N.repeat(self.act_h_1[t],self.no,
                                           axis=0)).reshape(self.nh1,self.no))
            dWho_1 += tmpAct*tmpD
            #dWho_1 = tmpAct*tmpD
            #self.Who_1 += self.eta_ho_1*(dWho_1 + self.momentum * dWho_1)

            tmpD = N.transpose((N.kron(N.ones((self.nh2,1)),
                                       delta_o_2[t])).reshape(self.nh2,self.no))
            tmpAct = N.transpose((N.repeat(self.act_h_2[t],self.no,
                                           axis=0)).reshape(self.nh2,self.no))
            dWho_2 += tmpAct*tmpD
            #dWho_2 = tmpAct*tmpD
            #self.Who_2 += self.eta_ho_2*(dWho_2 + self.momentum * dWho_2)

            
            
            # input to hidden weights
            
            
            tmpD = N.transpose((N.kron(N.ones((self.ni,1)),
                                       delta_h_1[t])).reshape(self.ni,self.nh1))
            
            tmpAct = N.transpose((N.repeat(signalIn[t],self.nh1,
                                           axis=0)).reshape(self.ni,self.nh1))
            dWih_1 += tmpAct*tmpD
            
            
            
            #dWih_1 = tmpAct*tmpD
            #self.Wih_1 += self.eta_ih_1*(dWih_1 + self.momentum * dWih_1)

            tmpD = N.transpose((N.kron(N.ones((self.ni,1)),
                                       delta_h_2[t])).reshape(self.ni,self.nh2))
            
            tmpAct = N.transpose((N.repeat(signalIn[t],self.nh2,
                                           axis=0)).reshape(self.ni,self.nh2))
            dWih_2 += tmpAct*tmpD
            #dWih_2 = tmpAct * tmpD
            #self.Wih_2 += self.eta_ih_2*(dWih_2 + self.momentum * dWih_2)
            
            # context to hidden weights
            tmpD = N.transpose((N.kron(N.ones((self.nh1,1)),
                                       delta_h_1[t])).reshape(self.nh1,self.nh1))
            tmpAct = N.transpose((N.repeat(self.act_c_1[t],self.nh1,
                                           axis=0)).reshape(self.nh1,self.nh1))
            dWch_1 += tmpAct*tmpD
            #dWch_1 = tmpAct*tmpD
            #self.Wch_1 += self.eta_ch_1*(dWch_1 + self.momentum * dWch_1)

            tmpD = N.transpose((N.kron(N.ones((self.nh2,1)),
                                       delta_h_2[t])).reshape(self.nh2,self.nh2))
            tmpAct = N.transpose((N.repeat(self.act_c_2[t],self.nh2,
                                           axis=0)).reshape(self.nh2,self.nh2))
            dWch_2 += tmpAct*tmpD
            #dWch_2 = tmpAct*tmpD
            #self.Wch_2 += self.eta_ch_2*(dWch_2  + self.momentum * dWch_2)
            
            if self.recurrent:
                # hidden to context weights
                tmpD = N.transpose((N.kron(N.ones((self.nh,1)),
                                    delta_c[t])).reshape(self.nh,self.nh))
                tmpAct = N.transpose((N.repeat(self.act_h[t],self.nh,
                                      axis=0)).reshape(self.nh,self.nh))
                dWhc += tmpAct*tmpD
            # parametric bias to hidden
            
            tmpD = N.transpose((N.kron(N.ones((self.np1,1)),
                                       delta_h_1[t])).reshape(self.np1,self.nh1))
            tmpAct = N.transpose((N.repeat(self.act_p_1,self.nh1,
                                           axis=0)).reshape(self.np1,self.nh1))
            dWph_1 += tmpAct*tmpD
            #dWph_1 = tmpAct*tmpD
            #self.Wph_1 += self.eta_ph_1*(dWph_1  + self.momentum * dWph_1)

            tmpD = N.transpose((N.kron(N.ones((self.np2,1)),
                                       delta_h_2[t])).reshape(self.np2,self.nh2))
            tmpAct = N.transpose((N.repeat(self.act_p_2,self.nh2,
                                           axis=0)).reshape(self.np2,self.nh2))
            dWph_2 += tmpAct*tmpD
            #dWph_2 = tmpAct*tmpD
            #self.Wph_2 += self.eta_ph_2*(dWph_2  + self.momentum * dWph_2)
            
            #tmpD = N.transpose((N.kron(N.ones((self.nhh,1)),
            #                           delta_h_1[t])).reshape(self.nhh,self.nh1))
            
            #tmpAct = N.transpose((N.repeat(self.act_hh_1[t],self.nh1,
            #                               axis=0)).reshape(self.nhh,self.nh1))
            #dWhh_1 += tmpAct*tmpD
       
        # adaptive learning rate.
        '''
          
        temp_mul_ho_1 = self.dWho_1_old * dWho_1
        self.eta_ho_1[(temp_mul_ho_1 < 0).nonzero()] = N.maximum((self.eta_ho_1[(temp_mul_ho_1 < 0).nonzero()]*self.negative_fac_eta),self.eta_min)
        self.eta_ho_1[(temp_mul_ho_1 > 0).nonzero()] = N.minimum((self.eta_ho_1[(temp_mul_ho_1 > 0).nonzero()]*self.positive_fac_eta),self.eta_max)
            
        #self.dWho_1[(temp_mul_ho_1 < 0).nonzero()] = -self.dWho_1[(temp_mul_ho_1 < 0).nonzero()]
        temp_mul_ho_2 = self.dWho_2_old * dWho_2
        self.eta_ho_2[(temp_mul_ho_2 < 0).nonzero()] = N.maximum((self.eta_ho_2[(temp_mul_ho_2 < 0).nonzero()]*self.negative_fac_eta),self.eta_min)
        self.eta_ho_2[(temp_mul_ho_2 > 0).nonzero()] = N.minimum((self.eta_ho_2[(temp_mul_ho_2 > 0).nonzero()]*self.positive_fac_eta),self.eta_max)
        #if temp_mul_ho_2 < 0:
        #    self.eta_ho_2 = max((self.eta_ho_2*self.negative_fac_eta),self.eta_min)
        #elif temp_mul_ho_2 > 0:
        #    self.eta_ho_2 = min((self.eta_ho_2*self.positive_fac_eta),self.eta_max)
        #self.dWho_2[(temp_mul_ho_2 < 0).nonzero()] = -self.dWho_2[(temp_mul_ho_2 < 0).nonzero()]   
            #self.dWho[(temp_mul_ho >= 0).nonzero()] = self.eta_ho*dWho[(temp_mul_ho >= 0).nonzero()] + self.momentum * self.dWho[(temp_mul_ho >= 0).nonzero()]
        #temp_mul_hh_1 = self.dWhh_1_old * dWhh_1
        #self.eta_hh_1[(temp_mul_hh_1 < 0).nonzero()] = N.maximum((self.eta_hh_1[(temp_mul_hh_1 < 0).nonzero()]*self.negative_fac_eta),self.eta_min)
        #self.eta_hh_1[(temp_mul_hh_1 > 0).nonzero()] = N.minimum((self.eta_hh_1[(temp_mul_hh_1 > 0).nonzero()]*self.positive_fac_eta),self.eta_max)    
                
        temp_mul_ih_1 = self.dWih_1_old * dWih_1
        self.eta_ih_1[(temp_mul_ih_1 < 0).nonzero()] = N.maximum((self.eta_ih_1[(temp_mul_ih_1 < 0).nonzero()]*self.negative_fac_eta),self.eta_min)
        self.eta_ih_1[(temp_mul_ih_1 > 0).nonzero()] = N.minimum((self.eta_ih_1[(temp_mul_ih_1 > 0).nonzero()]*self.positive_fac_eta),self.eta_max)
        #if temp_mul_ih_1 < 0:
        #    self.eta_ih_1 = max((self.eta_ih_1*self.negative_fac_eta),self.eta_min)
        #elif temp_mul_ih_1 > 0:
        #    self.eta_ih_1 = min((self.eta_ho_1*self.positive_fac_eta),self.eta_max)
        #self.dWih_1[(temp_mul_ih_1 < 0).nonzero()] = -self.dWih_1[(temp_mul_ih_1 < 0).nonzero()]
        temp_mul_ih_2 = self.dWih_2_old * dWih_2
        self.eta_ih_2[(temp_mul_ih_2 < 0).nonzero()] = N.maximum((self.eta_ih_2[(temp_mul_ih_2 < 0).nonzero()]*self.negative_fac_eta),self.eta_min)
        self.eta_ih_2[(temp_mul_ih_2 > 0).nonzero()] = N.minimum((self.eta_ih_2[(temp_mul_ih_2 > 0).nonzero()]*self.positive_fac_eta),self.eta_max)

        #if temp_mul_ih_2 < 0:
        #    self.eta_ih_2 = max((self.eta_ih_2*self.negative_fac_eta),self.eta_min)
        #elif temp_mul_ih_2 > 0:
        #    self.eta_ih_2 = min((self.eta_ih_2*self.positive_fac_eta),self.eta_max)
        #self.dWih_2[(temp_mul_ih_2 < 0).nonzero()] = -self.dWih_2[(temp_mul_ih_2 < 0).nonzero()] 
            #self.dWih[(temp_mul_ih >= 0).nonzero()] = self.eta_ih*dWih[(temp_mul_ih >= 0).nonzero()] + self.momentum * self.dWih[(temp_mul_ih >= 0).nonzero()]
            
            
        temp_mul_ch_1 = self.dWch_1_old * dWch_1
        self.eta_ch_1[(temp_mul_ch_1 < 0).nonzero()] = N.maximum((self.eta_ch_1[(temp_mul_ch_1 < 0).nonzero()]*self.negative_fac_eta),self.eta_min)
        self.eta_ch_1[(temp_mul_ch_1 > 0).nonzero()] = N.minimum((self.eta_ch_1[(temp_mul_ch_1 > 0).nonzero()]*self.positive_fac_eta),self.eta_max)

        #if temp_mul_ch_1 < 0:
        #    self.eta_ch_1 = max((self.eta_ch_1*self.negative_fac_eta),self.eta_min)
        #elif temp_mul_ch_1 > 0:
        #    self.eta_ch_1 = min((self.eta_ch_1*self.positive_fac_eta),self.eta_max)
        #self.dWch_1[(temp_mul_ch_1 < 0).nonzero()] = -self.dWch_1[(temp_mul_ch_1 < 0).nonzero()]
        temp_mul_ch_2 = self.dWch_2_old * dWch_2
        self.eta_ch_2[(temp_mul_ch_2 < 0).nonzero()] = N.maximum((self.eta_ch_2[(temp_mul_ch_2 < 0).nonzero()]*self.negative_fac_eta),self.eta_min)
        self.eta_ch_2[(temp_mul_ch_2 > 0).nonzero()] = N.minimum((self.eta_ch_2[(temp_mul_ch_2 > 0).nonzero()]*self.positive_fac_eta),self.eta_max)


        #if temp_mul_ch_2 < 0:
        #    self.eta_ch_2 = max((self.eta_ch_2*self.negative_fac_eta),self.eta_min)
        #elif temp_mul_ch_2 > 0:
        #    self.eta_ch_2 = min((self.eta_ch_2*self.positive_fac_eta),self.eta_max)
        #self.dWch_2[(temp_mul_ch_2 < 0).nonzero()] = -self.dWch_2[(temp_mul_ch_2 < 0).nonzero()]
            #self.dWch[(temp_mul_ch >= 0).nonzero()] = self.eta_ch*dWch[(temp_mul_ch >= 0).nonzero()] + self.momentum * self.dWch[(temp_mul_ch >= 0).nonzero()]            
            
        temp_mul_ph_1 = self.dWph_1_old * dWph_1
        self.eta_ph_1[(temp_mul_ph_1 < 0).nonzero()] = N.maximum((self.eta_ph_1[(temp_mul_ph_1 < 0).nonzero()]*self.negative_fac_eta),self.eta_min)
        self.eta_ph_1[(temp_mul_ph_1 > 0).nonzero()] = N.minimum((self.eta_ph_1[(temp_mul_ph_1 > 0).nonzero()]*self.positive_fac_eta),self.eta_max)
        #if temp_mul_ph_1 < 0:
        #    self.eta_ph_1 = max((self.eta_ph_1*self.negative_fac_eta),self.eta_min)
        #elif temp_mul_ph_1 > 0:
        #    self.eta_ph_1 = min((self.eta_ph_1*self.positive_fac_eta),self.eta_max)
        #self.dWph_1[(temp_mul_ph_1 < 0).nonzero()] = -self.dWph_1[(temp_mul_ph_1 < 0).nonzero()]
        temp_mul_ph_2 = self.dWph_2_old * dWph_2
        self.eta_ph_2[(temp_mul_ph_2 < 0).nonzero()] = N.maximum((self.eta_ph_2[(temp_mul_ph_2 < 0).nonzero()]*self.negative_fac_eta),self.eta_min)
        self.eta_ph_2[(temp_mul_ph_2 > 0).nonzero()] = N.minimum((self.eta_ph_2[(temp_mul_ph_2 > 0).nonzero()]*self.positive_fac_eta),self.eta_max)
        #if temp_mul_ph_2 < 0:
        #    self.eta_ph_2 = max((self.eta_ph_2*self.negative_fac_eta),self.eta_min)
        #elif temp_mul_ph_2 > 0:
        #    self.eta_ph_2 = min((self.eta_ph_2*self.positive_fac_eta),self.eta_max)
        #self.dWph_2[(temp_mul_ph_2 < 0).nonzero()] = -self.dWph_2[(temp_mul_ph_2 < 0).nonzero()]
            #self.dWph[(temp_mul_ph >= 0).nonzero()] = self.eta_ph*dWph[(temp_mul_ph >= 0).nonzero()] + self.momentum * self.dWph[(temp_mul_ph >= 0).nonzero()]
            
            
        if self.recurrent:
                
                temp_mul_hc = self.dWhc_old * dWhc
                self.dWhc[(temp_mul_hc < 0).nonzero()] = -self.dWhc[(temp_mul_hc < 0).nonzero()]               
                #self.dWhc[(temp_mul_hc >= 0).nonzero()] = self.eta_hc*dWhc[(temp_mul_hc >= 0).nonzero()]  + self.momentum * self.dWhc[(temp_mul_hc >= 0).nonzero()]
        
        '''
        # end going back through time
        # compute weight change over time series
        
        self.dWho_1 = self.eta_ho_1*(dWho_1 + self.momentum * self.dWho_1)
        self.dWho_2 = self.eta_ho_2*(dWho_2 + self.momentum * self.dWho_2)
        self.dWih_1 = self.eta_ih_1*(dWih_1 + self.momentum * self.dWih_1)
        #self.dWhh_1 = self.eta_hh_1*(dWhh_1 + self.momentum * self.dWhh_1)
        self.dWih_2 = self.eta_ih_2*(dWih_2 + self.momentum * self.dWih_2)
        self.dWch_1 = self.eta_ch_1*(dWch_1 + self.momentum * self.dWch_1)
        self.dWch_2 = self.eta_ch_2*(dWch_2 + self.momentum * self.dWch_2)
        self.dWph_1 = self.eta_ph_1*(dWph_1 + self.momentum * self.dWph_1)
        self.dWph_2 = self.eta_ph_2*(dWph_2 + self.momentum * self.dWph_2)
        if self.recurrent:
            self.dWhc = self.eta_hc*dWhc #+ self.momentum * self.dWhc
        
        

        
        aver_delta_p_1 = N.average(delta_p_1, axis=0)
        aver_delta_p_2 = N.average(delta_p_2, axis=0)
        self.gamma_1 =  self.gamma_factor * N.absolute(aver_delta_p_1)
        self.gamma_2 =  self.gamma_factor * N.absolute(aver_delta_p_2)
        
        
        # compute activity PB
        self.rho_1 += self.gamma_1*N.sum(delta_p_1,axis=0)
        self.act_p_1 = self.sigmoid(self.rho_1,'tanhOpt')

        self.rho_2 += self.gamma_2*N.sum(delta_p_2,axis=0)
        self.act_p_2 = self.sigmoid(self.rho_2,'tanhOpt')
        
        # save for next run
        
        self.dWho_1_old = N.copy(dWho_1)
        self.dWho_2_old = N.copy(dWho_2)
        self.dWih_1_old = N.copy(dWih_1)
        self.dWhh_1_old = N.copy(dWhh_1)
        self.dWih_2_old = N.copy(dWih_2)
        self.dWch_1_old = N.copy(dWch_1)
        self.dWch_2_old = N.copy(dWch_2)
        self.dWph_1_old = N.copy(dWph_1)
        self.dWph_2_old = N.copy(dWph_2)
        
        
        if self.recurrent:
            self.dWhc_old = N.copy(dWhc)
        
        
    def updateWeights(self, range=5.0):
        
        
        self.Who_1 += self.dWho_1
        self.Who_2 += self.dWho_2
        self.Wih_1 += self.dWih_1
        #self.Whh_1 += self.dWhh_1
        self.Wih_2 += self.dWih_2
        self.Wch_1 += self.dWch_1
        #self.Wch_2 += self.dWch_2
        self.Wph_1 += self.dWph_1
        #self.Wph_2 += self.dWph_2
        if self.recurrent:
            self.Whc += self.dWhc
            
        #print self.Wih_1
        
        for i in xrange(self.nh1):
            #self.Wph_1[i] /= N.linalg.norm(self.Wph_1[i])
            self.Wih_1[i] /= N.linalg.norm(self.Wih_1[i])
            self.Wch_1[i] /= N.linalg.norm(self.Wch_1[i])
        for i in xrange(self.nh2):
            #self.Wph_2[i] /= N.linalg.norm(self.Wph_2[i])
            self.Wih_2[i] /= N.linalg.norm(self.Wih_2[i])
            self.Wch_2[i] /= N.linalg.norm(self.Wch_2[i])
        
        outOfRangeDetected = False    

        '''
        range = 8.0
        if N.any(N.nonzero(self.Who < -range)) or \
                N.any(N.nonzero(self.Who > range)):
            self.Who = N.clip(self.Who,-range,range)
            self.logger.info("Who out of range")
        if N.any(N.nonzero(self.Wih < -range)) or \
                N.any(N.nonzero(self.Wih > range)):
            self.logger.info("Wih out of range")
            self.Wih = N.clip(self.Wih,-range,range)
        if N.any(N.nonzero(self.Wch < -range)) or \
                N.any(N.nonzero(self.Wch > range)):
            self.logger.info("Wch out of range")
            self.Wch = N.clip(self.Wch,-range,range)
        if N.any(N.nonzero(self.Wph < -range)) or \
                N.any(N.nonzero(self.Wph > range)):
            self.logger.info("Wph out of range")
            self.Wph = N.clip(self.Wph,-range,range)
        if self.recurrent:
            if N.any(N.nonzero(self.Whc < -range)) or \
                    N.any(N.nonzero(self.Whc > range)):
                self.logger.info("Whc out of range")
                self.Whc = N.clip(self.Whc,-range,range)
        '''
        outOfRangeDetected = False
        if N.any(N.nonzero(self.Who_1 < -range)) or \
                N.any(N.nonzero(self.Who_1 > range)):
            self.Who_1 -= self.dWho_1
            self.logger.error("Who_1 out of range")
            outOfRangeDetected = True
        if N.any(N.nonzero(self.Wih_1 < -range)) or \
                N.any(N.nonzero(self.Wih_1 > range)):
            self.Wih_1 -= self.dWih_1
            self.logger.error("Wih_1 out of range")
            outOfRangeDetected = True
        '''        
        if N.any(N.nonzero(self.Whh_1 < -range)) or \
                N.any(N.nonzero(self.Whh_1 > range)):
            self.Whh_1 -= self.dWhh_1
            self.logger.error("Whh_1 out of range")
            outOfRangeDetected = True
        '''
        if N.any(N.nonzero(self.Wch_1 < -range)) or \
                N.any(N.nonzero(self.Wch_1 > range)):
            self.Wch_1 -= self.dWch_1
            self.logger.error("Wch_1 out of range")
            outOfRangeDetected = True
        '''
        
        if N.any(N.nonzero(self.Wph < -range)) or \
                N.any(N.nonzero(self.Wph > range)):
            self.Wph -= self.dWph
            self.logger.error("Wph out of range")
            outOfRangeDetected = True
        
            '''
        if self.recurrent:
            if N.any(N.nonzero(self.Whc_1 < -range)) or \
                    N.any(N.nonzero(self.Whc_1 > range)):
                self.Whc_1 -= self.dWhc_1
                self.logger.error("Whc_1 out of range")
                outOfRangeDetected = True

        if N.any(N.nonzero(self.Who_2 < -range)) or \
                N.any(N.nonzero(self.Who_2 > range)):
            self.Who_2 -= self.dWho_2
            self.logger.error("Who_2 out of range")
            outOfRangeDetected = True
        if N.any(N.nonzero(self.Wih_2 < -range)) or \
                N.any(N.nonzero(self.Wih_2 > range)):
            self.Wih_2 -= self.dWih_2
            self.logger.error("Wih_2 out of range")
            outOfRangeDetected = True
        if N.any(N.nonzero(self.Wch_2 < -range)) or \
                N.any(N.nonzero(self.Wch_2 > range)):
            self.Wch_2 -= self.dWch_2
            self.logger.error("Wch_2 out of range")
            outOfRangeDetected = True
        
        if N.any(N.nonzero(self.Wph_1 < -range)) or \
                N.any(N.nonzero(self.Wph_1 > range)):
            self.Wph_1 -= self.dWph_1
            self.logger.error("Wph_1 out of range")
            outOfRangeDetected = True

        if N.any(N.nonzero(self.Wph_2 < -range)) or \
                N.any(N.nonzero(self.Wph_2 > range)):
            self.Wph_2 -= self.dWph_2
            self.logger.error("Wph_2 out of range")
            outOfRangeDetected = True
            
        if self.recurrent:
            if N.any(N.nonzero(self.Whc_2 < -range)) or \
                    N.any(N.nonzero(self.Whc_2 > range)):
                self.Whc_2 -= self.dWhc_2
                self.logger.error("Whc_2 out of range")
                outOfRangeDetected = True
        
        return outOfRangeDetected
        
    def setLearningRate(self, lr):
        self.eta = lr
        
    def train(self, data, transFunc, rho_1, rho_2, color):#, lr=0.001):#, test):
        self.rho_1 = rho_1
        self.rho_2 = rho_2
        self.act_p_1 = self.sigmoid(self.rho_1,'tanhOpt')
        self.act_p_2 = self.sigmoid(self.rho_2,'tanhOpt')
        self.logger.info('PB: '+str(self.act_p_1)+str(self.act_p_2))
        self.logger.info('rho: '+str(self.rho_1)+str(self.rho_2))
        signalIn, signalOut = data.getSignals()
        self.resetActivities(len(signalIn))
        
        error = N.zeros(1000000)
        converged = False
        itr = 0
        fail = False
        PB_THRESHOLD_PERCENT = 0.1
        
        #self.eta_hh_1 = N.ones((self.nh1,self.nhh)) * self.eta
        self.eta_ih_1 = N.ones((self.nh1,self.ni)) * self.eta
        self.eta_ho_1 = N.ones((self.no,self.nh1)) * self.eta
        self.eta_ch_1 = N.ones((self.nh1,self.nh1)) * self.eta
        self.eta_ph_1 = N.ones((self.nh1,self.np1)) * self.eta
        self.eta_ih_2 = N.ones((self.nh2,self.ni)) * self.eta
        self.eta_ho_2 = N.ones((self.no,self.nh2)) * self.eta
        self.eta_ch_2 = N.ones((self.nh2,self.nh2)) * self.eta
        self.eta_ph_2 = N.ones((self.nh2,self.np2)) * self.eta
        while(not converged) or itr>100000:
            
            # training happens here
            e = self.forwardTS(signalIn, signalOut, 1.0, color, transFunc)
            error[itr] = N.mean(e)
            self.backwardTS(signalIn, signalOut, transFunc)
            fail = self.updateWeights()
            act_p_1_old = self.act_p_1
            act_p_2_old = self.act_p_2
            if fail == True:
                break
            if itr%500 == 0:
                #print self.act_h_2
                self.logger.info("Iteration: "+str(itr)+\
                                     " Error: "+str(e)+" PB1: "+str(self.act_p_1)+" PB2: "+str(self.act_p_2))
                print "average error: ", N.mean(e)
                print N.mean(e) <= self.convergence
                if self.plot:
                    self.plotWeights()
                
                self.update_visualisation()
            itr += 1
            
            
            
            if N.mean(e) <= self.convergence: #\
            #and N.average(N.absolute(act_p_1_old-self.act_p_1)) < self.act_p_1*PB_THRESHOLD_PERCENT\
            #and N.average(N.absolute(act_p_2_old-self.act_p_2)) < self.act_p_2*PB_THRESHOLD_PERCENT:
                converged = True
            
#        P.figure()
#        P.plot(error,'b-')
#        P.savefig('error.png')
#        P.close()
        return self.act_p_1, self.act_p_2, self.rho_1, self.rho_2, itr, N.copy(error[0:itr]), fail
    def evaluate2(self, data, pb, transFunc, fname='generate.png'):
        signalIn, signalOut = data.getSignals()
        self.act_p = pb
        for i in range(100):
            e = self.forwardTS(signalIn, signalOut, 0.0, transFunc)
        self.logger.info( "Evaluation error: " + str(e) + " PB: " + str(self.act_p))
        P.figure()
        P.plot(self.act_o,'g:')
        P.plot(signalOut,'ro')
        P.savefig(fname)
        P.close()
    def evaluate(self, data, pb, transFunc, fname='generate.png'):
        signalIn, signalOut = data.getSignals()
        self.resetActivities(len(signalIn))
        self.act_p = pb
        for i in range(100):
            e = self.forwardTS(signalIn, signalOut, 1.0, transFunc)
        self.logger.info("Evaluation error: "+str(e))#+" PB: "+str(self.act_p))
        fig = P.figure()
        ax = fig.add_subplot(111)
        colors = ['b','g','r','c','m','y','k']
        for k in range(signalIn.shape[1]):
            l = ax.plot(self.act_o[:,k],str(colors[k]+':'))
            ax.plot(signalOut[:,k],str(colors[k]+'o'))
        ax.set_ylim([-1.0,1.0])
        P.savefig(fname)
        P.close()
        return e
        
    
    
    def recognition(self, data, rho, transFunc,lr,convThres=50, PB_diff_threshold=0.0001):
        signalIn, signalOut = data.getSignals()
        self.resetActivities(len(signalIn))
        self.rho = rho #N.zeros(self.np)
        self.act_p = self.sigmoid(rho, transFunc) #N.zeros(self.np)
        self.etaRho = lr
        itr = 0
        totalItr = 50000
        
        pbvalues = N.zeros((self.np,totalItr))
        convergenceCount = 0
        while (itr < totalItr):
            e = self.forwardTS(signalIn, signalOut, 1.0, transFunc)
            self.backwardTSrec(signalIn, signalOut, transFunc)
            for l in range(self.np):
                pbvalues[l][itr] = self.act_p[l]
            if itr%500 == 0:
                self.logger.info("Iteration: "+str(itr)+" PB: "+str(self.act_p))
                self.logger.info("Evaluation error: "+str(e))#+ " PB: "+str(self.act_p))
            
            if itr != 0:
                tmp = 0.
                for l in range(self.np):
                    tmp += N.abs(pbvalues[l][itr]-pbvalues[l][itr-1])
                if tmp < PB_diff_threshold:
#                if (N.abs(pbvalues[0][itr]-pbvalues[0][itr-1]) + \
#                        N.abs(pbvalues[1][itr]-pbvalues[1][itr-1])) < 0.00001:
                    convergenceCount += 1
                else:
                    convergenceCount = 0
                    
            
            
            
            itr += 1
            if convergenceCount == convThres:
                self.logger.info("Recognition Done!")
                self.logger.info("Iteration: "+str(itr)+" PB: "+str(self.act_p))
                
                self.logger.info("Evaluation error: "+str(e))#+ " PB: "+str(self.act_p))
                break
        return N.copy(pbvalues[:,0:itr]), self.rho, itr

    def update_visualisation(self):
        
        #print self.act_h_2
        KT.exporttiles(self.Who_1, self.no, self.nh1, basedir+"obs_v_2_1.pgm")
        KT.exporttiles(self.Wih_1, self.nh1, self.ni, basedir+"obs_v_1_0.pgm")
        #KT.exporttiles(self.Whh_1, self.nh1, self.nhh, basedir+"obs_v_3_2.pgm")
        KT.exporttiles(self.Wch_1, self.nh1, self.nh1, basedir+"obs_v_1_1.pgm")
        KT.exporttiles(self.Wph_1, self.nh1, self.np1, basedir+"obs_v_3_1.pgm")
        KT.exporttiles(self.Whc_1, self.nh1, self.nh1, basedir+"obs_v_2_2.pgm")
        KT.exporttiles(self.Who_2, self.no, self.nh2, basedir+"obs_w_2_1.pgm")
        KT.exporttiles(self.Wih_2, self.nh2, self.ni, basedir+"obs_w_1_0.pgm")
        KT.exporttiles(self.Wch_2, self.nh2, self.nh2, basedir+"obs_w_1_1.pgm")
        KT.exporttiles(self.Wph_2, self.nh2, self.np2, basedir+"obs_w_3_1.pgm")
        KT.exporttiles(self.Whc_2, self.nh2, self.nh2, basedir+"obs_w_2_2.pgm")

        KT.exporttiles(self.act_i, self.ni, self.timeSteps, basedir+"obs_S_0.pgm")
        KT.exporttiles(self.act_h_1, self.nh1, self.timeSteps, basedir+"obs_A_1.pgm")
        KT.exporttiles(self.act_h_2, self.nh2, self.timeSteps, basedir+"obs_B_1.pgm")
        KT.exporttiles(self.act_o, self.no, self.timeSteps, basedir+"obs_S_2.pgm")
        KT.exporttiles(self.act_p_1, self.np1, 1, basedir+"obs_P_1.pgm")
        KT.exporttiles(self.act_p_2, self.np2, 1, basedir+"obs_P_2.pgm")
        
       
basedir = "/tmp/coco/"         

if __name__ =='__main__':
    dataNames = [['sin2', 'cos2'],['sin2', 'sinCos']]    
    #dataNames = ['sin2','circle','sin2_2','circle_2']
    numOfSequences = len(dataNames)
    dim = 4
    data = []
    count = 0
    signalLength = 0
    Ncolor = 2
        

    for func in dataNames:
        
        for color in xrange(Ncolor):
        
            dataGen = D.DataGenerator(5, 1, logger, dim, func, Ncolor)
        
            dataGen.makeMulDsignals_color(2, color)
            
        #dataGen.funcLoader(str(func))
            data.append(dataGen)
            signalLength += dataGen.getSignalLength()
            dataGen.plotMulDSignals(str(func)+'.png')
            count += 1
        
    ''' Parameters '''
    numOfInput = Ncolor * 2
    numOfOutput = numOfInput
    numOfPB = 1
    numOfHidden = 10
    numOfHiddenHidden = 20
    numOfContext = numOfHidden
    lr = 0.001 #0.0001
    momentum = 0.0
    transferFunction = 'linear'
    convergenceThreshold = 0.006 #0.003
    epochThreshold = 4 # if sum of steps within one epoch is < break
    maxEpochs = 10000
    logger.info('Epoch threshold: '+str(epochThreshold))
    logger.info('Convergence threshold: '+str(convergenceThreshold))
    logger.info('Maximal number of epochs: '+str(maxEpochs))
    
                
    ''' Prepare Network '''    
    NET = RNNPB(numOfInput,
                numOfHidden,
                numOfHiddenHidden,
                numOfContext,
                numOfOutput,
                numOfPB,
                lr,momentum,
                logger,signalLength,
                convergenceThreshold,
                plot=False,recurrent=False)
        
    rho_1 = N.zeros((numOfSequences*Ncolor,numOfPB))
    rho_2 = N.zeros((numOfSequences*Ncolor,numOfPB))
    for k in range(rho_1.shape[0]):
        rho_1[k] = N.zeros(numOfPB)
        rho_2[k] = N.zeros(numOfPB)
    totalIter = 0
    numOfEpochsNeeded = 0
    error = []
    stepsPerEpoch = []
    for k in range(numOfSequences*Ncolor):
        error.insert(k,list())
        error[k] = N.zeros(0)
        stepsPerEpoch.insert(k,list()) 
    fail = False
    
    '''begin training'''
    for l in range(maxEpochs):
        numOfStepsTillConvergence = 0
        
        for j in range(len(data)):
                print j
                if j == 0 or j == 2:
                    color = 0
                else:
                    color = 1
                logger.info("Epoch: "+str(l)+" "+ str(dataNames[j/2]) + "_" + str(j))
                pb_1, pb_2,rho_1[j],rho_2[j],tmpItr,e, fail = NET.train(data[j], 
                                                 transferFunction, 
                                                 rho_1[j],rho_2[j],color)
                logger.info("Steps needed: "+str(tmpItr)+' PB: '+str(pb_1)+str(pb_2))
                
                totalIter += tmpItr
                numOfStepsTillConvergence += tmpItr
                error[j] = N.concatenate((error[j],e))
                stepsPerEpoch[j].append(tmpItr)
                
                if l % 10 == 0:
                    fname = str(data[j].function) + '_' + str(j) + '_' + str(l) 
                    fig = P.figure()
                    ax = fig.add_subplot(111)
                    colors = ['b','g','r','c','m','y','k']
                    for k in range(data[j].signalIn.shape[1]):
                        lm = ax.plot(NET.act_o[:,k],str(colors[k]+':'))
                        ax.plot(data[j].signalOut[:,k],str(colors[k]+'o'))
                    ax.set_ylim([-1.0,1.0])
                    P.savefig(fname)
                    P.close()    
        if fail:
                break
        if numOfStepsTillConvergence <= epochThreshold:
                break
        numOfEpochsNeeded += 1    
    logger.info('done training')
    logger.info('NO saving, NO display of results')
    
    
    
    logger.info('Epoch threshold: '+str(epochThreshold))
    logger.info('Convergence threshold: '+str(convergenceThreshold))
    logger.info('Maximal number of epochs: '+str(maxEpochs))
    
    ''' Trajectory recognition '''
    
    signalRecognition = False
    signalPrediction = True
    
    first_observe_samples = 8
    number_observe_samples = 4
    
    
    
    if signalRecognition == True and signalPrediction == True:
        # TODO: try online recognition and prediction
        
        RecdataNames = ['new_sin2','new_square']
        
        PB_update_step = 3
        Prediction_step = 3
        
        Recog_threshold = 0.00001
        Part_recog_threshold = 0.001
        
        numOfSequences = len(RecdataNames)
        dim = 2
        data = []
        
        signalLength = 0
        totalIter = 0
        
        #stepsPerEpoch = []
        
        #for k in range(numOfSequences):
        #   stepsPerEpoch.insert(k,list()) 
        
        ''' Parameters '''
    
        lr = 0.01
        transferFunction = 'tanhOpt'
        for func in RecdataNames:

            
            
            RecdataGen = D.DataGenerator(12, 1, logger, dim, func)
            RecdataGen.funcLoader(str(func))
            #RecdataGen.makeMulDsignals(3)
            #RecdataGen.noisyChannel()
            data.append(RecdataGen)
            signalLength += RecdataGen.getSignalLength()
            lr = 0.0001
            transferFunction = 'tanhOpt'
            RecdataGen.plotMulDSignals(func[0]+'.png')
            
            
        rho = N.zeros((numOfSequences,numOfPB))
        
        
        
        for k in range(numOfSequences):
            rho[k] = N.zeros(numOfPB)
        
        for j in range(numOfSequences):
            finished = False
            rec_fun = RecdataNames[j]
            start_I = 0
            end_I = start_I + first_observe_samples
            
            #first few recognition and prediction, which may be larger than ordinary one...
            PB_save, rho[j], Tempitr = NET.recognition_part(start_I, end_I, data[j], rho[j], transferFunction,lr,convThres=50,PB_diff_threshold=0.0001)
            NET.evaluate_part(end_I, start_I, data[j], NET.act_p, transferFunction, fname='generate'+str(rec_fun)+'.png')
            
            
            while not finished:
                rec_fun = RecdataNames[j]
                start_I +=  number_observe_samples
                end_I += number_observe_samples
                
                PB_save, rho[j], Tempitr = NET.recognition_part(start_I, end_I, data[j], rho[j], transferFunction,lr,convThres=50,Recog_threshold=0.0001)
                NET.evaluate_part(end_I, start_I, data[j], NET.act_p, transferFunction, fname='generate'+str(rec_fun)+'.png')
                
                totalIter += Tempitr
                
                if end_I >= data[j].timeSteps:
                    finished = True
               # stepsPerEpoch[j].append(tmpItr)
            print 'Final PB values:', NET.act_p
        
            
                
                
    
        
    
