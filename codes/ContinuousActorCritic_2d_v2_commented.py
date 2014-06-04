# CACLA reinforcement learning simulation
# in a 2D grid world with size of 10x10
# Author: J Zhong
# Date: 09-24-2012
# zhong@informatik.uni-hamburg.de

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

# Assumptions: 
# 1. Reward is in (4,4) (To change it, please refer line 131)
# 2. The only one output unit is a continuous value representing moving angle of the agent; the moving radius is set to be 1.


# First CACLA implementation as described at:
# http://homepages.cwi.nl/~hasselt/rl_algs/Cacla.html
# To run, first create a directory for the results:
#    mkdir /tmp/coco/
# Then run:
#    python ActorCritic.py
# The weights are written into "/tmp/coco/" as ".pnm" image files.
# To watch these image files conveniently, use lookpy.tcl as:
#    lookpy.tcl a w 0 1
# (Weights must exist first, so better use a 2nd xterm for this)
# (Parameters mean to see activation and weight files of areas 0 and 1)


import random
import numpy
import scipy
import pylab
import Image
import math


# used by exporttiles()
# insert into file a comment which looks e.g. like this:  # highS: 0.099849  lowS: -0.099849
def exportinfo (filename, highS, lowS):
    f = open(filename, 'rb')
    content = f.read()
    f.close()
    f = open(filename, 'wb')
    charcount = 0
    for char in content:
        f.write(char)
        if charcount == 2:
           f.write('# highS: %.6f  lowS: %.6f\n' % (highS, lowS))
        charcount += 1
    f.close()


def exporttiles (X, x, y, a, b, frame, filename):
    xy, ab = numpy.shape(X)    
    if  (xy != x*y) or (ab != a*b):
        print 'exporttiles: size error'

    Y = numpy.zeros((frame + x*(a+frame), frame + y*(b+frame)))

    image_id = 0
    for xx in range(x):
        for yy in range(y):
            if image_id >= xy: 
                break
            tile = numpy.reshape (X[image_id], (a, b))
            beginA, beginB = frame + xx*(a+frame), frame + yy*(b+frame)
            Y[beginA : beginA+a, beginB : beginB+b] = tile
            image_id += 1

    im = Image.new ("L", (frame + y*(b+frame), frame + x*(a+frame)))
    im.info = 'comment here does not work'
    im.putdata (Y.reshape((frame + x*(a+frame)) * (frame + y*(b+frame))), offset=-Y.min()*255.0/(Y.max()-Y.min()), scale=255.0/(Y.max()-Y.min()) )
    im.save(filename, cmap=pylab.cm.jet)  # seems to ignore the colormap
    exportinfo (filename,  numpy.max(X), numpy.min(X))

class world_model_RL:
    def __init__(self, size_a, size_b):
        # init input position

        self.sel_a = random.uniform (0, size_a)
        self.sel_b = random.uniform (0, size_b)
        
        self.size_a = size_a
        self.size_b = size_b
        self.states = self.update_activation()
        
    def newinit(self):
        
        self.sel_a = random.uniform (0, self.size_a)
        self.sel_b = random.uniform (0, self.size_b)
        self.states = self.update_activation()

    def update_activation(self):
        states = numpy.zeros((self.size_a*self.size_b))
        
        var = 1.5
        
        for a in range(0, self.size_a):
            for b in range(0, self.size_b):
                distance =(a+0.5-self.sel_a)**2+(b+0.5-self.sel_b)**2
                states[a * self.size_b + b] = math.exp(-distance/(2*var**2))
        states /= numpy.sum(states)

        return states
        
    def act(self, act): #act is CONTINUOUS from 0 .. 2*PI
        # position world reaction
        self.sel_a += math.sin(act)
        self.sel_b += math.cos(act)

        # position boundary conditions
        if  self.sel_a < 0.0:
            self.sel_a = 0.0
        elif self.sel_a > self.size_a - 1.0:
            self.sel_a = self.size_a - 1.0
        if  self.sel_b < 0.0:
            self.sel_b = 0.0
        elif self.sel_b > self.size_b - 1.0:
            self.sel_b = self.size_b - 1.0
        self.states = self.update_activation()
        
    def reward(self): #TODO how to define reward????
        if  self.sel_a>=4.5 and self.sel_a <= 5.5 and self.sel_b >= 4.5 and self.sel_b<=5.5:
            return 1.0
        else:
            return 0.0
    def sensor(self):
        
        return numpy.reshape(self.states, (size_map))

    def rand_winner (self, h, sigma):
        rand = random.normalvariate(h, sigma)
        if  rand < 0.0:
            rand += 2.0 * math.pi
        elif rand >= 2.0 * math.pi:
            rand -= 2.0 * math.pi

        return rand
        
    def process_boundary(self, w_mot, I):
        
        
        sum_a = numpy.dot(numpy.sin(w_mot),I)
        sum_b = numpy.dot(numpy.cos(w_mot),I)
        angle = math.atan2(sum_a, sum_b)
        
        if angle < 0 :
            return angle + 2*math.pi
        else:
            return angle


size_a, size_b = 10, 10
size_map = (size_a) * (size_b)
size_mot = 1

w_mot = numpy.random.uniform(0, 2.0*math.pi, (size_mot, size_map))
w_cri = numpy.random.uniform(0.0, 0.1, (size_map))

world = world_model_RL(size_a, size_b)

sigma = 2*0.314 
eps = 0.1
gamma = 0.7
eta = 0.7


for iter in range (10000):
        
        world.newinit()
        I = world.sensor()
        h2 = world.process_boundary(w_mot,I)
        act = world.rand_winner (h2, sigma)
        val = numpy.dot (w_cri, I)                          # value
        r = 0
        duration = 0
    
        
        while r == 0 and duration < 1000:
            
            duration += 1
            world.act(act)                                  # do selected action
            r = world.reward()                              # read reward
            I_tic = world.sensor()                          # read new state
            
            h2 = world.process_boundary(w_mot,I_tic)
            act_tic = world.rand_winner (h2, sigma)                 # choose next action
            
            val_tic = numpy.dot(w_cri, I_tic)
            if  r == 1.0:                                   # This is cleaner than defining
                target = r                                  # target as r + gamma * val_tic,
                print 'reward achieved!'
                print 'duration: ',duration
            else:                                           # because critic weights now converge.
                target = gamma * (val_tic)                   
            delta = target  - val                            # prediction error; gamma             w_cri += eps * delta * (I)
            w_cri += eps * delta * I
            w_cri = numpy.clip(w_cri, 0.0, numpy.inf)


            if val_tic > val:
                sum_a = (math.sin(act)*eta*I)-numpy.sin(w_mot)*eta*I
                sum_b = (math.cos(act)*eta*I)-numpy.cos(w_mot)*eta*I

                
                sum_a = numpy.reshape(sum_a,(1,size_a*size_b))
                sum_b = numpy.reshape(sum_b,(1,size_a*size_b))
                                      

                w_mot_a = numpy.sin(w_mot) + sum_a
                w_mot_b = numpy.cos(w_mot) + sum_b
                w_mot = numpy.arctan2(w_mot_a, w_mot_b)

                for i in range(numpy.shape(w_mot)[1]):
        
                    if w_mot[0,i] < 0 :
                       w_mot[0,i] += 2.0*math.pi
            # personally prefer this update rules.. more straightforward and original..
                            
            

            I[0:size_map] = I_tic[0:size_map]
            val = val_tic
            act = act_tic

        
        exporttiles (numpy.reshape(I,(1,size_a * size_b)), 1, 1, size_a, size_b, 1, "/tmp/coco/obs_I_0.pgm")

        exporttiles (w_mot, 1, size_mot, size_a, size_b, 1, "/tmp/coco/obs_v_0_0.pgm")
        
        exporttiles (numpy.reshape (w_cri, (1,size_a * size_b)), 1, 1, size_a, size_b, 1, "/tmp/coco/obs_w_1_1.pgm")

        print iter, duration, ' w_mot=%.2f..%.2f' % (numpy.min(w_mot), numpy.max(w_mot)), ' w_cri=%.2f..%.2f' % (numpy.min(w_cri), numpy.max(w_cri))
