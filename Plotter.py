# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 23:55:01 2017

@author: Alastair
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

#set up Latex labeling
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True

def plotF(function, xmin, xmax, samples, Label = False, color = 'C0'):
    #function should be a list of functions of the form f(x)
    #xmin should be the float value where you want to start the plot
    #end should be the float value where you want to end the plot
    #samples should be an int equal to the number of points you want 
    #plotting
    
    #make function a list    
    if (not isinstance(function, list)):
        function = [function]
    
    x = np.linspace(xmin, xmax, samples)
    
    
    #Initialize a Figure
    fig = plt.figure()
    
    #Add Axes to Figure
    ax = fig.add_subplot(111)
    
    if Label == False:
        for i in xrange(len(function)):
            fx = function[i](x)
            ax.plot(x, fx)
    else:
        for i in xrange(len(function)):
            fx = function[i](x)
            ax.plot(x, fx, label = Label, color = color)
    
    # making the top and right spine invisible:
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    # moving bottom spine up to y=0 position:
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data',0))
    # moving left spine to the right to position x == 0:
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data',0))
    ax.grid(b = 'on')
    
def plotPoints(x, y, yLog = False, xLog = False):
    #x should be a list of x values
    #y should be a list of corresponding y values
    
    if (yLog == True and xLog == True):
        plt.loglog(x, y, 'o-')
    elif (yLog == True and xLog == False):
        plt.semilogy(x, y, 'o-')
    elif (yLog == False and xLog == True):
        plt.semilogx(x, y, 'o-')
    else:
        plt.plot(x, y, 'o-')
    
    plt.show()