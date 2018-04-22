# -*- coding: utf-8 -*-
"""
Created on Fri Dec 01 10:29:18 2017

@author: Alastair Wiseman
"""
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines


#set up Latex labeling
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True

def RKStabilityFunction (z, RKMatrix, RKWeights):
    #z, a complex number
    #RKMatrix and RKweights, a Runge-Kutta matrix and its correspoding
    #weights

    return (1 + z * np.dot(RKWeights, np.dot(np.linalg.inv(
    np.identity(len(RKWeights)) - z  * RKMatrix), np.ones
    ((len(RKWeights),1)))))


def secantMethod(function, z_0, z_1, tol = 1e-10):
    #function, some complex valued function of a single variable
    #z_0, z_1, two complex numbers close to the desired zero of the
    #function
    #tol, how close we require the function value at a potential root
    #to be to 0 before we stop
    
    #set intial values
    p0 = z_0
    p1 = z_1
    f0 = function(p0)
    f1 = function(p1)
    
    #secant method 
    while np.abs(f1) > tol:
        p0, p1 = p1, p1 - (f1 * (p1 - p0)) / (f1 - f0)
        f0, f1 = f1, function(p1)
        
    return p1
  

def RKOrderStar (RKMatrix, RKWeights, RStart = -5, REnd = 5, ImStart = -5,
                 ImEnd = 5, meshDivisions = 100, legend = True):
    #RKMatrix and RKweights, a Runge-Kutta matrix and its correspoding
    #weights
    #RStart, REnd the bounds on the real components of points plotted
    #ImStart, ImEnd the bounds on the real components of points plotted
    #meshDivisions, the number of subdivisions of the real/imaginary axes
    #legend, set false if key not wanted
    

    #setup grid for function evaluations
    A = np.linspace(RStart, REnd, meshDivisions)
    B = 1j * np.linspace(ImStart, ImEnd, meshDivisions)
    p, q = np.meshgrid(A,B)
    C = p + q


    #evaluate the Runge-Kutta stability function on the grid
    for i in xrange(meshDivisions):
        for j in xrange(meshDivisions):
            C[i][j] = abs(np.exp(-1 * C[i][j]) * (RKStabilityFunction(C[i][j],
            RKMatrix, RKWeights)))   
    
    #Initialize a Figure
    fig = plt.figure()
    
    #Add Axes to Figure
    ax = fig.add_subplot(111)
    
    #plot the order star and its boundary
    ax.contour(p,q*1j,C, [1], colors = 'C0')
    ax.contourf(p,q*1j,C, [1, np.inf], alpha = 0.1, colors = 'C0')
    
    #find approximate locations of the zeros and poles
    CS1 = plt.contour(p, q * 1j, C, [0.01, 0.1])
    CS2 = plt.contour(p,q *1j, C, [10, 100])
    
    #get the closed contours containing zeros
    ZContours = CS1.allsegs
    ZContoursClosed = []
    for j in xrange(len(ZContours)):
        for i in xrange(len(ZContours[j])):
            if len(ZContours[j][i]) > 1 and (ZContours[j][i][0][0] - 
                  ZContours[j][i][-1][0] + ZContours[j][i][0][1] - 
                  ZContours[j][i][-1][1] < 1e-10):
                ZContoursClosed.append(ZContours[j][i])
            
    #get the closed contours conatining poles
    PContours = CS2.allsegs
    PContoursClosed = []
    for j in xrange(len(PContours)):
        for i in xrange(len(PContours[j])):
            if len(PContours[j][i]) > 1 and (PContours[j][i][0][0] - 
                  PContours[j][i][-1][0] + PContours[j][i][0][1] - 
                  PContours[j][i][-1][1] < 1e-10):
                PContoursClosed.append(PContours[j][i])
    
    #clean up contours containing zeros and poles from plot
    for coll in CS1.collections:
        coll.remove()
    
    for coll in CS2.collections:
        coll.remove()
    
    #redefine the stability function so it takes one argument, used to
    #find zeros
    def f(x):
        return RKStabilityFunction(x, RKMatrix, RKWeights)
    
    #define the multiplicative inverse of the stability function, used
    #to find poles
    def g(x):
        return 1.0 / RKStabilityFunction(x, RKMatrix, RKWeights)
        
    #apply secant method to find zeros of stability function
    for i in xrange(len(ZContoursClosed)):
        a = secantMethod(f, ZContoursClosed[i][0][0] + 
        ZContoursClosed[i][0][1] * 1j, ZContoursClosed[i][1][0]
        + ZContoursClosed[i][1][1] * 1j)
        plt.plot(np.real(a), np.imag(a), 'o', color = 'C1')
    
    #apply secant method to find zeros of the multiplicative inverse of
    #the stability function, i.e. find poles of the stability function
    for i in xrange(len(PContoursClosed)):
        a = secantMethod(g, PContoursClosed[i][0][0] + 
        PContoursClosed[i][0][1] * 1j, PContoursClosed[i][1][0] + 
        PContoursClosed[i][1][1] * 1j)
        plt.plot(np.real(a), np.imag(a), 's', color = 'C1')    
    
    #setup legend
    OS = mpatches.Rectangle((0, 0), 1, 1, fc="C0",alpha=0.1)
    OSB = mlines.Line2D([], [], color='C0')
    DOS = mpatches.Rectangle((0, 0), 1, 1, ec="C0",alpha=1, fill = False)
    zeros = mlines.Line2D([], [], color='C1', marker = 'o', ls = 'none')
    poles = mlines.Line2D([], [], color='C1', marker = 's', ls = 'none')
    handles = [OS, OSB, DOS, zeros, poles]
    labels = ['$\mathcal{A}_+$', '$\mathcal{A}_0$','$\mathcal{A}_-$', 'Zeros', 'Poles']
    if legend == True:
        ax.legend(handles, labels)
        
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
    plt.axes().set_aspect('equal')
    plt.show()
    return