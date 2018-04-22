# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 13:59:10 2017

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

    return (np.abs(1 + z * np.dot(RKWeights, np.dot(np.linalg.inv(
    np.identity(len(RKWeights)) - z  * RKMatrix), np.ones
    ((len(RKWeights),1))))))

def RKLSDPlotter (RKMatrix, RKWeights, RStart = -5, REnd = 5, ImStart = -5,
                  ImEnd = 5, meshDivisions = 100, legend = True):
    #RKMatrix and RKweights, a Runge-Kutta matrix and its correspoding
    #weights
    #RStart, REnd the bounds on the real components of points plotted
    #ImStart, ImEnd the bounds on the real components of points plotted
    #meshDivisions, the number of subdivisions of the real/imaginary axes
    #legend, set False if you don't want a key

    #setup grid for function evaluations
    A = np.linspace(RStart, REnd, meshDivisions)
    B = 1j * np.linspace(ImStart, ImEnd, meshDivisions)
    p, q = np.meshgrid(A,B)
    C = p + q
    
    #evaluate the Runge-Kutta stability function on the grid
    for i in xrange(meshDivisions):
        for j in xrange(meshDivisions):
            C[i][j] = RKStabilityFunction(C[i][j], RKMatrix, RKWeights)
    
    #Initialize a Figure
    fig = plt.figure()
    
    #Add Axes to Figure
    ax = fig.add_subplot(111)
    
    #plot the boundary and region of linear stability
    ax.contour(p,q*1j,C, [1], colors = 'C0')
    ax.contourf(p,q*1j,C, [0,1], alpha = 0.1, colors = 'C0')
    
    #setup legend
    LSD = mpatches.Rectangle((0, 0), 1, 1, fc="C0",alpha=0.1)
    LSDB = mlines.Line2D([], [], color='C0')
    handles = [LSD, LSDB]
    labels = ['LSD', 'Boundary']
    if legend == True:
        ax.legend(handles, labels)
    
    #setup plot window
    ax = plt.gca()
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
    plt.axes().set_aspect('equal', 'datalim')
    plt.show()
    return

def LMMStabilityFunction (w, YCoefficients, FCoefficients):
    #w, a complex number
    #YCoefficients, the "alpha" coefficents, a_0, ... ,a_k
    #FCoefficients, the "beta" coefficients, b_0, ... b_k

    #setup numerator/denominator

    rho = 0.0
    sigma = 0.0

    #calculate numerator and denominator for boundary locus method

    for i in xrange(len(YCoefficients)):
        rho += (w ** i) * YCoefficients[i]
        sigma += (w ** i) * FCoefficients[i]

    return (rho / sigma)
    
def SchurCriterion (z, YCoefficients, FCoefficients):
    #z, a complex number
    #YCoefficients, the "alpha" coefficents, a_0, ... ,a_k
    #FCoefficients, the "beta" coefficients, b_0, ... b_k
    
    poly = YCoefficients - z * FCoefficients
    
    #reduce polynomial to the order 1 case
    while len(poly) > 2:
        #check coefficient condition
        if (np.abs(poly[-1]) > np.abs(poly[0])):
            #get conjugate reciprical polynomial
            polyConj = np.conjugate(poly)[:: -1]
            #setup blank polynomial with order one less
            polyTemp = 1j * np.zeros(len(poly) - 1)
            #evaluate the next polynomial in the sequence
            for i in xrange(len(polyTemp)):
                polyTemp[i] = polyConj[0] * poly[1 + i] - poly[0] * polyConj[1 + i]
            
            poly = polyTemp
        else:
            return False
    
    #check if roots of the order 1 polynomial are in the unit circle      
    if np.abs(- poly[0] / poly[1]) < 1:
        return True
    else:
        return False


def LMMLSDPlotter (YCoefficients, FCoefficients, steps = 100):
    #YCoefficients, the "alpha" coefficents, a_0, ... ,a_k
    #FCoefficients, the "beta" coefficients, b_0, ... b_k

    #setup values on the unit circle centered at 0 + 0i
    A = np.linspace(- np.pi, np.pi, steps)
    A = np.exp(A * 1j)
    
    #setup vector to hold function values
    B = 1j * np.zeros(steps)
    
    #evaluate the boundary locus (on the unit circle)
    for i in xrange(steps):
        B[i] = LMMStabilityFunction(A[i], YCoefficients, FCoefficients)

    #plot the boundary locus
    plt.plot(B.real,B.imag, '-', color = 'C0')
    return



def LMMLSDPlotterSchur (YCoefficients, FCoefficients, RStart = -10, REnd = 10,
                        ImStart = -10, ImEnd = 10, meshDivisions = 100):
    #YCoefficients, the "alpha" coefficents, a_0, ... ,a_k
    #FCoefficients, the "beta" coefficients, b_0, ... b_k
    #RStart, REnd the bounds on the real components of points plotted
    #ImStart, ImEnd the bounds on the real components of points plotted
    #meshDivisions, the number of subdivisions of the real/imaginary axes

    #setup grid points for function evaluations
    A = np.linspace(RStart, REnd, meshDivisions)
    B = 1j * np.linspace(ImStart, ImEnd, meshDivisions)
    p, q = np.meshgrid(A,B)
    C= p + q
    
    #evaluate Schur criterion on the previously setup grid
    for i in xrange(meshDivisions):
        for j in xrange(meshDivisions):
            C[i][j] = (int(SchurCriterion(C[i][j], YCoefficients,
                         FCoefficients)))

    
    #plot region for where the polynomial passes the Schur criterion
    plt.contourf(p, q * 1j, C, [0.9, 1.1], alpha = 0.1, colors = 'C0')
    return
    
def LMMLSDPlotterComplete (YCoefficients, FCoefficients, RStart = -10, 
                           REnd = 10,ImStart = -10, ImEnd = 10, 
                           meshDivisions = 100, boundaryDivisions = 100,
                           legend = True):
    #YCoefficients, the "alpha" coefficents, a_0, ... ,a_k
    #FCoefficients, the "beta" coefficients, b_0, ... b_k
    #RStart, REnd the bounds on the real components of points plotted
    #ImStart, ImEnd the bounds on the real components of points plotted
    #meshDivisions, the number of subdivisions of the real/imaginary axes
    #boundaryDivisions, the number of points where the boundary curve is 
    #evaluated at
    #legend, set False if you don't want a key
    
    #Initialize a Figure
    fig = plt.figure()
    
    #Add Axes to Figure
    ax = fig.add_subplot(111)
    
    #get boundary
    LMMLSDPlotter(YCoefficients, FCoefficients, boundaryDivisions)
    
    #get filled region
    LMMLSDPlotterSchur(YCoefficients, FCoefficients, RStart, REnd,ImStart,
                       ImEnd, meshDivisions)
    #setup legend
    LSD = mpatches.Rectangle((0, 0), 1, 1, fc="C0",alpha=0.1)
    LSDB = mlines.Line2D([], [], color='C0')
    handles = [LSD, LSDB]
    labels = ['LSD', 'Boundary']
    if legend == True:
        ax.legend(handles, labels)
    
    #setup plot window
    ax = plt.gca()
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
    plt.axes().set_aspect('equal', 'datalim')
    plt.show()