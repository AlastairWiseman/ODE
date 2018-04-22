# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 15:48:14 2017

@author: Alastair Wiseman
"""
import numpy as np
import RungeKuttaCoefficients as RKC
import RungeKuttaMethod as RKM

def LMMStepExplicit (function, initialt, initialY, stepSize, YCoefficients, FCoefficients, stepNum):
    #function takes arguments as [t,Y], t a scalar, Y a vector(list), 
    #it should output a vector(list) of the same dimension as Y
    #initialt should be a list of t_n values
    #intialY should be a list of Y_n values
    #t_i a scalar , Y_i a vector(list)
    #stepSize should be a float
    #YCoefficients, the "alpha" coefficents, a_0, ... ,a_k
    #FCoefficients, the "beta" coefficients, b_0, ... b_k
    #dim, the dimension of the Y vector
    
    #calculate sum of f_i 's
    BSum = 0
    for j in xrange(stepNum):
        BSum += FCoefficients[j] * function(initialt[j], initialY[j])
    
    #calculate sum of y_i 's
    ASum = 0
    for k in xrange(stepNum):
        ASum += YCoefficients[k] * initialY[k]
    
    
    #rearange for y_(n+1)
    ynew = ((stepSize * BSum) - ASum) / YCoefficients[-1]
    
    tnew = initialt[-1] + stepSize
    
    return tnew, ynew
    

def LMMStepImplicit (function, initialt, initialY, stepSize, YCoefficients, FCoefficients ,dim, stepNum):
    #function takes arguments as [t,Y], t a scalar, Y a vector(list), 
    #it should output a vector(list) of the same dimension as Y
    #initialt should be a list of t_n values
    #intialY should be a list of Y_n values
    #t_i a scalar , Y_i a vector(list)
    #stepSize should be a float
    #YCoefficients, the "alpha" coefficents, a_0, ... ,a_k
    #FCoefficients, the "beta" coefficients, b_0, ... b_k
    #dim, the dimension of the Y vector
    
    #calculate sum of f_i 's
    BSum = 0
    for j in xrange(stepNum):
        BSum += FCoefficients[j] * function(initialt[j], initialY[j])
    
    #calculate sum of y_i 's
    ASum = 0
    for k in xrange(stepNum):
        ASum += YCoefficients[k] * initialY[k]
    
    #rearange for y_(n+1)
    beta = ((stepSize * BSum) - ASum) / YCoefficients[-1]
 
    #setup blank Jacobian matrix
    J = np.ones([dim, dim])

    #fill in Jacobian matrix using a central difference 
    #formula
    for a in xrange(dim):
        for b in xrange(dim):
            H = np.zeros(dim)
            H[b] = 0.00000001
            W = initialY[-1] + (H * 0.5)
            X = initialY[-1] - (H * 0.5)
            J[a][b] = ((function(initialt[-1], W)[a] - 
             function(initialt[-1], X)[a]) / 0.00000001)
    #print J
    #setup for modified newton method
    w = initialY[-1]
    #print w
    Idim = np.identity(dim)
    
    A = Idim - np.multiply(stepSize, np.multiply(FCoefficients[-1] / YCoefficients[-1], J))
        
    A = np.linalg.inv(A)

    B = w - np.multiply(stepSize, np.multiply(FCoefficients[-1] / YCoefficients[-1], function(initialt[-1] + stepSize, w))) - beta
    
    C = np.dot(A, B)
    
    
    #carry out modified Newton-Raphson to convergence
    while np.linalg.norm(C) > 1e-10:
        w = w - C
        
        B = w - np.multiply(stepSize, np.multiply(FCoefficients[-1] / YCoefficients[-1], function(initialt[-1] + stepSize, w))) - beta
        
        C = np.dot(A, B)
        
    #Evaluate y_(n+1)
    ynew = w
    
    tnew = initialt[-1] + stepSize
    
    return tnew, ynew

def LinearMultistepMethod (function, initialConditions, stepSize, target, 
                           YCoefficients, FCoefficients, LMMOrder, 
                           detailed = False, stiff = True):
    #function takes arguments as [y,t], function ,f, of the form 
    #"dy/dt = f(y,t)"
    #initialConditions should be given as [y_0,t_0] s.t. "y(t_0) = y_0" 
    #stepSize should be a float
    #target should be a float
    #YCoefficients, the "alpha" coefficents, a_0, ... ,a_k
    #FCoefficients, the "beta" coefficients, b_0, ... b_k
    #detailed, if true, will cause the function to output each of the 
    #values from each step
    #stiff, if stiff is true we initialise using IRK2GL instead of ERK4
    
    #Outputs an approximated function value at each target(s)
    
    
    #check target is valid
    if target < initialConditions[0]:
        print "One of more targets is less than t_0"
        return
    
    #Check if method is implicit
    implicit = False
    if FCoefficients[-1] != 0.0:
        implicit = True
             
    
    #get initial conditions for first steps
    initialTarget = initialConditions[0] + stepSize * (len(YCoefficients) - 2)
    
    detailedRK = True
    
    if stiff == True:
        #initialise using IRK2GL
        a = RKM.RungeKuttaMethod(function, initialConditions, stepSize, 
                             initialTarget, RKC.IRK2GL[0], RKC.IRK2GL[1], 
                             RKC.IRK2GL[2], RKC.IRK2GL[3], detailedRK)
    else:
        #initialise using IRK2GL
        a = RKM.RungeKuttaMethod(function, initialConditions, stepSize, 
                             initialTarget, RKC.ERK4[0], RKC.ERK4[1], 
                             RKC.ERK4[2], RKC.ERK4[3], detailedRK)
    t = a[0]
    y = a[1]
    
    #store the dimension of the vectors in the system
    dim = len(initialConditions[1])
    
    #store number of steps
    stepNum = len(YCoefficients) - 1
    
    #for LMM    
    while t[-1] < target:
        
        initialt = t[-stepNum : ]
        initialY = y[-stepNum : ]
                    
        if implicit == True:
            
            tnew, ynew = LMMStepImplicit(function, initialt, initialY, stepSize, YCoefficients, FCoefficients, dim, stepNum)
            
        else:
            
            tnew, ynew = LMMStepExplicit(function, initialt, initialY, stepSize, YCoefficients, FCoefficients, stepNum)
            
        #Add new y and t values to lists
        y = np.append(y, np.array([ynew]), axis = 0)
        t = np.append(t, tnew)
            
    output = [t, np.array(y)]
    
    #linearly interpolate in order to get values at target
    if target != t[-1]:
        output[0][-1], output[1][-1] = target, y[-2] + np.multiply(target - t[-2], (y[-1] - y[-2]) / stepSize)
                        
    if detailed == True:
        return output
    
    else:
        return [output[0][-1], output[1][-1]]


    
    
    
    
    