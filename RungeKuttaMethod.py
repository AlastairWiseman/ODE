# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 11:02:30 2017

@author: Alastair Wiseman
"""
import numpy as np


def RKStepExplicit (function, initialt, initialY, stepSize,RKMatrix, RKWeights,
                    RKNodes, dim):
    #function takes arguments as [t,Y], t a scalar, Y a vector(list), 
    #it should output a vector(list) of the same dimension as Y
    #initialt should be t_n
    #intialY should be Y_n
    #t a scalar , Y a vector(list)
    #stepSize should be a float
    #*RKMethod should be a description of an RK method in the form
    #[RKMatrix, RKWeights, RKNodes, RKOrder]
    #dim, the dimension of the Y vector
            
    #set epsilon_1
    Y = np.array([initialY])
    
    #Set remaining epsilon_i
    for j in xrange(len(RKNodes) - 1):
        #Calculate "sum portion" of epsilon_(j+1)
        matrixSum = np.zeros([1, dim])
        for l in xrange(j + 1):
            matrixSum += np.multiply(stepSize * RKMatrix[j + 1][l],
            function(initialt  + stepSize * RKNodes[l], Y[l]))
        Y = np.append(Y, initialY + matrixSum, axis = 0)
                
            
    #Evaluate y_(n+1)
    ynew = np.array([initialY])
    
    for j in xrange(len(RKWeights)):
        ynew = ynew + (np.multiply(stepSize, np.multiply(RKWeights[j], 
        function(initialt + stepSize * RKNodes[j], Y[j]))))
    
    
    tnew = initialt + stepSize
    
    return tnew, ynew
    

def RKStepImplicit (function, initialt, initialY, stepSize, RKMatrix, 
                    RKWeights, RKNodes,dim):
    #function takes arguments as [t,Y], t a scalar, Y a vector(list), 
    #it should output a vector(list) of the same dimension as Y
    #initialt should be t_n
    #intialY should be Y_n
    #t a scalar , Y a vector(list)
    #stepSize should be a float
    #*RKMethod should be a description of an RK method in the form
    #[RKMatrix, RKWeights, RKNodes, RKOrder]
    #dim, the dimension of the Y vector
            
    #setup blank Jacobian matrix
    J = np.ones([dim, dim])
    
    #fill in Jacobian matrix using a central difference 
    #formula
    for a in xrange(dim):
        for b in xrange(dim):
            H = np.zeros(dim)
            H[b] = 0.00000001
            W = initialY + (H * 0.5)
            X = initialY - (H * 0.5)
            J[a][b] = ((function(initialt, W)[a] - 
             function(initialt, X)[a]) / 0.00000001)
    
    #set up for modified Newton-Raphson
    Istage = np.identity(len(RKNodes))
    Idim = np.identity(dim)
    
    eps = np.tile(initialY, len(RKNodes))
    eps = np.reshape(eps, (len(eps), 1))
    
    beta = np.tile(initialY, len(RKNodes))
    beta = np.reshape(beta , (len(beta), 1))
    
    #create the function vector F^[i-1]:
    def Feps (eps_i):
        
        feps = np.zeros([len(eps_i), 1])
        
        for a in xrange(len(RKNodes)):
            F = function(initialt + stepSize * RKNodes[a], 
                         eps[a * dim : (a + 1) * dim : 1])
            for b in xrange(dim):
                feps[a * dim + b] = F[b]
        return feps
    
    A = np.kron(Istage, Idim) - np.multiply(stepSize, np.kron(RKMatrix, J))
    
    A = np.linalg.inv(A)
    
    B = (eps - np.multiply(stepSize, np.matmul(np.kron(RKMatrix, Idim), 
                                               Feps(eps))) - beta)
    
    C = np.matmul(A, B)
    
    #carry out modified Newton-Raphson to convergence
    while np.linalg.norm(C) > 1e-10:
        eps = eps - C
        
        B = (eps - np.multiply(stepSize, np.matmul(np.kron(RKMatrix, Idim), 
                                                   Feps(eps))) - beta)
        
        C = np.matmul(A, B)
    
    Y = np.reshape(eps, (len(RKNodes), dim))
    
    #Evaluate y_(n+1)
    ynew = np.array([initialY])
    
    for j in xrange(len(RKWeights)):
        ynew = ynew + (np.multiply(stepSize, np.multiply(RKWeights[j], 
        function(initialt + stepSize * RKNodes[j], Y[j]))))
    
    
    tnew = initialt + stepSize
    
    return tnew, ynew


def RungeKuttaMethod (function, initialConditions, stepSize, target, RKMatrix,
                      RKWeights, RKNodes, RKOrder, detailed = False, 
                      varyStep = False, tol = 1e-5):
    #function takes arguments as [t,Y], t a scalar, Y a vector(list), 
    #it should output a vector(list) of the same dimension as Y
    #initialConditions should be given as [t_0,Y_0] s.t. "y(t_0) = y_0"
    #t a scalar , Y a vector(list)
    #stepSize should be a float
    #target should be a float
    #*RKMethod should be a description of an RK method in the form
    #[RKMatrix, RKWeights, RKNodes, RKOrder]
    #detailed, if true, will cause the function to output each of the 
    #values from each step
    #varyStep, if true we vary stepsize via method of Richardson 
    #extrapolation
    #tol, a user specified tolerance on allowable local error
    
    #Outputs an approximated function value at each target(s)
    
    #Check target is valid
    if target < initialConditions[0]:
        print "One of more targets is less than t_0"
        return
    
    #Check if method is implicit
    implicit = False
    for i in xrange(len(RKMatrix[0])):
        for j in xrange(len(RKMatrix[0]) - i):
            if RKMatrix[i][-(j + 1)] != 0.0:
                implicit = True
                break
        if implicit == True:
            break
    
    #get initial conditions
    t_0 = initialConditions[0]
    y_0 = np.array(initialConditions[1])

    #set up lists for y and t values at current step
    t = np.array([t_0])
    y = np.array([y_0])
    
    #store the dimension of the vectors in the system
    dim = len(initialConditions[1])
    
    #take first step
    if varyStep == True:
        
        errorEst = True
        
        #we take a first try at a step, then we loop whilst reducing
        #stepSize so errorEst is less than tol, then we loop whilst
        #increasing stepSize if errorEst is much smaller than tol
        for i in xrange(3):
            
            if i == 0:
                def test(x):
                    return x == True
            elif i == 1:
                def test(x):
                    return x > tol
            else:
                def test(x):
                    return x < tol / (2.0 ** (RKOrder + 1.0))
                
            stepSizeVariation = [1.0, 0.5, 2.0]
            
            while test(errorEst):
               
                stepSize = stepSize * stepSizeVariation[i]
                
                if implicit == True:
                    tnew1, ynew1 = RKStepImplicit(function, t[-1], y[-1],
                                                  stepSize, RKMatrix, 
                                                  RKWeights, RKNodes, dim)
                    tnew2, ynew2 = RKStepImplicit(function, tnew1, ynew1[0],
                                                  stepSize, RKMatrix,
                                                  RKWeights, RKNodes, dim)
                    
                    tnew3, ynew3 = RKStepImplicit(function, t[-1], y[-1],
                                                  2.0 * stepSize, RKMatrix,
                                                  RKWeights, RKNodes, dim)
                
                    
                else:
                    tnew1, ynew1 = RKStepExplicit(function, t[-1], y[-1],
                                                  stepSize, RKMatrix,
                                                  RKWeights, RKNodes, dim)
                    tnew2, ynew2 = RKStepExplicit(function, tnew1, ynew1[0],
                                                  stepSize, RKMatrix,
                                                  RKWeights, RKNodes, dim)
                    
                    tnew3, ynew3 = RKStepExplicit(function, t[-1], y[-1],
                                                  2.0 * stepSize, RKMatrix,
                                                  RKWeights, RKNodes, dim)
                    
                LTEE = (ynew2 - ynew3) / (2.0 ** (RKOrder + 1.0) - 1.0)
                
                errorEst = max(map(abs, LTEE[0]))
        
        if errorEst > tol:
            stepSize = stepSize / 2.0
        
    else:
        
        if implicit == True:
            tnew1, ynew1 = RKStepImplicit(function, t[-1], y[-1], stepSize,
                                          RKMatrix, RKWeights, RKNodes, dim)
            
        else:
            tnew1, ynew1 = RKStepExplicit(function, t[-1], y[-1], stepSize,
                                          RKMatrix, RKWeights, RKNodes, dim)
        
        
    #Add new y and t values to lists
    y = np.append(y, ynew1, axis = 0)
    t = np.append(t, tnew1)
            
        
    #take remaining steps
    while t[-1] < target:
        
        if varyStep == True:
            
            errorEst = True
        
            #we take a first try at a step, then we loop whilst reducing
            #stepSize so errorEst is less than tol, then we loop whilst
            #increasing stepSize if errorEst is much smaller than tol
            for i in xrange(3):
                
                if i == 0:
                    def test(x):
                        return x == True
                elif i == 1:
                    def test(x):
                        return x > tol
                else:
                    def test(x):
                        return x < tol / (2.0 ** (RKOrder + 1.0))
                    
                stepSizeVariation = [1.0, 0.5, 2.0]
                
                while test(errorEst):
               
                    stepSize = stepSize * stepSizeVariation[i]
        
                    if implicit == True:
                        tnew1, ynew1 = RKStepImplicit(function, t[-1], y[-1],
                                                      stepSize, RKMatrix,
                                                      RKWeights, RKNodes, dim)
                        
                        bigStepFactor = (t[-1] - t[-2]) / stepSize + 1
                        tnew2, ynew2 = RKStepImplicit(function, t[-2], y[-2],
                                                      bigStepFactor * stepSize,
                                                      RKMatrix, RKWeights,
                                                      RKNodes, dim)
                    
                        
                    else:
                        tnew1, ynew1 = RKStepExplicit(function, t[-1], y[-1],
                                                      stepSize, RKMatrix,
                                                      RKWeights, RKNodes, dim)
                        
                        bigStepFactor = ((t[-1] - t[-2]) / stepSize) + 1.0
                        tnew2, ynew2 = RKStepExplicit(function, t[-2], y[-2],
                                                      bigStepFactor * stepSize,
                                                      RKMatrix,RKWeights,
                                                      RKNodes, dim)
                        
                    LTEE = ((ynew1 - ynew2) / (bigStepFactor **
                            (RKOrder + 1.0) - 1.0))
                    
                    errorEst = max(map(abs, LTEE[0]))
            
            if errorEst > tol:
                stepSize = stepSize / 2.0
            
                
        else:
            if implicit == True:
                tnew1, ynew1 = RKStepImplicit(function, t[-1], y[-1], stepSize,
                                              RKMatrix, RKWeights, RKNodes,
                                              dim)
                
            else:
                tnew1, ynew1 = RKStepExplicit(function, t[-1], y[-1], stepSize,
                                              RKMatrix, RKWeights, RKNodes,
                                              dim)
            
            
        #Add new y and t values to lists
        y = np.append(y, ynew1, axis = 0)
        t = np.append(t, tnew1)
            
    output = [t, np.array(y)]
    
    #Redo last step with specific step size to hit the target
    if target != t[-1]:
        stepSizeLast = target - t[-2]
        
        if implicit == True:
            t[-1], y[-1] = RKStepImplicit(function, t[-2], y[-2], stepSizeLast, 
                                          RKMatrix, RKWeights, RKNodes, dim)
                    
        else:
            t[-1], y[-1] = RKStepExplicit(function, t[-2], y[-2], stepSizeLast, 
                                          RKMatrix, RKWeights, RKNodes, dim)
        
        output = [t, np.array(y)]
                
    if detailed == True:
        return output
    
    else:
        return [output[0][-1], output[1][-1]]