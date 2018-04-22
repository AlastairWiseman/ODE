# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 10:43:34 2018

@author: Alastair Wiseman
"""

import numpy as np
import matplotlib.pyplot as pltt
import RungeKuttaMethod as RKM
import RungeKuttaCoefficients as RKC
import LinearStabilityDomains as LSD
import LinearMultistepMethodCoefficients as LMMC
import matplotlib.patches as mpatches
import LinearMultistepMethod as LMM

#Plot 1

def func1(t, Y):
    function1 = -2.0 * Y[0] + 1.0 * Y[1] + 2.0 * np.sin(t)
    function2 = 1.0 * Y[0] - 2.0 * Y[1] + 2.0 * (np.cos(t) - np.sin(t))
    return np.array([function1, function2])

def func2(t, Y):
    function1 = -2.0 * Y[0] + 1.0 * Y[1] + 2.0 * np.sin(t)
    function2 = 998.0 * Y[0] - 999.0 * Y[1] + 999.0 * (np.cos(t) - np.sin(t))
    return np.array([function1, function2])

def solution(t):
    solution1 = 2.0 * np.exp(-t) + np.sin(t)
    solution2 = 2.0 * np.exp(-t) + np.cos(t)
    return np.array([solution1, solution2])

t = np.linspace(0, 12, 1000)

#Initialize a Figure
fig = pltt.figure()

#Add Axes to Figure
ax = fig.add_subplot(111)

pltt.plot(t, solution(t)[0], color = 'C0', label = '$y^{[1]}$')
pltt.plot(t, solution(t)[1], color = 'C1', label = '$y^{[2]}$')

ax.legend(fontsize = 18)
pltt.xlim(0, 11)
pltt.minorticks_off()

# making the top and right spine invisible:
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')

# moving bottom spine up to y=0 position:
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data',0))
ax.grid(b = 'on')


points1 = RKM.RungeKuttaMethod(func1, [0.0, [2.0, 3.0]], 0.1, 11, *RKC.ERK4C, 
                               detailed = True, varyStep = True, tol = 0.01)
points2 = RKM.RungeKuttaMethod(func2, [0.0, [2.0, 3.0]], 0.1, 11, *RKC.ERK4C, 
                               detailed = True, varyStep = True, tol = 0.01)

stepSize1 = []
stepSize2 = []

for i in xrange(len(points1[0]) -1):
    stepSize1.append(points1[0][i + 1] - points1[0][i])
    
for i in xrange(len(points2[0]) -1):
    stepSize2.append(points2[0][i + 1] - points2[0][i])
    

#Initialize a Figure
fig = pltt.figure()

#Add Axes to Figure
ax = fig.add_subplot(111)

pltt.plot(t, solution(t)[0], color = 'C0', label = '$y^{[1]}$')
pltt.plot(t, solution(t)[1], color = 'C1', label = '$y^{[2]}$')

pltt.plot(points1[0], points1[1][: , 0], 'o', color = 'C0',
          label = '$y^{[1]}_n$')
pltt.plot(points1[0], points1[1][: , 1], 'o', color = 'C1',
          label = '$y^{[2]}_n$')

ax.legend(fontsize = 16)
pltt.xlim(0, 11)
pltt.minorticks_off()

# making the top and right spine invisible:
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')

# moving bottom spine up to y=0 position:
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data',0))
ax.grid(b = 'on')

#Initialize a Figure
fig = pltt.figure()

#Add Axes to Figure
ax = fig.add_subplot(111)

pltt.plot(t, solution(t)[0], color = 'C0', label = '$y^{[1]}$')
pltt.plot(t, solution(t)[1], color = 'C1', label = '$y^{[2]}$')

pltt.plot(points2[0], points2[1][: , 0], 'o', color = 'C0',
          label = '$y^{[1]}_n$')
pltt.plot(points2[0], points2[1][: , 1], 'o', color = 'C1',
          label = '$y^{[2]}_n$')

ax.legend(fontsize = 16)
pltt.xlim(0, 11)
pltt.minorticks_off()

# making the top and right spine invisible:
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')

# moving bottom spine up to y=0 position:
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data',0))
ax.grid(b = 'on')

points3 = RKM.RungeKuttaMethod(func1, [0.0, [2.0, 3.0]], 0.1, 11, *RKC.IRK2GL, 
                               detailed = True, varyStep = True, tol = 0.01)
points4 = RKM.RungeKuttaMethod(func2, [0.0, [2.0, 3.0]], 0.1, 11, *RKC.IRK2GL, 
                               detailed = True, varyStep = True, tol = 0.01)


stepSize3 = []
stepSize4 = []

for i in xrange(len(points3[0]) -1):
    stepSize3.append(points3[0][i + 1] - points3[0][i])
    
for i in xrange(len(points4[0]) -1):
    stepSize4.append(points4[0][i + 1] - points4[0][i])
    

#Initialize a Figure
fig = pltt.figure()

#Add Axes to Figure
ax = fig.add_subplot(111)

pltt.plot(t, solution(t)[0], color = 'C0', label = '$y^{[1]}$')
pltt.plot(t, solution(t)[1], color = 'C1', label = '$y^{[2]}$')

pltt.plot(points3[0], points3[1][: , 0], 'o', color = 'C0',
          label = '$y^{[1]}_n$')
pltt.plot(points3[0], points3[1][: , 1], 'o', color = 'C1',
          label = '$y^{[2]}_n$')

ax.legend(fontsize = 16)
pltt.xlim(0, 11)
pltt.minorticks_off()

# making the top and right spine invisible:
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')

# moving bottom spine up to y=0 position:
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data',0))
ax.grid(b = 'on')


#Initialize a Figure
fig = pltt.figure()

#Add Axes to Figure
ax = fig.add_subplot(111)

pltt.plot(t, solution(t)[0], color = 'C0', label = '$y^{[1]}$')
pltt.plot(t, solution(t)[1], color = 'C1', label = '$y^{[2]}$')

pltt.plot(points4[0], points4[1][: , 0], 'o', color = 'C0',
          label = '$y^{[1]}_n$')
pltt.plot(points4[0], points4[1][: , 1], 'o', color = 'C1',
          label = '$y^{[2]}_n$')

ax.legend(fontsize = 16)
pltt.xlim(0, 11)
pltt.minorticks_off()

# making the top and right spine invisible:
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')

# moving bottom spine up to y=0 position:
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data',0))
ax.grid(b = 'on')


#Plot 2

LSD.RKLSDPlotter(RKC.ERK1FE[0], RKC.ERK1FE[1], -6, 1, -4, 4, 1000, False)


LSD.RKLSDPlotter(RKC.ERK2MP[0], RKC.ERK2MP[1], -6, 1, -4, 4, 1000, False)


LSD.RKLSDPlotter(RKC.ERK3K[0], RKC.ERK3K[1], -6, 1, -4, 4, 1000, False)


LSD.RKLSDPlotter(RKC.ERK4C[0], RKC.ERK4C[1], -6, 1, -4, 4, 1000, False)


LSD.RKLSDPlotter(RKC.ERK5B[0], RKC.ERK5B[1], -6, 1, -4, 4, 1000, False)


LSD.RKLSDPlotter(RKC.ERK5KN[0], RKC.ERK5KN[1], -6, 1, -4, 4, 1000, False)


#Plot 3

LSD.LMMLSDPlotterComplete(LMMC.AB1[0], LMMC.AB1[1], -2, 1, -2, 2, 3000, 1000,
                          False)

LSD.LMMLSDPlotterComplete(LMMC.AB3[0], LMMC.AB3[1], -2, 1, -2, 2, 3000, 1000,
                          False)

LSD.LMMLSDPlotterComplete(LMMC.AB5[0], LMMC.AB5[1], -2, 1, -2, 2, 3000, 1000,
                          False)

LSD.LMMLSDPlotterComplete(LMMC.AM1B[0], LMMC.AM1B[1], -8, 1, -4, 4, 2000,
                          1000, False)
pltt.xlim(-7.9, 2.9)
pltt.ylim(-4.0, 4.0)

LSD.LMMLSDPlotterComplete(LMMC.AM2[0], LMMC.AM2[1], -6, 1, -4, 4, 1000, 1000,
                          False)


LSD.LMMLSDPlotterComplete(LMMC.AM4[0], LMMC.AM4[1], -6, 1, -4, 4, 1000, 1000,
                          False)


#Plot 4

LSD.RKLSDPlotter(RKC.IRK2GL[0], RKC.IRK2GL[1], -8, 1, -4, 4, 100, False)
pltt.xlim(-7.9, 2.9)
pltt.ylim(-4.0, 4.0)


#Plot 5

LSD.LMMLSDPlotterComplete(LMMC.BDF1[0], LMMC.BDF1[1], -1, 3, -1.5, 1.5, 1000,
                          1000, False)

LSD.LMMLSDPlotterComplete(LMMC.BDF2[0], LMMC.BDF2[1], -2, 6, -3, 3, 1000,
                          1000, False)

LSD.LMMLSDPlotterComplete(LMMC.BDF3[0], LMMC.BDF3[1], -3, 9, -4.5, 4.5, 1000,
                          1000, False)

LSD.LMMLSDPlotterComplete(LMMC.BDF4[0], LMMC.BDF4[1], -6, 18, -9, 9, 2000,
                          3000, False)

LSD.LMMLSDPlotterComplete(LMMC.BDF5[0], LMMC. BDF5[1], -10, 30, -15, 15, 2000,
                          3000, False)

LSD.LMMLSDPlotterComplete(LMMC.BDF6[0], LMMC. BDF6[1], -16, 48, -24, 24, 2000,
                          3000, False)



#Plot 6

def func1(t, Y):
    function1 = -2.0 * Y[0] + 1.0 * Y[1] + 2.0 * np.sin(t)
    function2 = 1.0 * Y[0] - 2.0 * Y[1] + 2.0 * (np.cos(t) - np.sin(t))
    return np.array([function1, function2])

def func2(t, Y):
    function1 = -2.0 * Y[0] + 1.0 * Y[1] + 2.0 * np.sin(t)
    function2 = 998.0 * Y[0] - 999.0 * Y[1] + 999.0 * (np.cos(t) - np.sin(t))
    return np.array([function1, function2])

def solution(t):
    solution1 = 2.0 * np.exp(-t) + np.sin(t)
    solution2 = 2.0 * np.exp(-t) + np.cos(t)
    return np.array([solution1, solution2])

t = np.linspace(0, 12, 1000)


points1 = LMM.LinearMultistepMethod(func1, [0.0, [2.0, 3.0]], 1.0, 11,
                                    *LMMC.BDF6, detailed = True)
points2 = LMM.LinearMultistepMethod(func2, [0.0, [2.0, 3.0]], 1.0, 11,
                                    *LMMC.BDF6, detailed = True)

#Initialize a Figure
fig = pltt.figure()

#Add Axes to Figure
ax = fig.add_subplot(111)

pltt.plot(t, solution(t)[0], color = 'C0', label = '$y^{[1]}$')
pltt.plot(t, solution(t)[1], color = 'C1', label = '$y^{[2]}$')

pltt.plot(points1[0], points1[1][: , 0], 'o', color = 'C0',
          label = '$y^{[1]}_n$')
pltt.plot(points1[0], points1[1][: , 1], 'o', color = 'C1',
          label = '$y^{[2]}_n$')

ax.legend(fontsize = 16)
pltt.xlim(0, 11)
pltt.minorticks_off()

# making the top and right spine invisible:
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')

# moving bottom spine up to y=0 position:
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data',0))
ax.grid(b = 'on')

#Initialize a Figure
fig = pltt.figure()

#Add Axes to Figure
ax = fig.add_subplot(111)

pltt.plot(t, solution(t)[0], color = 'C0', label = '$y^{[1]}$')
pltt.plot(t, solution(t)[1], color = 'C1', label = '$y^{[2]}$')

pltt.plot(points2[0], points2[1][: , 0], 'o', color = 'C0',
          label = '$y^{[1]}_n$')
pltt.plot(points2[0], points2[1][: , 1], 'o', color = 'C1',
          label = '$y^{[2]}_n$')

ax.legend(fontsize = 16)
pltt.xlim(0, 11)
pltt.minorticks_off()

# making the top and right spine invisible:
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')

# moving bottom spine up to y=0 position:
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data',0))
ax.grid(b = 'on')