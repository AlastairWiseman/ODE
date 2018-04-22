# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 13:38:40 2018

@author: Alastair Wiseman
"""
import numpy as np
import matplotlib.pyplot as pltt
import RungeKuttaMethod as RKM
import RungeKuttaCoefficients as RKC
import LinearMultistepMethodCoefficients as LMMC
import matplotlib.lines as mlines
import LinearMultistepMethod as LMM

#Plot #1

def a(z):
    return np.sin(z)

def b(t,y):
    return np.cos(t)
    
points = RKM.RungeKuttaMethod(b, [0.0, [0.0]], 0.4, 4.0, *RKC.ERK1FE, detailed = True)

def t(z):
    return z * np.cos(1.2) + np.sin(1.2) - np.cos(1.2) * 1.2
def s(z):
    return z * np.cos(1.2) + points[1][3] - np.cos(1.2) * 1.2

T = [np.linspace(0, 2.5, 1000), t(np.linspace(0, 2.5, 1000))]
S = [np.linspace(1.2, 1.6, 1000), s(np.linspace(1.2, 1.6, 1000))]


x = np.linspace(0, 5, 1000)

#Initialize a Figure
fig = pltt.figure()

#Add Axes to Figure
ax = fig.add_subplot(111)

fx = a(x)
ax.plot(x, fx)

pltt.plot(T[0], T[1], ls = '--', color = 'C2')
pltt.plot(S[0], S[1], color = 'C2')
pltt.plot([1.2, 1.2], np.linspace(0, points[0][3], 2), ls = ':', 
          color = 'C2')
pltt.plot([1.6, 1.6], np.linspace(0, points[0][4], 2), ls = ':', 
          color = 'C2')
pltt.plot(4, a(4), 'o', color = 'r')
pltt.ylim(-1.1, 1.3)
ax = pltt.gca()
ax.annotate('', xy = (1.1, 1.2), xytext = (1.1, t(1.1)), 
            arrowprops = dict(arrowstyle="->", color = 'C2'))
func = mlines.Line2D([], [], color='C0')
yi = mlines.Line2D([], [], color='C1', marker = 'o', ls = 'none')
y4 = mlines.Line2D([], [], color='r', marker = 'o', ls = 'none')
handles = [func, yi, y4]
labels = ['$y(t)$', '$y_i$', '$y(4)$']
ax.legend(handles, labels)
for i in xrange(len(points[0])):
    pltt.plot(points[0][i], points[1][i], 'o', color = 'C1')
    
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

pltt.show()


#Plot 2

def a(z):
    return np.sqrt(1.0 / (1.0 + np.log(z)))

def b(t, y):
    return - (y ** 3) / (2 * t)


def ERK1(s):
    return (abs(a(T) - RKM.RungeKuttaMethod(b, [1.0, [1.0]], s, T, 
                *RKC.ERK1FE)[1][0]))

def ERK2(s):
    return (abs(a(T) - RKM.RungeKuttaMethod(b, [1.0, [1.0]], s, T, 
                *RKC.ERK2MP)[1][0]))

def ERK4(s):
    return (abs(a(T) - RKM.RungeKuttaMethod(b, [1.0, [1.0]], s, T, 
                *RKC.ERK4C)[1][0]))

stepsize = np.logspace(0, -3, 50)
stepsize2 = np.logspace(0, -5, 20)

T = 3.0

#find errors inccurred with diffeent step sizes
x = []
y = []
z = []

for j in xrange(len(stepsize)):
    x.append(ERK1(stepsize[j]))
    y.append(ERK2(stepsize[j]))
for j in xrange(len(stepsize2)):
    z.append(ERK4(stepsize2[j]))


# convert stepsizes to  number of steps
stepsize = T / stepsize
stepsize2 = T / stepsize2


#plot number of steps against error

#Initialize a Figure
fig = pltt.figure()

#Add Axes to Figure
ax = fig.add_subplot(111)

pltt.semilogy(stepsize, x, label = '$RK1$')
pltt.semilogy(stepsize, y, label = '$RK2$')
pltt.semilogy(stepsize2, z, label = '$RK4$')

ax.legend()
pltt.xlabel('Number of Steps')
pltt.ylabel('Error')
pltt.xlim(0, 200)
pltt.ylim(10 ** (-8.0), 1.0)
pltt.minorticks_off()

#plot normalised number of steps against error

#Initialize a Figure
fig = pltt.figure()

#Add Axes to Figure
ax = fig.add_subplot(111)

pltt.semilogy(stepsize, x, label = '$RK1$')
pltt.semilogy(stepsize * 2, y, label = '$RK2$')
pltt.semilogy(stepsize2 * 4, z, label = '$RK4$')

ax.legend()
pltt.xlabel(r'Number of "Normalised" Steps')
pltt.ylabel('Error')
pltt.xlim(0, 200)
pltt.ylim(10 ** (-8.0), 1.0)
pltt.minorticks_off()


#Plot 3

def a(z):
    return np.sqrt(1.0 / (1.0 + np.log(z)))

def b(t, y):
    return - (y ** 3) / (2 * t)


def ERK1(s):
    return (abs(a(T) - RKM.RungeKuttaMethod(b, [1.0, [1.0]], s, T, 
                *RKC.ERK1FE)[1][0]))

def ERK2(s):
    return (abs(a(T) - RKM.RungeKuttaMethod(b, [1.0, [1.0]], s, T, 
                *RKC.ERK2MP)[1][0]))
def ERK4(s):
    return (abs(a(T) - RKM.RungeKuttaMethod(b, [1.0, [1.0]], s, T, 
                *RKC.ERK4C)[1][0]))

def AB2(s):
    return (abs(a(T) - LMM.LinearMultistepMethod(b, [1.0, [1.0]], s, T, 
                *LMMC.AB2)[1][0]))
def AB4(s):
    return (abs(a(T) - LMM.LinearMultistepMethod(b, [1.0, [1.0]], s, T, 
                *LMMC.AB4)[1][0]))

stepsize = np.logspace(1, -3, 50) / 5.0
stepsize2 = np.logspace(0, -5, 15) 

T = 3.0

#find errors inccurred with diffeent step sizes
x = []
y = []
z = []
u = []
v = []

for j in xrange(len(stepsize)):
    x.append(ERK1(stepsize[j]))
    y.append(ERK2(stepsize[j]))
    z.append(AB2(stepsize[j]))
for j in xrange(len(stepsize2)):
    u.append(ERK4(stepsize2[j]))
    v.append(AB4(stepsize2[j]))


# convert stepsizes to  number of steps
stepsize = T / stepsize
stepsize2 = T / stepsize2


#plot number of steps against error

#Initialize a Figure
fig = pltt.figure()

#Add Axes to Figure
ax = fig.add_subplot(111)

pltt.semilogy(stepsize, x, label = '$RK1$')
pltt.semilogy(stepsize, y, label = '$RK2$')
pltt.semilogy(stepsize, z, label = '$AB2$')
pltt.semilogy(stepsize2, u, label = '$RK4$')
pltt.semilogy(stepsize2, v, label = '$AB4$')

ax.legend()
pltt.xlabel('Number of Steps')
pltt.ylabel('Error')
pltt.xlim(0, 200)
pltt.ylim(10 ** (-8.0), 1.0)
pltt.minorticks_off()

#plot normalised number of steps against error

#Initialize a Figure
fig = pltt.figure()

#Add Axes to Figure
ax = fig.add_subplot(111)

pltt.semilogy(stepsize, x, label = '$RK1$')
pltt.semilogy(stepsize * 2, y, label = '$RK2$')
pltt.semilogy(stepsize, z, label = '$AB2$')
pltt.semilogy(stepsize2 * 4, u, label = '$RK4$')
pltt.semilogy(stepsize2, v, label = '$AB4$')

ax.legend()
pltt.xlabel(r'Number of "Normalised" Steps')
pltt.ylabel('Error')
pltt.xlim(0, 200)
pltt.ylim(10 ** (-8.0), 1.0)
pltt.minorticks_off()


#Plot 4

def a(z):
    return 0.1 * (z **2) + 0.5 *z  + np.sin((10 - z) * z) * 1 / (z)

x = np.linspace(0 , 6, 5000)

y = a(x)

#Initialize a Figure
fig = pltt.figure()

#Add Axes to Figure
ax = fig.add_subplot(111)

pltt.plot(x, y, label = '$y(t)$')
pltt.plot(5.5, a(5.5), 'o' ,color = 'r', label = '$y(t^*)$')

ax.legend()
ax.grid()
pltt.xlim(0, 5.8)
pltt.ylim(-3, 9.5)

 # making the top and right spine invisible:
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')

# moving bottom spine up to y=0 position:
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data',0))