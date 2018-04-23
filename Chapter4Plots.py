# -*- coding: utf-8 -*-
"""
Created on Thu Feb 01 10:52:23 2018

@author: Alastair Wiseman
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import PolynomialOrderStar as POS


#Plot #1: Trapezium Rule Order Star

def p(z):
    return (1.0 + 0.5 * z) / (1.0 - 0.5 * z)

POS.polyOrderStar(p, -3, 3, -2, 2) 


#Plot #2: Index Example

def p(z):
    return 1.0 + z + z ** 2 /2.0 + z ** 3 / 6.0 + z ** 4 / 24.0  
 
    
#Plot #3

x = np.linspace(0, np.pi, 1000)

c1 = 1 + np.exp(x * 1j) / 2.0
c2 = np.sqrt(2) / 2.0 + (np.sqrt(2) * 1j) / 2.0 + np.exp(x * 1j) / 2.0
c3 = 1j + np.exp(x * 1j) / 2.0
c4 = - np.sqrt(2) / 2.0 + (np.sqrt(2) * 1j) / 2.0 + np.exp(x * 1j) / 2.0
c5 = -1 + np.exp(x * 1j) / 2.0

#Initialize a Figure
fig = plt.figure()

#Add Axes to Figure
ax = fig.add_subplot(111)

#plot first chain

ax.plot(np.real(c1), np.imag(c1), color = 'C2')
ax.plot(np.real(c1), - np.imag(c1), color = 'C2')
ax.fill_between(np.real(c1), np.imag(c1), -np.imag(c1), color = 'C2', 
                alpha = 0.1)
ax.plot(np.real(c2), np.imag(c2), color = 'C0')
ax.plot(np.real(c2), 2 * np.imag(c2[0]) - np.imag(c2), color = 'C0')
ax.fill_between(np.real(c2), np.imag(c2), 2 * np.imag(c2[0]) - 
                np.imag(c2), color = 'C0', alpha = 0.1)
ax.plot(np.real(c3), np.imag(c3), color = 'C0')
ax.plot(np.real(c3), 2 * np.imag(c3[0]) - np.imag(c3), color = 'C0')
ax.fill_between(np.real(c3), np.imag(c3), 2 * np.imag(c3[0]) - 
                np.imag(c3), color = 'C0', alpha = 0.1)
ax.plot(np.real(c4), np.imag(c4), color = 'C0')
ax.plot(np.real(c4), 2 * np.imag(c4[0]) - np.imag(c4), color = 'C0')
ax.fill_between(np.real(c4), np.imag(c4), 2 * np.imag(c4[0]) - 
                np.imag(c4), color = 'C0', alpha = 0.1)
ax.plot(np.real(c5), np.imag(c5), color = 'C3')
ax.plot(np.real(c5), - np.imag(c5), color = 'C3')
ax.fill_between(np.real(c5), np.imag(c5), -np.imag(c5), color = 'C3',
                alpha = 0.1)

#plot second chain

ax.plot(np.real(c1), np.imag(c1), color = 'C2')
ax.plot(np.real(c1), - np.imag(c1), color = 'C2')
ax.fill_between(np.real(c1), np.imag(c1), -np.imag(c1), color = 'C2', 
                alpha = 0.1)
ax.plot(np.real(c2), - np.imag(c2), color = 'C1')
ax.plot(np.real(c2), - (2 * np.imag(c2[0]) - np.imag(c2)), color = 'C1')
ax.fill_between(np.real(c2), - np.imag(c2), - (2 * np.imag(c2[0]) 
                     - np.imag(c2)), color = 'C1', alpha = 0.1)
ax.plot(np.real(c3), - np.imag(c3), color = 'C1')
ax.plot(np.real(c3), - (2 * np.imag(c3[0]) - np.imag(c3)), color = 'C1')
ax.fill_between(np.real(c3), - np.imag(c3), - (2 * np.imag(c3[0]) - 
                        np.imag(c3)), color = 'C1', alpha = 0.1)
ax.plot(np.real(c4), - np.imag(c4), color = 'C1')
ax.plot(np.real(c4), - (2 * np.imag(c4[0]) - np.imag(c4)), color = 'C1')
ax.fill_between(np.real(c4), - np.imag(c4), - (2 * np.imag(c4[0]) - 
                        np.imag(c4)), color = 'C1', alpha = 0.1)
ax.plot(np.real(c5), np.imag(c5), color = 'C3')
ax.plot(np.real(c5), - np.imag(c5), color = 'C3')
ax.fill_between(np.real(c5), np.imag(c5), -np.imag(c5), color = 'C3', 
                alpha = 0.1)

#setup legend
omega_1 = mpatches.Rectangle((0, 0), 1, 1, fc="C0",alpha=0.1)
omega_2 = mpatches.Rectangle((0, 0), 1, 1, fc="C1",alpha=0.1)
omega_3 = mpatches.Rectangle((0, 0), 1, 1, fc="C2",alpha=0.2)
omega_4 = mpatches.Rectangle((0, 0), 1, 1, fc="C3",alpha=0.2)
handles = [omega_3, omega_4, omega_1, omega_2]
labels = [r'$\Omega$', r'$\Omega_*$', r'$\Omega_i$', r'$\widetilde{\Omega}_i$']
ax.legend(handles, labels, fontsize = 14)

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
plt.xlim(-1.6, 2.6)
plt.ylim(-1.6, 1.6)
plt.axes().set_aspect('equal')


#Plot #4

def func1(z):
    return (2.0 + np.sqrt(1 + 2 * z)) / (3.0 - 2.0 * z)

def func2(z):
    return (2.0 - np.sqrt(1 + 2 * z)) / (3.0 - 2.0 * z)

#setup grid for function evaluations
A = np.linspace(-5, 5, 1000)
B = 1j * np.linspace(-5, 5, 1000)
p, q = np.meshgrid(A,B)
C = p + q

#evaluate the Runge-Kutta stability function on the grid
for i in xrange(1000):
    for j in xrange(1000):
        C[i][j] = np.abs(func1(C[i][j]))

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
ax.legend(handles, labels)

#draw discontinuity
plt.plot([-10.0,-0.5], [0., 0.], color = 'r', lw = 2.0)

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
plt.ylim(-3, 3)
plt.xlim(-3, 5)
plt.axes().set_aspect('equal')
plt.show()

#setup grid for function evaluations
A = np.linspace(-5, 5, 1000)
B = 1j * np.linspace(-5, 5, 1000)
p, q = np.meshgrid(A,B)
C = p + q

#evaluate the Runge-Kutta stability function on the grid
for i in xrange(1000):
    for j in xrange(1000):
        C[i][j] = np.abs(func2(C[i][j]))

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
ax.legend(handles, labels)

#draw discontinuity
plt.plot([-10.0,-0.5], [0., 0.], color = 'r', lw = 2.0)

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
plt.ylim(-3, 3)
plt.xlim(-3, 5)
plt.axes().set_aspect('equal')
plt.show()


#Plot 5

#BDF2 Factors   
def l(z):
    return (2.0 + np.sqrt(1.0 + 2.0 * z)) / (3.0 * (1.0 - z * (2.0 / 3.0)))

def m(z):
    return (2.0 - np.sqrt(1.0 + 2.0 * z)) / (3.0 * (1.0 - z * (2.0 / 3.0)))

#BDF2 Order Star
POS.polyOrderStar(l, -3, 3, -3, 3, 1000, False)

POS.polyOrderStar(m, -3, 3, -3, 3, 1000, False, True)

#Add branch point
plt.plot(-0.5, 0, 'x', color = 'C1')

#make legend
OS = mpatches.Rectangle((0, 0), 1, 1, fc="C0",alpha=0.1)
OSB = mlines.Line2D([], [], color='C0')
DOS = mpatches.Rectangle((0, 0), 1, 1, ec="C0",alpha=1, fill = False)
zeros = mlines.Line2D([], [], color='C1', marker = 'o', ls = 'none')
poles = mlines.Line2D([], [], color='C1', marker = 's', ls = 'none')
branch = mlines.Line2D([], [], color = 'C1', marker = 'x', ls = 'none')
handles = [OS, OSB, DOS, zeros, poles, branch]
labels = ['$\mathcal{A}_+^{\mathcal{R}}$', '$\mathcal{A}_0^{\mathcal{R}}$',
          '$\mathcal{A}_-^{\mathcal{R}}$', 'Zeros', 'Poles', 'Branch Points']
plt.legend(handles, labels)

plt.xlim(-2,2)
plt.ylim(-1.5, 2.)