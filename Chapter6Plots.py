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


#Plot #3 Cusp

a1 = np.linspace(0, 1, 1000)

def rotate(al):
    return (np.array([[np.cos(al), -np.sin(al)],
                      [np.sin(al), np.cos(al)]]))


def p(t):
    return np.array([t, np.sqrt(t ** 3)])
def q(t):
    return np.array([t, - np.sqrt(t ** 3)])

a = p(a1)
b = np.dot(rotate(0.8), a)

c = q(a1)
d = np.dot(rotate(0.8), c)


#Initialize a Figure
fig = plt.figure()

#Add Axes to Figure
ax = fig.add_subplot(111)

#Plot Cusp
ax1 = ax.plot(b[0], b[1], color = 'C0')
ax2 = ax.plot(d[0], d[1], color = 'C0')

#Mark z_0 
ax3 = ax.plot(0, 0, 'o', ms = 10, color = 'C1')
ax.annotate('$z_0$', xy = (0.0, 0.0), xytext = (-0.01, 0.013), color = 'C1', 
            fontsize = 20)

#Mark gamma
e = [d[0][150], d[1][150]]
f = [b[0][150], b[1][150]]

theta1 = np.arctan2(e[1], e[0])
theta2 = np.arctan2(f[1], f[0])

def gamma(t):
    return np.sqrt(e[0] ** 2 + e[1] ** 2) * np.array([np.cos(theta1 + t * 
                  (theta2 - theta1)), np.sin(theta1 + t * (theta2 - theta1))])
t = np.linspace(0,1, 1000)
g = gamma(t)
ax4 = ax.plot(g[0], g[1], color = 'C1')

t = np.linspace(-0.6, 1.6, 1000)
g = gamma(t)
ax5 = ax.plot(g[0], g[1], color = 'C1', ls = ':')

ax6 = ax.plot(e[0], e[1], 'o', ms = 10, color = 'C1')
ax.annotate('$z_{1_n}$', xy = (e[0], e[1]), xytext = (e[0] + 0.001, e[1] + 
            0.012), color = 'C1', fontsize = 20)

ax7 = ax.plot(f[0], f[1], 'o', ms = 10, color = 'C1')
ax.annotate('$z_{2_n}$', xy = (f[0], f[1]), xytext = (f[0]+ 0.01, f[1]), 
            color = 'C1', fontsize = 20)

# making the spines invisible:
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.spines['bottom'].set_color('none')
ax.spines['left'].set_color('none')

# getting rid of ticks
ax.tick_params(
    axis ='both',          
    which ='both',      
    bottom ='off',
    top='off', 
    left = 'off',
    right = 'off',
    labelbottom ='off',
    labelleft = 'off'
    )

#setup legend
lvlSet = mlines.Line2D([], [], color='C0')
gamma = mlines.Line2D([], [], color='C1')
circle = mlines.Line2D([], [], color='C1', ls = ':')
handles = [lvlSet, gamma, circle]
labels = [r'$\phi(r, \theta ) = 1$', r'$\gamma_n$',
          r'$\left \{ z : \left | z - z_0 \right | = \frac{1}{n} \right \} $']
ax.legend(handles, labels)

plt.ylim(-0.01, 0.25)
plt.xlim(-0.01, 0.25)
plt.axes().set_aspect('equal', 'datalim')