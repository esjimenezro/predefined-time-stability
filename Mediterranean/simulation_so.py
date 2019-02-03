#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 16:47:17 2018

@author: esteban
"""

import numpy as np
from solver import odd_pow, ode1
from scipy.special import beta
import matplotlib.pyplot as plt
import matplotlib as mpl
label_size = 14
mpl.rcParams['xtick.labelsize'] = label_size
mpl.rcParams['font.size'] = label_size
mpl.rcParams['agg.path.chunksize'] = 10000

def phi1(x, r):
    r0, r1, r2, r3, r4, r5, r6 = r
    mr5 = (1-r4*r5)/(r6-r5)
    mr6 = (r4*r6-1)/(r6-r5)
    g = beta(mr5, mr6)/(r2**(mr6)*r3**(mr5)*(r6-r5))
    return -((g/r1)*(r2*np.abs(x)**r5+r3*np.abs(x)**r6)**r4+r0)*np.sign(x)

def phi2(x1, x2, r1, r2, r7):
    r11, r21, r31, r41, r51, r61 = r1
    mr51 = (1-r41*r51)/(r61-r51)
    mr61 = (r41*r61-1)/(r61-r51)
    g1 = beta(mr51, mr61)/(r21**(mr61)*r31**(mr51)*(r61-r51))
    
    r12, r22, r32, r42, r52, r62 = r2
    mr52 = (1-r42*r52)/(r62-r52)
    mr62 = (r42*r62-1)/(r62-r52)
    g2 = beta(mr52, mr62)/(r22**(mr62)*r32**(mr52)*(r62-r52))
    
    xi = r21*odd_pow(x1, r51)+r31*odd_pow(x1, r61)
    
    s = x2 + odd_pow(odd_pow(x2, 2)+odd_pow(xi, 2*r41), 1/2)
    
    return -((g2/r12)*(r22*np.abs(s)**r52+r32*np.abs(s)**r62)**r42+r7)*np.sign(s)-(2*g1**2*r41/(r11**2))*(r21*r51+r31*r61*np.abs(x1)**(r61-r51))*(r21+r31*np.abs(x1)**(r61-r51))*np.abs(x1)**(2*r41*r51-1)*np.sign(s)



def system(t, x):
    # Controller parameters
    r = np.array([0.1, 2, 1, 1, 1, 0.9, 1.1])
    r1 = np.array([0.5, 1, 1, 0.6, 1, 3])
    r2 = np.array([5, 1, 1, 1, 0.9, 1.1])
    r7 = 0.03
    r8 = -0.1
    r9 = 0.3
    # Perturbations
    D0 = 0.1*np.sin(t)
    D1 = 0.3*np.sin(t)
    # State variables
    x0, x1, x2 = x
    # Control inputs
    if t<r1[0]+r2[0]:
        u0 = r8
        u1 = phi2(x1, r8*x2, r1, r2, r7)/r8
    else:
        u0 = phi1(x0, r)
        u1 = -r9*np.sign(x2)
    return np.array([u0+D0, u0*x2, u1+D1])


# Simulation parameters
t0, tf, h = 0, 10, 1e-4
# Simulation
t, x = ode1(system, np.array([3, -1, 0]), t0, tf, h)
# States
x0, x1, x2 = x
# Control
# Controller parameters
r = np.array([0.1, 2, 1, 1, 1, 0.9, 1.1])
r1 = np.array([0.5, 1, 1, 0.5, 1, 3])
r2 = np.array([5, 1, 1, 1, 0.9, 1.1])
r7 = 0.03
r8 = -0.1
r9 = 0.3
u0 = np.zeros(len(x0))
u1 = np.zeros(len(x0))
N = int(5.5/1e-4)
u0[:N+1] = r8
u0[N+1:] = phi1(x0[N+1:], r)
u1[:N+1] = phi2(x1[:N+1], r8*x2[:N+1], r1, r2, r7)/r8
u1[N+1:] = r9*np.sign(x2[N+1:])
# Original variables
x = x0
y = x1
theta = np.arctan(x2)
v = u0/np.cos(theta)
w = u1*np.cos(theta)**2

# Figures
# States   
plt.figure(num=1)
plt.plot(t, x, color=0*np.ones(3), lw=3, label='$x(t)$')
plt.plot(t, y, color=0.3*np.ones(3), lw=3, label='$y(t)$')
plt.plot(t, theta, color=0.6*np.ones(3), lw=2, label=r'$\theta(t)$')
plt.xlabel('Time (s)', fontsize = 14)
plt.legend(loc='best')
plt.text(7.5, -1, '$T_c=7.5$ s')
plt.axvline(x = 7.5, ymin = -2, ymax = 3, linestyle='dashed', color = 0*np.ones(3))
plt.grid()
plt.savefig('figures/states.eps', bbox_inches='tight', format='eps', dpi=1500)

# Control
plt.figure(num=2)
plt.plot(t, w, color=0.5*np.ones(3), lw=2, label='$w(t)$')
plt.plot(t, v, color=0*np.ones(3), lw=2, label='$v(t)$')
plt.xlabel('Time (s)', fontsize = 14)
plt.ylim(-200,200)
plt.legend(loc='best')
plt.grid()
#plt.savefig('figures/control.eps', bbox_inches='tight', format='eps', dpi=1500)