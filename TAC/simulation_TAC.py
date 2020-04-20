#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 11:54:39 2018

@author: esteban
"""

# Predefined-time stabilizing function
def predefined(x, a, b, p, q, k):
    from scipy.special import gamma
    import solver as sol
    mp, mq = (1-k*p)/(q-p), (k*q-1)/(q-p)
    g = gamma(mp)*gamma(mq)/((a**k)*gamma(k)*(q-p))*(a/b)**mp
    return g*sol.vec_pow(a*sol.vec_pow(x, p)+b*sol.vec_pow(x, q), k)

# System
def system(t, x):
    import numpy as np
    xref = np.array([np.cos(2*np.pi*t), np.sin(2*np.pi*t)])
    #dxref = 2*np.pi*np.array([-np.sin(2*np.pi*t), np.cos(2*np.pi*t)])
    e = x-xref
    r1, r2, r3, r4 = 1, 0, 2*2*np.pi, 0*1e-2
    r5, r6, r7, r8, r9 = 1, 1, 0.9, 1.1, 1
    u = -1/r1*predefined(e,r5,r6,r7,r8,r9)-r3*e/(np.linalg.norm(e,axis=0)+r4)
    #Delta = -dxref 
    return u

# Libraries
import numpy as np
import solver as sol
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
# Numbers size in graphs
label_size = 16
mpl.rcParams['xtick.labelsize'] = label_size
mpl.rcParams['font.size'] = label_size

# Simulation
t0, tf, h = 0, 2, 1e-5
xx0 = np.array([10, 1000, 1e21])
#xx0 = np.logspace(1, 10, 10)
#xx0 = np.array([10**1])
#T_x0 = np.zeros(xx0.size)

# Time-vector
t = np.arange(t0, tf+h, h)
# Reference signal
xref = np.array([np.cos(2*np.pi*t), np.sin(2*np.pi*t)])

# Reference signal plot
plt.figure(figsize=(8,8), num=1)
gs1 = GridSpec(4, 1)
plt.subplot(gs1[:3,0])
plt.plot(t, xref[0], lw = 5, color=0*np.ones(3), label='$r_1(t)$')
plt.plot(t, xref[1], lw = 5, color=.2*np.ones(3), label='$r_2(t)$')

# Reference signal plot in coordinates frame
plt.figure(figsize=(8,8), num=2)
plt.plot(xref[0], xref[1], lw = 5, color=0*np.ones(3), label='$(r_1,r_2)$')


for x0 in xx0:
    t, x = sol.ode1(system, np.array([x0, x0]), t0, tf, h)
    e = x-xref
    
    if x0==xx0[-1]:
        plt.figure(num=1)
        plt.subplot(gs1[:3,0])
        plt.plot(t, x[0], color=0.4*np.ones(3), label='$x_1(t)$')
        plt.plot(t, x[1], color=0.6*np.ones(3), label='$x_2(t)$')
        plt.ylim(-2, 2)
        plt.subplot(gs1[3,0])
        plt.plot(t, np.linalg.norm(e, axis=0), 'k')
        
        plt.figure(num=2)
        plt.plot(x[0], x[1], color=0.7*np.ones(3), label='$(x_1,x_2)$')
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
    else:
        plt.figure(num=1)
        plt.subplot(gs1[:3,0])
        plt.plot(t, x[0], color=0.4*np.ones(3))
        plt.plot(t, x[1], color=0.6*np.ones(3))
        plt.ylim(-2, 2)
        plt.subplot(gs1[3,0])
        plt.ylim(0, 1e-1)
        plt.plot(t, np.linalg.norm(e, axis=0), 'k')
        
        plt.figure(num=2)
        plt.plot(x[0], x[1], color=0.7*np.ones(3))
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
    
    
"""    
plt.figure(num=1)
plt.subplot(gs1[:3,0])
plt.grid()
plt.axvline(x=1, ymin=-2, ymax=2, color=0.5*np.ones(3))
#plt.xlabel('$t$')
plt.text(0.1, 1.5, '$x_0=10^1$')
plt.text(0.25, 1.2, '$x_0=10^3$')
plt.text(0.5, 1.0, '$x_0=10^{21}$')
plt.text(1.1, -1.5, '$T_c=1$')
plt.legend(loc='best')
plt.subplot(gs1[3,0])
plt.ylim(-1e-2, 1e-2)
plt.grid()
plt.axvline(x=1, ymin=-2, ymax=2, color=0.5*np.ones(3))
plt.axhline(y=1e-2, xmin=0, xmax=2, ls='--', color=0.5*np.ones(3))
plt.xlabel('$t$')
plt.ylabel('$||s(t)||$')
plt.text(0.1, 0.0055, '$x_0=10^1$')
plt.text(0.2, 0.0025, '$x_0=10^3$')
plt.text(0.54, -0.0035, '$x_0=10^{21}$')
plt.text(1.1, 0.005, '$T_c=1$')
#plt.text(0.1, 0.0055, '$x_0=10^1$')
#plt.text(0.2, 0.015, '$x_0=10^3$')
#plt.text(0.54, 0.012, '$x_0=10^{21}$')
#plt.text(1.1, 0.005, '$T_c=1$')
#plt.text(1.25, 0.011, '$b=0.01$')
plt.savefig('figures/xvst_disc.eps', bbox_inches='tight', format='eps', dpi=1500)
"""

plt.figure(num=2)
plt.grid()
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.text(1.5, 0.7, '$x_0=10^1$')
plt.text(1.0, 1.4, '$x_0=10^3$')
plt.text(0.0, 1.5, '$x_0=10^{21}$')
plt.legend(loc='best')
plt.savefig('figures/x2vsx1_disc.eps', bbox_inches='tight', format='eps', dpi=1500)
     
