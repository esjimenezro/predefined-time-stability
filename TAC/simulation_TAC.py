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
plt.figure(figsize=(8,6), num=1)
plt.plot(t, xref[0], lw = 5, color=0*np.ones(3))
plt.plot(t, xref[1], lw = 5, color=.3*np.ones(3))

# Reference signal plot in coordinates frame
plt.figure(figsize=(8,8), num=2)
plt.plot(xref[0], xref[1], lw = 5, color=0*np.ones(3))


for x0 in xx0:
    t, x = sol.ode1(system, np.array([x0, x0]), t0, tf, h)
    e = x-xref
    #if x0>=0:
    #    T_x0[i] = np.argmax(np.abs(x)<1.5e-4)*h
    #    i      += 1
    plt.figure(num=1)
    plt.plot(t, x[0], color=0.5*np.ones(3))
    plt.plot(t, x[1], color=0.7*np.ones(3))
    plt.ylim(-2, 2)
    
    plt.figure(num=2)
    plt.plot(x[0], x[1], color=0.7*np.ones(3))
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
     
