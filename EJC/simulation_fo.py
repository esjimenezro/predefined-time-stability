#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 15:52:48 2018

@author: esteban
"""
import numpy as np
import solver as sol
import matplotlib.pyplot as plt
import matplotlib as mpl

label_size = 16
mpl.rcParams['xtick.labelsize'] = label_size
mpl.rcParams['font.size'] = label_size

def predefined1(x, q, Tc):
    return -1/(q*Tc)*np.exp(np.abs(x)**q)*sol.odd_pow(x, 1-q)

def predefined2(x, q, Tc):
    return -np.pi/(2*q*Tc)*(sol.odd_pow(x, 1+q) + sol.odd_pow(x, 1-q))

def predefined3(x, q, a, Tc):
    return -1/(a*q*Tc)*(np.abs(x)**q+a)**2*sol.odd_pow(x, 1-q)

def predefined4(x, q, a, Tc):
    from scipy.special import gamma
    return -gamma(a)/(q*Tc)*np.exp(np.abs(x)**q)*sol.odd_pow(x, 1-a*q)

def system(t, x):
    import solver as sol
    import numpy as np
    Delta = np.sin(2*np.pi*t/5)
    q, Tc, zeta = 0.3, 1, 1
    a = 1
    return predefined4(x,q,a,Tc)-zeta*sol.odd_pow(x, 0)+Delta


t0, tf, h,  i = 0, 1.2, 1e-5, 0
xx0 = np.logspace(-1, 3, 5)
T_x0 = np.zeros(xx0.size)
plt.figure(figsize=(8,6), num=1)
plt.figure(figsize=(8,6), num=2)

for x0 in xx0:
    t, x = sol.ode1(system, x0, t0, tf, h)
    if x0>=0:
        T_x0[i] = np.argmax(np.abs(x)<1.5e-4)*h
        i      += 1
    plt.figure(num=1)
    plt.plot(t, x[0], color=0*np.ones(3))
    
# Trajectories    
plt.figure(num=1)
plt.ylim(-3, 5)
plt.xlim(0, 1.2)
plt.xlabel('$t$', fontsize = 18)
plt.ylabel('$x(t,x_0)$', fontsize = 18)
plt.axvline(x = 1, ymin = -1, ymax = 2, linestyle='dashed', color = 0.3*np.ones(3))
plt.grid()
plt.savefig('figures/basic.eps', bbox_inches='tight', format='eps', dpi=1500)

# Settling-time function figure
plt.figure(num=2)
plt.semilogx(xx0, T_x0, 'k', lw=2)
plt.grid()
plt.xlabel('|$x_0$|', fontsize = 18)
plt.ylabel('$T(x_0)$', fontsize = 18)
plt.axhline(y = 1, xmin = -1, xmax = 2, linestyle='dashed', color = 0.3*np.ones(3))
plt.ylim(0, 1.2)
plt.savefig('figures/settling_basic.eps', bbox_inches='tight', format='eps', dpi=1500)