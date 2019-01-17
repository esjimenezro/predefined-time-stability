#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 16:47:17 2018

@author: esteban
"""

import numpy as np
import solver as sol
import matplotlib.pyplot as plt
import matplotlib as mpl
label_size = 16
mpl.rcParams['xtick.labelsize'] = label_size
mpl.rcParams['font.size'] = label_size

def invdW1(x, q):
    return 1/q*np.exp(np.abs(x)**q)*sol.odd_pow(x, 1-q)

def proddW1(x, q):
    return (1/q)**2*np.exp(2*np.abs(x)**q)*(q*np.abs(x)**q+1-q)*np.abs(x)**(1-2*q)

def invdW2(x, q):
    return np.pi/(2*q)*(sol.odd_pow(x, 1+q) + sol.odd_pow(x, 1-q))

def proddW2(x, q):
    return (np.pi/(2*q))**2*((1+q)*np.abs(x)**(2*q)+1-q)*(np.absx+np.abs(x)**(1-2*q))

def invdW3(x, q, a):
    return 1/(a*q)*(np.abs(x)**q+a)**2*sol.odd_pow(x, 1-q)

def proddW3(x, q, a):
    return (1/(a*q))**2*(np.abs(x)**q+a)**3*(2*np.abs(x)**q+(1-q)*(np.abs(x)**q+a))*np.abs(x)**(1-2*q)

def invdW4(x, q, a):
    from scipy.special import gamma
    return gamma(a)/(q)*np.exp(np.abs(x)**q)*sol.odd_pow(x, 1-a*q)

def proddW4(x, q, a):
    return (gamma(a)/(q))**2*np.exp(2*np.abs(x)**q)*(q*np.abs(x)**q+1-a*q)*np.abs(x)**(1-2*a*q)


def system(t, x):
    import solver as sol
    import numpy as np
    
    x1, x2 = x[0], x[1]
    
    Delta = np.sin(2*np.pi*t/5)
    
    r1, r2, r3 = 0.1, 1, 1
    q1, q2, a1, a2 = 0.3, 0.3, 1, 1
        
    s = x2 + sol.odd_pow(sol.odd_pow(x2,2)+
        2/(r1**2)*sol.odd_pow(invdW1(x1,q1),2),0.5)
    
    u = -1/r2*invdW1(s,q2)-r3*np.sign(s)-2/(r1**2)*np.abs(invdW1(x1,q1))*
    
    return np.array([x2, u+Delta])



t0, tf, h,  i = 0, 1.2, 1e-5, 0
#xx0 = np.logspace(-1, 5, 7)

t, x = sol.ode1(system, np.array([0, 0]), t0, tf, h)

#xx0 = np.logspace(-1, 1, 3)
#T_x0 = np.zeros(xx0.size)
#plt.figure(figsize=(8,6), num=1)
#plt.figure(figsize=(8,6), num=2)
#
#for x0 in xx0:
#    t, x = sol.ode1(system, np.array([x0, 0]), t0, tf, h)
#    if x0>=0:
#        T_x0[i] = np.argmax(np.linalg.norm(x)<1.5e-4)*h
#        i      += 1
#    plt.figure(num=1)
#    plt.plot(t, x[0], color=0*np.ones(3))
#    plt.plot(t, x[1], color=.5*np.ones(3))
#    
## Trajectories    
#plt.figure(num=1)
#plt.ylim(-3, 5)
#plt.xlim(0, 1.2)
#plt.xlabel('$t$', fontsize = 18)
#plt.ylabel('$x(t,x_0)$', fontsize = 18)
#plt.axvline(x = 1, ymin = -1, ymax = 2, linestyle='dashed', color = 0.3*np.ones(3))
#plt.grid()
#plt.savefig('figures/basic2.eps', bbox_inches='tight', format='eps', dpi=1500)
#
## Settling-time function figure
#plt.figure(num=2)
#plt.semilogx(xx0, T_x0, 'k', lw=2)
#plt.grid()
#plt.xlabel('|$x_0$|', fontsize = 18)
#plt.ylabel('$T(x_0)$', fontsize = 18)
#plt.axhline(y = 1, xmin = -1, xmax = 2, linestyle='dashed', color = 0.3*np.ones(3))
#plt.ylim(0, 1.2)
#plt.savefig('figures/settling_basic2.eps', bbox_inches='tight', format='eps', dpi=1500)