#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 16:47:17 2018

@author: esteban
"""

import numpy as np
import solver as sol
from scipy.special import gamma
import matplotlib.pyplot as plt
import matplotlib as mpl
label_size = 14
mpl.rcParams['xtick.labelsize'] = label_size
mpl.rcParams['font.size'] = label_size
mpl.rcParams['agg.path.chunksize'] = 10000

def invdW1(x, q):
    return 1/q*np.exp(np.abs(x)**q)*sol.odd_pow(x, 1-q)

def proddW1(x, q):
    return (1/q)**2*np.exp(2*np.abs(x)**q)*(q*np.abs(x)**q+1-q)*np.abs(x)**(1-2*q)

def invdW2(x, q):
    return np.pi/(2*q)*(sol.odd_pow(x, 1+q) + sol.odd_pow(x, 1-q))

def proddW2(x, q):
    return (np.pi/(2*q))**2*((1+q)*np.abs(x)**(2*q)+1-q)*(np.abs(x)+np.abs(x)**(1-2*q))

def invdW3(x, q, a):
    return 1/(a*q)*(np.abs(x)**q+a)**2*sol.odd_pow(x, 1-q)

def proddW3(x, q, a):
    return (1/(a*q))**2*(np.abs(x)**q+a)**3*(2*np.abs(x)**q+(1-q)*(np.abs(x)**q+a))*np.abs(x)**(1-2*q)

def invdW4(x, q, a):
    return gamma(a)/(q)*np.exp(np.abs(x)**q)*sol.odd_pow(x, 1-a*q)

def proddW4(x, q, a):
    return (gamma(a)/(q))**2*np.exp(2*np.abs(x)**q)*(q*np.abs(x)**q+1-a*q)*np.abs(x)**(1-2*a*q)

def invdW(i, x, q, a):
    if i==1:
        return invdW1(x, q)
    elif i==2:
        return invdW2(x, q)
    elif i==3:
        return invdW3(x, q, a)
    elif i==4:
        return invdW4(x, q, a)

def proddW(i, x, q, a):
    if i==1:
        return proddW1(x, q)
    elif i==2:
        return proddW2(x, q)
    elif i==3:
        return proddW3(x, q, a)
    elif i==4:
        return proddW4(x, q, a)


def system(t, x):
    # System parameters
    g, mc, m, l = 9.8, 1, 0.1, 0.5
    # Controller parameters
    r1, r2, r3 = 0.618, 0.382, 2
    q1, q2, a1, a2 = 0.2, 0.2, 1, 1
    i = 3
    # Reference
    r = np.sin(0.5*np.pi*t)
    dr = 0.5*np.pi*np.cos(0.5*np.pi*t)
    d2r = -(0.5*np.pi)**2*r
    # State variables
    x1, x2 = x[0], x[1]
    # Error variables
    e1, e2 = x1-r, x2-dr
    # Disturbance
    Delta = np.sin(10*x1)+np.cos(x2)
    # Sliding variable
    s = e2 + sol.odd_pow(sol.odd_pow(e2,2)+
        2/(r1**2)*sol.odd_pow(invdW(i,e1,q1,a1),2),0.5)
    # Controller
    us = -1/r2*invdW(i,s,q2,a2)-r3*np.sign(invdW(i,s,q2,a2))-2/(r1**2)*proddW(i,e1,q1,a1)*np.sign(invdW(i,s,q2,a2))
    f = (g*np.sin(x1)-m*l*x2**2*np.cos(x1)*np.sin(x1)/(mc+m))/(l*(4/3-m*np.cos(x1)**2/(mc+m)))-d2r
    g = (np.cos(x1)/(mc+m))/(l*(4/3-m*np.cos(x1)**2/(mc+m)))
    u = (-f+us)/g
    return np.array([x2, f+g*u+Delta])


# Simulation parameters
t0, tf, h, i = 0, 1.2, 1e-6, 0
# Simulation
t, x = sol.ode1(system, np.array([-1, 0]), t0, tf, h)
# States
x1, x2 = x
# Reference
r = np.sin(0.5*np.pi*t)
dr = 0.5*np.pi*np.cos(0.5*np.pi*t)
d2r = -(0.5*np.pi)**2*r
# Error variables
e1, e2 = x1-r, x2-dr
# Controller
# System parameters
g, mc, m, l = 9.8, 1, 0.1, 0.5
# Controller parameters
r1, r2, r3 = 0.618, 0.382, 2
q1, q2, a1, a2 = 0.2, 0.2, 1, 1
i = 3
s = e2 + sol.odd_pow(sol.odd_pow(e2,2)+2/(r1**2)*sol.odd_pow(invdW(i,e1,q1,a1),2),0.5)
us = -1/r2*invdW(i,s,q2,a2)-r3*np.sign(invdW(i,s,q2,a2))-2/(r1**2)*proddW(i,e1,q1,a1)*np.sign(invdW(i,s,q2,a2))
f = (g*np.sin(x1)-m*l*x2**2*np.cos(x1)*np.sin(x1)/(mc+m))/(l*(4/3-m*np.cos(x1)**2/(mc+m)))-d2r
g = (np.cos(x1)/(mc+m))/(l*(4/3-m*np.cos(x1)**2/(mc+m)))
u = (-f+us)/g

# Trajectories    
plt.figure(num=1)
plt.plot(t, r, color=0.5*np.ones(3), lw=3, label='$r(t)$')
plt.plot(t, dr, color=0.7*np.ones(3), lw=3, label='$\dot{r}(t)$')
plt.plot(t, x1, '--', color=0*np.ones(3), lw=2, label='$x_1(t)$')
plt.plot(t, x2, '--', color=0.3*np.ones(3), lw=2, label='$x_2(t)$')
plt.ylim(-2,5)
plt.xlim(0, 1.2)
plt.xlabel('$t$', fontsize = 14)
plt.legend(loc=9)
plt.text(1, -1.3, '$T_c=1$')
plt.axvline(x = 1, ymin = -1, ymax = 2, linestyle='dashed', color = 0.6*np.ones(3))
plt.grid()
#plt.savefig('figures/trajW4.eps', bbox_inches='tight', format='eps', dpi=1500)

# Error    
plt.figure(num=2)
plt.plot(t, e1, '--', color=0*np.ones(3), lw=2, label='$e_1(t)$')
plt.plot(t, e2, '--', color=0.3*np.ones(3), lw=2, label='$e_2(t)$')
plt.ylim(-2,5)
plt.xlim(0, 1.2)
plt.xlabel('$t$', fontsize = 14)
plt.legend(loc=9)
plt.text(1, -1.3, '$T_c=1$')
plt.axvline(x = 1, ymin = -1, ymax = 2, linestyle='dashed', color = 0.6*np.ones(3))
plt.grid()
#plt.savefig('figures/errorW4.eps', bbox_inches='tight', format='eps', dpi=1500)

# Control    
plt.figure(num=3)
plt.plot(t, u, '--', color=0*np.ones(3))
plt.ylim(-100,100)
plt.xlim(0, 1.2)
plt.xlabel('$t$', fontsize = 14)
plt.ylabel('$u$', fontsize = 14)
plt.grid()
#plt.savefig('figures/controllerW4.eps', bbox_inches='tight', format='eps', dpi=1500)

IAEu = np.abs(u).sum()*h