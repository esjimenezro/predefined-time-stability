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
label_size = 14
mpl.rcParams['xtick.labelsize'] = label_size
mpl.rcParams['font.size'] = label_size
mpl.rcParams['agg.path.chunksize'] = 10000

def v(x, q, r1):
    return -np.pi/(2*q*r1)*(1+np.abs(x)**(2*q))*np.abs(x)**(1-q)*np.sign(x)

def vdv(x, q, r1):
    return -(np.pi/(2*q*r1))**2*(1+np.abs(x)**(2*q))*((1+q)*np.abs(x)**(2*q)+1-q)*np.abs(x)**(1-2*q)

def system(t, x):
    # Controller parameters
    r1, r2, q = 0.5, 1, 0.2
    # Disturbance
    Delta = np.sin(2*np.pi*t/0.2)
    # State variables
    x1, x2 = x[0], x[1]
    # Sliding variable
    s = x2-v(x1, q, r1)
    sd = x2+sol.odd_pow(sol.odd_pow(x2,2)-2*sol.odd_pow(v(x1, q, r1), 2), 0.5)
    # Controller
    u = v(sd, q, r1) - r2*np.sign(sd) - s + 2*vdv(x1, q, r1)*np.sign(sd)
    return np.array([x2, u+Delta])

# Simulation parameters
t0, tf, h, i = 0, 1.2, 1e-5, 0
# Simulation
for x0 in np.logspace(0, 3, 4):
    t, x = sol.ode1(system, np.array([x0, 0]), t0, tf, h)
    # States
    x1, x2 = x
#    # Disturbance
#    Delta = np.sin(0.5*np.pi*t)
#    # Controller parameters
#    r1, r2, q = 0.5, 1, 0.2
#    # Sliding variable
#    s = x2-v(x1, q, r1)
#    sd = x2+sol.odd_pow(sol.odd_pow(x2,2)-2*sol.odd_pow(v(x1, q, r1), 2), 0.5)
#    # Controller
#    u = v(sd, q, r1) - r2*np.sign(sd) - s + 2*vdv(x1, q, r1)*np.sign(sd)
    
    # Trajectories
    plt.figure(num=1)
    plt.plot(t, x1, color=0*np.ones(3), lw=2)
    plt.plot(t, x2, '--', color=0.5*np.ones(3), lw=2)    
    
    #plt.savefig('figures/trajW'+str(i)+'.eps', bbox_inches='tight', format='eps', dpi=1500)

plt.figure(num=1)
plt.plot(t, x1, color=0*np.ones(3), lw=2, label='$x_1(t)$')
plt.plot(t, x2, '--', color=0.5*np.ones(3), lw=2, label='$x_2(t)$')  
plt.xlim(0, 1.2)
plt.ylim(-5, 5)
plt.xlabel('$t$', fontsize = 14)
plt.text(1, 2, '$T_c=1$')
plt.axvline(x = 1, ymin = -1, ymax = 2, linestyle='dashed', color = 0.6*np.ones(3))
plt.legend(loc='best')
plt.grid()
plt.savefig('figures/solutions.eps', bbox_inches='tight', format='eps', dpi=1500)


# Control    
#plt.figure(num=3)
#plt.plot(t, u, '--', color=0*np.ones(3))
#plt.ylim(-10,10)
#plt.xlim(0, 1.2)
#plt.xlabel('$t$', fontsize = 14)
#plt.ylabel('$u$', fontsize = 14)
#plt.grid()
#plt.savefig('figures/controllerW'+str(i)+'.eps', bbox_inches='tight', format='eps', dpi=1500)