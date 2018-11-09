#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 15:52:48 2018

@author: esteban
"""

def predefined(x, p, q, k, a, b, Tc):
    from scipy.special import gamma
    import solver as sol
    mp, mq = (1-k*p)/(q-p), (k*q-1)/(q-p)
    g = gamma(mp)*gamma(mq)/((a**k)*gamma(k)*(q-p))*(a/b)**mp
    return -g/Tc*sol.odd_pow(a*sol.odd_pow(x, p)+b*sol.odd_pow(x, q), k)

def system(t, x):
    import solver as sol
    import numpy as np
    Delta = np.sin(2*np.pi*t/5)
    p, q, k, a, b, Tc, zeta = 0.5, 1.1, 1.2, 4, 0.25, 1, 1
    return predefined(x,p,q,k,a,b,Tc)-zeta*sol.odd_pow(x, 0)+Delta

import numpy as np
import solver as sol
import matplotlib.pyplot as plt
import matplotlib as mpl
label_size = 16
mpl.rcParams['xtick.labelsize'] = label_size
mpl.rcParams['font.size'] = label_size

t0, tf, h,  i = 0, 1.2, 1e-5, 0
xx0 = np.logspace(-1, 5, 7)
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