#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 16:47:17 2018

@author: esteban
"""

def system(t, x):
    from scipy.special import gamma
    import solver as sol
    import numpy as np
    
    x1, x2 = x[0], x[1]
    
    Delta = np.sin(2*np.pi*t/5)
    
    p, q, k, a2, b2, Tc2 = 0.5, 1.1, 1.2, 4, 0.25, 0.5
    mp, mq = (1-k*p)/(q-p), (k*q-1)/(q-p)
    a1, b1, Tc1, zeta = 4, 0.25, 0.5, 1
    
    g1 = gamma(0.25)*gamma(0.25)/((a1**0.5)*gamma(0.5)*2)*(a1/b1)**0.25
    g2 = gamma(mp)*gamma(mq)/((a2**k)*gamma(k)*(q-p))*(a2/b2)**mp
    
    s = x2 + sol.odd_pow(sol.odd_pow(x2,2)+
        ((g1/Tc1)**2)*(a1*sol.odd_pow(x1,1)+b1*sol.odd_pow(x1,3)),0.5)
    
    u = -(g2/Tc2*(a2*np.abs(s)**p+b2*np.abs(s)**q)**k+
         (g1**2/(2*Tc1**2))*(a1+3*b1*x1**2)+zeta)*np.sign(s)
    
    return np.array([x2, u+Delta])

import numpy as np
import solver as sol
import matplotlib.pyplot as plt
import matplotlib as mpl
label_size = 16
mpl.rcParams['xtick.labelsize'] = label_size
mpl.rcParams['font.size'] = label_size

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