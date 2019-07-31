#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 17:31:23 2019

@author: esteban
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import solver as sol
import numpy as np

label_size = 14
mpl.rcParams['xtick.labelsize'] = label_size
mpl.rcParams['font.size'] = label_size
mpl.rcParams['agg.path.chunksize'] = 10000

def system(t, x):
    # Controller parameters
    a1, a2, b1, b2 = 128, 64, 128, 64
    k = 1.1
    # Disturbance
    Delta = np.sin(2 * np.pi * 5 * t)
    # State variables
    x1, x2 = x[0], x[1]
    # Sliding variable
    s = x2 + sol.odd_pow(sol.odd_pow(x2,2) + a1*x1 + b1 * sol.odd_pow(x1, 3), 0.5)
    # Controller
    u = -(a1 + 3 * b1 * x1**2 + 2 * k) / 2 * np.sign(s) - sol.odd_pow(a2*s + b2 * sol.odd_pow(s, 3), 0.5)
    return np.array([x2, u+Delta])

# Simulation parameters
t0, tf, h, i = 0, 1.2, 1e-5, 0

# Space to plot
fig, [ax1, ax2] = plt.subplots(nrows=1, ncols=2) 
fig.subplots_adjust(wspace=0.31)

# Simulation
t, x = sol.ode1(system, np.array([1000, 0]), t0, tf, h)
# States
x1, x2 = x
# Plot of the trajectories
ax1.plot(t, x1, color=0*np.ones(3), lw=2, label='$x_1(t)$')
ax1.plot(t, x2, '--', color=0.5*np.ones(3), lw=2, label='$x_2(t)$')  
ax1.set_xlim(0, 1.2)
ax1.set_ylim(-5, 5)
ax1.set_xlabel('$t$', fontsize = 14)
ax1.axvline(x = 1, ymin = -1, ymax = 2, linestyle='dashed', color = 0.6*np.ones(3))
ax1.legend(loc='best')
ax1.grid()

# Controller parameters
a1, a2, b1, b2 = 128, 64, 128, 64
k = 1.1
# Sliding variable
s = x2 + sol.odd_pow(sol.odd_pow(x2,2) + a1*x1 + b1 * sol.odd_pow(x1, 3), 0.5)
# Controller
u = -(a1 + 3 * b1 * x1**2 + 2 * k) / 2 * np.sign(s) - sol.odd_pow(a2*s + b2 * sol.odd_pow(s, 3), 0.5)
# Plot of the control
ax2.plot(t, u, color=0*np.ones(3), lw=2)
ax2.set_xlim(0, 1.2)
ax2.set_ylim(-100, 100)
ax2.set_xlabel('$t$', fontsize = 14)
ax2.grid()


fig.savefig('figures/poly_controller.eps', bbox_inches='tight', format='eps', dpi=1500)