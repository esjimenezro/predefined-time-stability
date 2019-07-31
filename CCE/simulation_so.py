#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 16:47:17 2018

@author: esteban
"""

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import matplotlib.pyplot as plt
import matplotlib as mpl
import solver as sol
import numpy as np

label_size = 14
mpl.rcParams['xtick.labelsize'] = label_size
mpl.rcParams['font.size'] = label_size
mpl.rcParams['agg.path.chunksize'] = 10000

def v(x, q, r1):
    return -np.pi / (2 * q * r1) * (1 + np.abs(x)**(2 * q)) * np.abs(x)**(1 - q) * np.sign(x)

def vdv(x, q, r1):
    return -(np.pi / (2 * q * r1))**2 * (1 + np.abs(x)**(2 * q)) * ((1 + q) * np.abs(x)**(2 * q) + 1 - q) * np.abs(x)**(1 - 2 * q)

def system(t, x):
    # Controller parameters
    r1, r2, q = 0.5, 2, 0.4
    # Disturbance
    Delta = np.sin(2 * np.pi * 5 * t)
    # State variables
    x1, x2 = x[0], x[1]
    # Sliding variable
    s = x2 - v(x1, q, r1)
    sd = x2 + sol.odd_pow(sol.odd_pow(x2,2) - 2 * sol.odd_pow(v(x1, q, r1), 2), 0.5)
    # Controller
    u = v(sd, q, r1) - r2 * np.sign(sd) - s + 2 * vdv(x1, q, r1) * np.sign(sd)
    return np.array([x2, u+Delta])

# Simulation parameters
t0, tf, h, i = 0, 1.2, 1e-5, 0

# Space to plot
fig, ax = plt.subplots() # create a new figure with a default 111 subplot
# Zoomed inset plot
axins = zoomed_inset_axes(ax, 20, loc='upper right')
x1, x2, y1, y2 = 0.99, 1.01, -0.05, 0.05 # specify the limits
axins.set_xlim(x1, x2) # apply the x-limits
axins.set_ylim(y1, y2) # apply the y-limits
#axins.xaxis.set_visible(False)
#axins.yaxis.set_visible(False)
mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

# Simulation
for x0 in np.logspace(0, 3, 4):
    t, x = sol.ode1(system, np.array([x0, x0]), t0, tf, h)
    # States
    x1, x2 = x
    
    # Trajectories
    ax.plot(t, x1, color=0*np.ones(3), lw=2)
    ax.plot(t, x2, '--', color=0.5*np.ones(3), lw=2) 

axins.plot(t, x1, color=0*np.ones(3), lw=2)
axins.plot(t, x2, '--', color=0.5*np.ones(3), lw=2)
axins.axvline(x = 1, ymin = -1, ymax = 2, linestyle='dashed', color = 0.6*np.ones(3))

ax.plot(t, x1, color=0*np.ones(3), lw=2, label='$x_1(t)$')
ax.plot(t, x2, '--', color=0.5*np.ones(3), lw=2, label='$x_2(t)$')  
ax.set_xlim(0, 1.2)
ax.set_ylim(-5, 5)
ax.set_xlabel('$t$', fontsize = 14)
ax.text(0.8, -2, r'$T_c=2\rho_1=1$')
ax.axvline(x = 1, ymin = -1, ymax = 2, linestyle='dashed', color = 0.6*np.ones(3))
ax.legend(loc='best')
ax.grid()

fig.savefig('figures/solutions.eps', bbox_inches='tight', format='eps', dpi=1500)