#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 11:54:39 2018

@author: esteban
"""

# Libraries
import numpy as np
import solver as sol
import matplotlib as mpl
import matplotlib.pyplot as plt
from numpy.linalg import norm
from matplotlib.gridspec import GridSpec
# Numbers size in graphs
label_size = 14
mpl.rcParams['xtick.labelsize'] = label_size
mpl.rcParams['font.size'] = label_size
mpl.rcParams['agg.path.chunksize'] = 10000

# Predefined-time stabilizing function
def tau0(Sr, q, ga, r1):
    return -np.pi / (2 * q * r1) * (1 + ga**(2 * q) * norm(Sr, axis=0)**(4 * q)) * ga**(1 - q) * sol.vec_pow(Sr, 1 - 2 * q)

# System
def system(t, x):
    # Positions and velocities
    q1, q2, dq1, dq2 = x
    q = np.array([[q1], [q2]])
    dq = np.array([[dq1], [dq2]])
    
    # Robot parameters
    m1, m2, l1, l2 = 1.5, 0.7, 0.15, 0.07
    # Matrices
    H11 = l1**2 * (m1 + m2) + 2 * (l2**2 * m2 + l1 * l2 * m2 * np.cos(q2)) - l2**2 * m2
    H12 = l2**2 * m2 + l1 * l2 * m2 * np.cos(q2)
    H21 = l2**2 * m2 + l1 * l2 * m2 * np.cos(q2)
    H22 = l2**2 * m2
    h = l1 * l2 * m2 * np.sin(q2)
    C11 = -h * dq2
    C12 = -h * (dq1 + dq2)
    C21 = h * dq1
    C22 = 0
    H = np.array([[H11, H12],
                  [H21, H22]])
    C = np.array([[C11, C12],
                  [C21, C22]])
    g = np.array([[0], [0]])
    detH = H11 * H22 - H12 * H21
    invH = np.array([[H22, -H12],
                     [-H21, H11]]) / detH
    # Perturbations
    d = 0.2 * dq + 0.1 * np.tanh(1000 * dq)
    
    # Control parameters
    r0, r1, r2, r3 = 10, 1, 10, 0.3
    ga = 0.3
    qp = 0.1
    # Reference
    i = np.array([list(range(5))]).T
    qd = 4 / np.pi * (np.sin((2 * i + 1) * t) / (2 * i + 1)).sum(axis=0) * np.array([[1], [2]])
    dqd = 4 / np.pi * (np.cos((2 * i + 1) * t)).sum(axis=0) * np.array([[1], [2]])
    d2qd = -4 / np.pi * ((2 * i + 1) * np.sin((2 * i + 1) * t)).sum(axis=0) * np.array([[1], [2]])
    dqr = dqd - r0 * (q - qd)
    d2qr = d2qd - r0 * (dq - dqd)
    # Controller
    Sr = dq - dqr
    tau = tau0(Sr, qp, ga, r1) - r2 * Sr / (norm(Sr, axis=0) + r3)
    
    # Model
    d2q = invH.dot(tau + d - C.dot(dq) + g)
    
    # Term Yr
    Yr = H.dot(d2qr) + C.dot(dqr) + g - d
    
    return np.concatenate((dq, d2q), axis=0).T[0]

# Simulation
t0, tf, h = 0, 10, 1e-4
x0 = np.array([0, 0, 0, 0])

# Simulation
t, x = sol.ode1(system, x0, t0, tf, h)    

# Positions and velocities
q1, q2, dq1, dq2 = x
q = np.array([q1, q2])
dq = np.array([dq1, dq2])

# Robot parameters
m1, m2, l1, l2 = 1.5, 0.7, 0.15, 0.07
# Matrices
H11 = l1**2 * (m1 + m2) + 2 * (l2**2 * m2 + l1 * l2 * m2 * np.cos(q2)) - l2**2 * m2
H12 = l2**2 * m2 + l1 * l2 * m2 * np.cos(q2)
H21 = l2**2 * m2 + l1 * l2 * m2 * np.cos(q2)
H22 = l2**2 * m2
h = l1 * l2 * m2 * np.sin(q2)
C11 = -h * dq2
C12 = -h * (dq1 + dq2)
C21 = h * dq1
C22 = 0
# Perturbations
d = 0.2 * dq + 0.1 * np.tanh(1000 * dq)

# Control parameters
r0, r1, r2, r3 = 10, 1, 10, 0.3
ga = 0.3
qp = 0.1
# Reference
i = np.array([list(range(5))]).T
qd = 4 / np.pi * (np.sin((2 * i + 1) * t) / (2 * i + 1)).sum(axis=0) * np.array([[1], [2]])
dqd = 4 / np.pi * (np.cos((2 * i + 1) * t)).sum(axis=0) * np.array([[1], [2]])
d2qd = -4 / np.pi * ((2 * i + 1) * np.sin((2 * i + 1) * t)).sum(axis=0) * np.array([[1], [2]])
dqr = dqd - r0 * (q - qd)
d2qr = d2qd - r0 * (dq - dqd)
# Controller
Sr = dq - dqr
tau = tau0(Sr, qp, ga, r1) - r2 * Sr / (norm(Sr, axis=0) + r3)

# Term Yr
Yr = np.array([H11 * d2qr[0] + H12 * d2qr[1],
               H21 * d2qr[0] + H22 * d2qr[1]]) + \
     np.array([C11 * dqr[0] + C12 * dqr[1],
               C21 * dqr[0] + C22 * dqr[1]]) - d

# Norm of Sr
plt.figure(num=1)
plt.subplots_adjust(hspace=0.5)
gs1 = GridSpec(4, 1)
plt.subplot(gs1[:3,0])
plt.plot(t, norm(Sr, axis=0), lw = 2, color=0*np.ones(3))
plt.ylabel('$||S_r(t)||$')
plt.text(1.2, 6, r'$T_c=\rho_1=1$')
plt.axvline(x=1, ymin=0, ymax=15, linestyle='dashed', color = 0.6*np.ones(3))
plt.grid()

plt.subplot(gs1[3,0])
plt.plot(t, norm(Sr, axis=0), lw = 2, color=0*np.ones(3))
plt.axvline(x=1, ymin=0, ymax=15, linestyle='dashed', color = 0.6*np.ones(3))
plt.axhline(y=0.2, xmin=0, xmax=10, linestyle='dashed', color = 0.3*np.ones(3))
plt.text(2, 0.25, r'$b=0.2$')
plt.grid()
plt.xlabel('$t$')
plt.ylim(0, 0.5)
plt.savefig('figures/norm_sr.eps', bbox_inches='tight', format='eps', dpi=1500)

## Tracking
plt.figure(num=2)
plt.plot(t, qd[0], lw = 3, color=0.7*np.ones(3), label='$q_{d_1}(t)$')
plt.plot(t, qd[1], lw = 3, color=0.5*np.ones(3), label='$q_{d_2}(t)$')
plt.plot(t, q[0], lw = 1.5, linestyle='dashed', color=0.2*np.ones(3), label='$q_{1}(t)$')
plt.plot(t, q[1], lw = 1.5, linestyle='dashed', color=0.*np.ones(3), label='$q_{2}(t)$')
plt.text(1.2, 1.5, r'$T_c=\rho_1=1$')
plt.axvline(x=1, ymin=0, ymax=15, linestyle='dashed', color = 0.6*np.ones(3))
plt.xlabel('$t$')
plt.legend(loc='best')
plt.grid()
plt.savefig('figures/track.eps', bbox_inches='tight', format='eps', dpi=1500)

## Control
plt.figure(num=3)
plt.plot(t, tau[0], lw = 3, color=0.*np.ones(3), label=r'$\tau_1(t)$')
plt.plot(t, tau[1], lw = 3, color=0.5*np.ones(3), label=r'$\tau_2(t)$')
plt.xlabel('$t$')
plt.ylim(-5, 15)
plt.legend(loc='best')
plt.grid()
plt.savefig('figures/tau.eps', bbox_inches='tight', format='eps', dpi=1500)


