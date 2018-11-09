#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 11:54:39 2018

@author: esteban
"""

def predefined(x, a, b, p, q, k):
    from scipy.special import gamma
    import solver as sol
    mp, mq = (1-k*p)/(q-p), (k*q-1)/(q-p)
    g = gamma(mp)*gamma(mq)/((a**k)*gamma(k)*(q-p))*(a/b)**mp
    return g*sol.odd_pow(a*sol.odd_pow(x, p)+b*sol.odd_pow(x, q), k)

def system(t, x):
    import solver as sol
    import numpy as np
    xref = np.array([np.cos(2*np.pi*t), np.sin(2*np.pi*t)])
    dxref = 2*np.pi*np.array([-np.sin(2*np.pi*t), np.cos(2*np.pi*t)])
    e = x-xref
    r1, r2, r3, r4 = 1, 0, 2*np.pi+1, 1e-3
    r5, r6, r7, r8, r9, r1, zeta = 1, 1, 0.9, 1.1, 1.2, 1, 1
    return -1/r1*predefined(e,r5,r6,r7,r8,r9)-r3*e/(np.linalg.norm(e)+r4)-dxref