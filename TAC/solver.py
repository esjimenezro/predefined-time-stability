#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 12:56:42 2018

@author: esteban
"""

# Euler integration algorithm
def ode1(func, x0, t0, tf, h):
    import numpy as np
    t = np.arange(t0, tf+h, h)
    x = np.zeros((np.size(x0), t.size))
    x[:, 0] = x0
    for i in range(1, t.size):
        x[:, i] = h * func(t[i-1], x[:, i-1]) + x[:, i-1]
    return t, x

# Vector power function
def vec_pow(x, q):
    import numpy as np
    if np.linalg.norm(x)==0:
        return np.zeros(np.size(x))
    else:
        return np.array(x)*np.linalg.norm(x)**(q-1)
    
# Vector power function
def odd_pow(x, q):
    import numpy as np
    return (np.abs(x)**q)*np.sign(x)