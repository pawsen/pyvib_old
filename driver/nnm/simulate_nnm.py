#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from scipy import io

import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../../'))
sys.path.insert(1, os.path.join(sys.path[0], '../../fem'))
from nnm import NNM
from nlforce import NL_force, NL_polynomial

M = 1
C = 0
K = 1
k1 = 1

inl = np.array([[0,-1]])
enl = np.array([3])
knl = np.array([k1])
inl = np.atleast_2d(inl)
nl_pol = NL_polynomial(inl, enl, knl)
nl = NL_force()
nl.add(nl_pol)

M, C, K = np.atleast_2d(M, C, K)

par_nnm = {
    'omega_min': 1e-5*2*np.pi,
    'omega_max': 0.5*2*np.pi,
    'opt_it_NR': 3,
    'max_it_NR': 15,
    'tol_NR': 1e-6,
    'step': 0.01,
    'step_min': 1e-6,
    'step_max': 1,
    'scale': 1e-4,
    'angle_max_beta': 90,
    'adaptive_stepsize': True,
    'mode':0
}

nnm = NNM(M, C, K, nl, **par_nnm)

nnm.periodic()
nnm.continuation()

plt.show()
