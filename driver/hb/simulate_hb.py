#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
import numpy as np
from scipy import io

import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../../fem/hb/'))
sys.path.insert(1, os.path.join(sys.path[0], '../../'))
from hb import HB
from nlforce import NL_force, NL_polynomial
from hbplots import Anim, nonlin_frf, plots

plot = False # True


M0 = 1
C0 = 0.02
K0 = 1
M0, C0, K0 = np.atleast_2d(M0,C0,K0)
f_amp = 0.1
f0 = 1e-3
fdofs = 0

inl = np.array([[0,-1]])
enl = np.array([3])
knl = np.array([1])

par_hb ={
    'NH': 3,
    'npow2': 8,
    'nu': 1,
    'stability': True,
    'rcm_permute': False,
    'tol_NR': 1e-6,
    'max_it_NR': 15,
    'scale_x': 1,
    'scale_t': 1,
    'amp0':1e-4
}
par_cont = {
    'omega_cont_min': 1e-5*2*np.pi,
    'omega_cont_max': 0.5*2*np.pi,
    'cont_dir': 1,
    'opt_it_NR': 3,
    'step': 0.001,
    'step_min': 0.001,
    'step_max': 1,
    'angle_max_pred': 90,
    'it_cont_max': 1e6,
    'adaptive_stepsize': True
}


abspath='/home/paw/ownCloud/speciale/code/python/vib/'
relpath = 'data/T05_Data/'
path = abspath + relpath
mat =  io.loadmat(path + 'NLBeam.mat')
# M0 = mat['M']
# C0 = mat['C']
# K0 = mat['K']

## Force parameters
# location of harmonic force
# fdofs = 7
# f_amp = 3
# # Excitation frequency. lowest sine freq in Hz
# f0 = 25
# par_hb ={
#     'NH': 5,
#     'npow2': 9,
#     'nu': 1,
#     'stability': True,
#     'rcm_permute': False,
#     'tol_NR': 1e-6,
#     'max_it_NR': 15,
#     'scale_x': 5e-6, # == 5e-12
#     'scale_t': 3000
# }

# par_cont = {
#     'omega_cont_min': 25*2*np.pi,
#     'omega_cont_max': 40*2*np.pi,
#     'cont_dir': 1,
#     'opt_it_NR': 3,
#     'step': 0.1,
#     'step_min': 0.1,
#     'step_max': 20,
#     'angle_max_pred': 90,
#     'it_cont_max': 1e4,
#     'adaptive_stepsize': True
# }

# inl = np.array([[27,-1], [27,-1]])
# # inl = np.array([])
# enl = np.array([3,2])
# knl = np.array([8e9,-1.05e7])


nl_pol = NL_polynomial(inl, enl, knl)
nl = NL_force()
nl.add(nl_pol)
# machine precision for float
eps = np.finfo(float).eps


hb = HB(M0,C0,K0,nl, **par_hb)
omega, z, stab, B = hb.periodic(f0, f_amp, fdofs)
tp, omegap, zp, cnorm, c, cd, cdd = hb.get_components()

dof = 0
if plot:
    plots(tp, omegap, cnorm, c, cd, cdd, dof, B, inl, gtype='displ',
          savefig=False)
    plt.show()


hb.continuation(**par_cont)

nonlin_frf(hb, dof=0)



#plt.ioff()
