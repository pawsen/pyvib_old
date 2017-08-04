#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
from scipy import io
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from signal2 import Signal
from fnsi import FNSI
from common import modal_properties
from helper.fnsi_plots import plot_modes, plot_knl, plot_linfrf, plot_stab


# abspath =  os.path.dirname(os.path.realpath(sys.argv[0]))
abspath='/home/paw/ownCloud/speciale/code/python/vib/'
relpath = 'data/T03b_Data/'
path = abspath + relpath

mat_u =  io.loadmat(path + 'u_15.mat')
mat_y =  io.loadmat(path + 'y_15.mat')
#mat_u =  io.loadmat(path + 'u_01.mat')
#mat_y =  io.loadmat(path + 'y_01.mat')

fs = mat_u['fs'].item() # 3000
fmin = mat_u['fmin'].item()  # 5
fmax = mat_u['fmax'].item()  # 500
iu = mat_u['iu'].item()  # 2 location of force
nper = mat_u['P'].item()
nsper = mat_u['N'].item()
u = mat_u['u'].squeeze()
y = mat_y['y']

signal = Signal(u, y, fs)
per = [4,5,6,7,8,9]
signal.cut(nsper, per)

isnoise = False

inl = np.array([[6,-1], [6,-1]])
#inl = np.array([[]])
enl = np.array([3,2,5])
knl = np.array([1,1,1])

# zero-based numbering of dof
# idof are selected dofs.
# iu are dofs of force
iu = iu-1
idof = [0,1,2,3,4,5,6]
# dof where nonlinearity is
nldof = []


# ims: matrix block order. At least n+1
# nmax: max model order for stabilisation diagram
# ncur: model order for erstimation
ims = 22
nmax = 20
ncur = 6
nlist = np.arange(2,nmax+3,2)

fnsi = FNSI(signal, inl, enl, knl, idof, fmin, fmax)
fnsi.calc_EY()
fnsi.svd_comp(ims)

sd = fnsi.stabilisation_diagram(nlist)

# Do identification
fnsi.id(ncur)
knl, H, He = fnsi.nl_coeff(iu,[6])

# calculate modal parameters
modal = modal_properties(fnsi.A, fnsi.C)

plot_modes(idof, modal)
plot_knl(fnsi, knl)
plot_linfrf(fnsi, idof, H)
plot_stab(fnsi, nlist, sd)

plt.show()
