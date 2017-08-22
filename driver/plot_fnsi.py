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

savefig = True
savefig = False
nonlin = True
nonlin = False

# abspath =  os.path.dirname(os.path.realpath(sys.argv[0]))
abspath='/home/paw/ownCloud/speciale/code/python/vib/'
relpath = 'data/T03b_Data/'
path = abspath + relpath

if nonlin:
    mat_u =  io.loadmat(path + 'u_15.mat')
    mat_y =  io.loadmat(path + 'y_15.mat')
else:
    mat_u =  io.loadmat(path + 'u_01.mat')
    mat_y =  io.loadmat(path + 'y_01.mat')

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
inl = np.array([[]])
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

natfreq = modal['natfreq']
nw = min(len(natfreq), 8)
print('Undamped ω: {}'.format(natfreq[:nw]))

fig_stab, ax1 = plt.subplots()
ax1.clear()
ax2 = ax1.twinx()
ax2.clear()

fig_modes = plot_modes(idof, modal)
figs_knl = plot_knl(fnsi, knl)
plot_linfrf(fnsi, idof, H, ax2)
plot_stab(fnsi, nlist, sd, ax1)

fig_frf, ax3 = plt.subplots()
plot_opt = {'ls':'--','label':'nonlin'}
plot_linfrf(fnsi, idof, H, ax3,**plot_opt)
plot_opt = {'label':'lin'}
#plot_linfrf(fnsi_lin, idof_lin, H_lin, ax3, **plot_opt)


fig_stab, *_ = plot_stab(fnsi, nlist, sd)

relpath = 'plots/fnsi/'
path = abspath + relpath
if savefig:
    if nonlin:

        str1 = 'nonlin' + str(len(inl))
    else:
        str1 = 'lin'
    fig_stab.savefig(path + 'stab_dia_' + str1 + '.png')
    fig_modes.savefig(path + 'modes_' + str1 + '.png')

    fig_frf.savefig(path + 'frf_' + str1 + '.png')

    if inl.size != 0:
        for i, fig_knl in enumerate(figs):
            exponent = enl[i]
            # plt.savefig(path + 'knl' + '{:d}_'.format(exponent) + str1 + '.png')
            fig_knl.savefig(path + 'knl' + '{:d}_'.format(exponent) + str1 + '.png')


#plt.ion()
plt.show()
