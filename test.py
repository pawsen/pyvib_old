#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import scipy.io
from rfs_plot import rfsPlotBuilder
from signal_cut import RFS



# directory = 'data/T03a_Data/'
# mat =  scipy.io.loadmat(directory + 'f16.mat')

# u = mat['u']
# ddy = mat['y'].T

# dofs = [66, 102]
# fs =400;
# ts = 0.0025

# rfs = RFS(ddy, fs, dofs)


# directory = 'data/'
# #mat = np.load(directory + 'duffing_time_inc.npz')
# mat = np.load(directory + 'duffing_inc.npz')

# ddy = mat['a']
# fs = mat['fs']

# rfs = RFS(ddy, fs, numeric=True)
# #rfs = RFS(y, fs,displ=True , numeric=True)


# rfs_plot = rfsPlotBuilder(rfs)



# Perodicity
from signal_cut import periodicity
from common import db
import frf
import matplotlib.pyplot as plt


directory = 'data/T03b_Data/'
forcing = '01'
forcing = '15'
umat = scipy.io.loadmat(directory + 'u_' + forcing + '.mat')
ymat = scipy.io.loadmat(directory + 'y_' + forcing + '.mat')['y']

# number of points pr period
nsper = 32768

ndof, _ = ymat.shape

# sample freq
fs = umat['fs'].item()
ido = 0
# periodicity(ymat, nsper, fs, ido)

# select periods to include
per = [3,4,5,6,7,8,9]
nper = len(per)

y = np.empty((ndof, nper*nsper))
u = np.empty(nper*nsper)

# extract periodic signal
for i, p in enumerate(per):
    y[:,i*nsper : (i+1)*nsper] = ymat[:, p*nsper : (p+1)*nsper]
    u[i*nsper : (i+1)*nsper] = umat['u'][p*nsper : (p+1)*nsper].squeeze()


fmin = 5
fmax = 500

for i in range(ndof):
    freq, H1, sigN = frf.periodic(u, y[i,:], nper, fs, fmin, fmax)

    if i == 0:
        H = np.empty((ndof, len(freq)), dtype=complex)
    H[i,:] = H1


H2 = H[ido,:]
# G2 = G[ido,:]

plt.figure()
plt.clf()
plt.plot(freq, np.angle(H2)/ np.pi * 180 )
plt.title('FRF for dof {}'.format(ido))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Phase angle (deg)')
#plt.ylim([-180, 180])
plt.yticks(np.linspace(-180,180,360/90))

# plt.figure()
# plt.clf()
# plt.plot(freq, G2 )
# plt.title('FRF for dof {}'.format(ido))
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Coherence')


plt.figure()
plt.clf()
plt.plot(freq, db(np.abs(H2)))
plt.title('FRF for dof {}'.format(ido))
plt.xlabel('Frequency (Hz)')
# For linear scale: 'Amplitude (m/N)'
plt.ylabel('Amplitude (dB)')
plt.show()
