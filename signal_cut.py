#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import scipy.io
import numpy as np
from scipy import linalg
import matplotlib.pylab as plt

def db(x):
    """relative value in dB

    TODO: Maybe x should be rescaled to ]0..1].?
    log10(0) = inf.
    """
    return 20*np.log10( np.abs(x **2 ))

def rescale(x):
    """Rescale data to range

    To 0..1:
    z_i = (x_i− min(x)) / (max(x)−min(x))

    Or custom range:
    a = (maxval-minval) / (max(x)-min(x))
    b = maxval - a * max(x)
    z = a * x + b

    """
    if hasattr(x, "__len__") is False:
        return x
    else:
        return (x - np.amin(x)) / (np.amax(x) - np.amin(x))



directory = 'T03b_Data/'
u = scipy.io.loadmat(directory + 'u_15.mat')
ymat = scipy.io.loadmat(directory + 'y_15.mat')['y']

# number of points pr period
nper = 32768

# sample freq
fs = u['fs'].item()
ts = 1/fs
t = np.arange(len(u['u']))*ts
# number of measurement/sensors
ndof = ymat.shape[0]
# total sample points
ns = ymat.shape[1]
# number of periods
nn = int(np.ceil(ns / nper))

# dof of shaker
id_force = u['iu'].item()

# dof of measurement to use
# ie. where is the nonlinearity
ido = 0
y = ymat[ido]

# y = np.array( [ np.arange(0,10)+np.random.normal(), np.arange(0,10)+np.random.normal()*0.01, np.arange(0,10)]).flatten()
# nper = 10
# ns = len(y)
# nn = int(np.ceil(ns / nper))

# first index of last period
ilast = ns - nper

# reference period. The last measured period
yref = y[ilast:ilast+nper]
yscale = np.amax(y) - np.amin(y)

# holds the similarity of the signal, compared to reference period
va =  np.empty(nper*(nn-1))

nnvec = np.arange(-nn,nn+1, dtype='int')
va_per = np.empty(nnvec.shape)
for i, n in enumerate(nnvec):
    # index of the current period. Moving from 1 to nn-1 because of if/break statement
    ist = ilast + n * nper
    if ist < 0:
        continue
    elif ist > ns - nper - 1:
        break

    idx = np.arange(ist,ist+nper)
    # difference between signals in dB
    va[idx] = db(y[idx] - yref)
    va_per[i] = np.amax(va[idx])

va2 = db(y)

plt.figure(1)
plt.clf()
plt.ion()
plt.title('Periodicity of signal for DOF {}'.format(ido))
plt.plot(t,va2, label='signal')  # , rasterized=True)
plt.plot(t[:ilast],va, label='periodicity')
for i in range(1,nn):
    x = t[nper * i]
    plt.axvline(x, color='k', linestyle='--')

plt.xlabel('Time (s)')
plt.ylabel(r'$\varepsilon$ dB')
plt.legend()
plt.show()

f = "periodicity"
plt.savefig(f + '.pdf', orientation='landscape')
plt.savefig(f + '.png', orientation='landscape')
