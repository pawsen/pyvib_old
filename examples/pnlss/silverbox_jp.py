#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pyvib.statespace import StateSpace as linss
from pyvib.statespace import Signal
from pyvib.pnlss import PNLSS
from pyvib.common import db
from scipy.linalg import norm
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

"""PNLSS model of the silverbox.

The Silverbox system can be seen as an electroninc implementation of the
Duffing oscilator. It is build as a 2nd order linear time-invariant system with
a 3rd degree polynomial static nonlinearity around it in feedback. This type of
dynamics are, for instance, often encountered in mechanical systems.
nonlinearity. The input-output data is synthetic.

See http://www.nonlinearbenchmark.org/#Silverbox
"""

# def load(var, amp):
#     fname = f"SNJP_{var}m_full_FNSI_{amp}.mat"
#     data = sio.loadmat(fname)
#     if var == 'u':
#         return (data[k] for k in ['um', 'fs', 'flines', 'P'])
#     else:
#         return data['ym']

def load(var, amp, k):
    path = 'data/'
    fname = f"{path}SNJP_{var}m_full_FNSI_{amp}.mat"
    return sio.loadmat(fname)[k]

# save figures to disk
savefig = True

P = load('u',100,'P').item()
lines = load('u',100,'flines').squeeze() - 1
fs = load('u',100,'fs').item()
u = load('u',100,'um')
y = load('y',100,'ym')
# u -= u.mean()
# y -= y.mean()

NT, m = u.shape
NT, p = y.shape
npp = NT/P

# odd multisine till 200 Hz, without DC.
lines = np.arange(0,2683,2)

# partitioning the data



#uest = np.repeat(uest[:,None],p,axis=1)
uest = uest.reshape(npp,m,R,P)
yest = yest.reshape(npp,p,R,P)

# Validation data.
u = np.delete(u, np.s_[:Nz+Ntr])
y = np.delete(y, np.s_[:Nz+Ntr])

uval = u[:npp]
yval = y[:npp]

# model orders and Subspace dimensioning parameter
n = 2
maxr = 20

sig = Signal(uest,yest,fs=1)
sig.lines(lines)
# average signal over periods. Used for training of PNLSS model
# Even if there's only 1 period, pnlss expect this to be run first. It also
# reshapes the signal, so um: (npp*m*R)
um, ym = sig.average()

linmodel = linss()
# estimate bla, total distortion, and noise distortion
linmodel.bla(sig)
models, infodict = linmodel.scan(n, maxr)

# estimate PNLSS
# transient: Add two periods before the start of each realization. Note that
# this for the signal averaged over periods
Ntr = Ptr*npp
T1 = np.r_[Ntr, 0]

model = PNLSS(linmodel.A, linmodel.B, linmodel.C, linmodel.D)
model.signal = sig
model.nlterms('x', [2,3], 'statesonly')
model.nlterms('y', [2], 'empty')
model.transient(T1)
model.optimize(weight=None, nmax=50)

# compute linear and nonlinear model output on training data
tlin, ylin, xlin = linmodel.simulate(um, T1=T1)
_, ynlin, _ = model.simulate(um)

# get best model on validation data. Change Transient settings, as there is
# only one realization
nl_errvec = model.extract_model(yval, uval, T1=Ntr)

# compute model output on test data(unseen data)
_, ylval, _ = linmodel.simulate(uval, T1=Ntr)
_, ynlval, _ = model.simulate(uval, T1=Ntr)

# compute model output on test data(unseen data)
_, yltest, _ = linmodel.simulate(utest, T1=0)
_, ynltest, _ = model.simulate(utest, T1=0)

## Plots ##
# store figure handle for saving the figures later
figs = {}

plt.ion()
# linear and nonlinear model error
resamp = 20
plt.figure()
plt.plot(ym[::resamp])
plt.plot(ym[::resamp]-ylin[::resamp])
plt.plot(ym[::resamp]-ynlin[::resamp])
plt.xlabel('Time index')
plt.ylabel('Output (errors)')
plt.legend(('output','linear error','PNLSS error'))
plt.title('Estimation results')
figs['estimation_error'] = (plt.gcf(), plt.gca())

# optimization path for PNLSS
plt.figure()
plt.plot(db(nl_errvec))
imin = np.argmin(nl_errvec)
plt.scatter(imin, db(nl_errvec[imin]))
plt.xlabel('Successful iteration number')
plt.ylabel('Validation error [dB]')
plt.title('Selection of the best model on a separate data set')
figs['pnlss_path'] = (plt.gcf(), plt.gca())

# result on validation data
plt.figure()
N = len(yval)
freq = np.arange(N)/N*fs
plottime = np.hstack((yval, yval-ylval, yval-ynlval))
plotfreq = np.fft.fft(plottime, axis=0)
nfd = plotfreq.shape[0]
plt.plot(freq[:nfd//2], db(plotfreq[:nfd//2]), '.')
plt.xlim((0, 300))
plt.xlabel('Frequency')
plt.ylabel('Output (errors) (dB)')
plt.legend(('Output','Linear error','PNLSS error'))
plt.title('Validation results')
figs['val_data'] = (plt.gcf(), plt.gca())

# result on test data
# resample factor, as there is 153000 points in test data
plt.figure()
N = len(ytest)
resamp = 1
freq = np.arange(N)/N*fs
plottime = np.hstack((ytest, ytest-yltest, ytest-ynltest))
plotfreq = np.fft.fft(plottime, axis=0)
nfd = plotfreq.shape[0]
plt.plot(freq[:nfd//2:resamp], db(plotfreq[:nfd//2:resamp]), '.')
plt.xlim((0, 300))
plt.xlabel('Frequency')
plt.ylabel('Output (errors) (dB)')
plt.legend(('Output','Linear error','PNLSS error'))
plt.title('Test results')
figs['test_data'] = (plt.gcf(), plt.gca())

# subspace plots
figs['subspace_optim'] = linmodel.plot_info()
figs['subspace_models'] = linmodel.plot_models()

if savefig:
    for k, fig in figs.items():
        fig = fig if isinstance(fig, list) else [fig]
        for i, f in enumerate(fig):
            f[0].tight_layout()
            f[0].savefig(f"boucwen_{k}{i}.pdf")
