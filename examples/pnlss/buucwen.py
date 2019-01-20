#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pyvib.statespace import StateSpace as linss
from pyvib.statespace import Signal
from pyvib.pnlss import PNLSS
from pyvib.common import db
from pyvib.forcing import multisine
from scipy.linalg import norm
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

"""PNLSS model of BoucWen system with hysteresis acting as a dynamic
nonlinearity. The input-output data is synthetic.


See http://www.nonlinearbenchmark.org/#BoucWen
"""

# save figures to disk
savefig = True

data = sio.loadmat('BoucWenData.mat')
# partitioning the data
uval = data['uval_multisine'].T
yval = data['yval_multisine'].T
utest = data['uval_sinesweep'].T
ytest = data['yval_sinesweep'].T
u = data['u']
y = data['y']
lines = data['lines'].squeeze()
fs = data['fs'].item()

# model orders and Subspace dimensioning parameter
nvec = [2,3,4]
maxr = 7
#nvec = [3]
#maxr = 5

# create signal object
sig = Signal(u,y,fs=fs)
sig.lines(lines)
# average signal over periods. Used for training of PNLSS model
um, ym = sig.average()
npp, F = sig.npp, sig.F
R, P = sig.R, sig.P

linmodel = linss()
# estimate bla, total distortion, and noise distortion
linmodel.bla(sig)
# get best model on validation data
models, infodict = linmodel.scan(nvec, maxr)
l_errvec = linmodel.extract_model(yval, uval)

# estimate PNLSS
# transient: Add one period before the start of each realization. Note that
# this for the signal averaged over periods
T1 = np.r_[npp, np.r_[0:(R-1)*npp+1:npp]]

model = PNLSS(linmodel.A, linmodel.B, linmodel.C, linmodel.D)
model.signal = linmodel.signal
model.nlterms('x', [2,3], 'statesonly')
model.nlterms('y', [2,3], 'empty')
model.transient(T1)
model.optimize(lamb=100, weight=None, nmax=10)

# compute linear and nonlinear model output on training data
tlin, ylin, xlin = linmodel.simulate(um, T1=T1)
_, ynlin, _ = model.simulate(um)

# get best model on validation data. Change Transient settings, as there is
# only one realization
nl_errvec = model.extract_model(yval, uval, T1=npp)

# compute model output on test data(unseen data)
_, yltest, _ = linmodel.simulate(utest, T1=0)
_, ynltest, _ = model.simulate(utest, T1=0)

## Plots ##
# store figure handle for saving the figures later
figs = {}

# linear and nonlinear model error
plt.figure()
plt.plot(ym)
plt.plot(ym-ylin)
plt.plot(ym-ynlin)
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

# result on test data
plt.figure()
#  Normalized frequency vector
N = len(ytest)
freq = np.arange(N)/N*fs
plottime = np.hstack((ytest, ytest-yltest, ytest-ynltest))
plotfreq = np.fft.fft(plottime, axis=0)
nfd = plotfreq.shape[0]
plt.plot(freq[:nfd//2], db(plotfreq[:nfd//2]), '.')
plt.xlabel('Frequency (normalized)')
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
