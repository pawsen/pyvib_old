#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from scipy.linalg import norm

from pyvib.common import db
from pyvib.frf import covariance
from pyvib.pnlss import PNLSS
from pyvib.signal import Signal
from pyvib.subspace import Subspace

"""PNLSS model of BoucWen system with hysteresis acting as a dynamic
nonlinearity. The input-output data is synthetic.

From the paper by J.P. Noel: https://arxiv.org/pdf/1610.09138.pdf
| Polynomial degree | RMS validation error (dB) | Number of parameters |
|-------------------+---------------------------+----------------------|
|                 2 |                    -85.32 |                   34 |
|               2-3 |                    -90.35 |                   64 |
|             2-3-4 |                    -90.03 |                  109 |
|           2-3-4-5 |                    -94.87 |                  172 |
|         2-3-4-5-6 |                    -94.85 |                  256 |
|       2-3-4-5-6-7 |                    -97.96 |                  364 |
|             3-5-7 |                    -98.32 |                  217 |

See http://www.nonlinearbenchmark.org/#BoucWen
"""

# save figures to disk
savefig = True
savedata = True

data = sio.loadmat('data/BoucWenData.mat')
# partitioning the data
uval = data['uval_multisine'].T
yval = data['yval_multisine'].T
utest = data['uval_sinesweep'].T
ytest = data['yval_sinesweep'].T
uest = data['u']
yest = data['y']
lines = data['lines'].squeeze()  # [:-1]
fs = data['fs'].item()
nfreq = len(lines)
npp, m, R, P = uest.shape

# noise estimate over estimation periods
covY = covariance(yest)
Pest = yest.shape[-1]

# model orders and Subspace dimensioning parameter
nvec = [2,3,4]
maxr = 7

sig = Signal(uest,yest,fs=fs)
sig.lines = lines
sig.bla()
# average signal over periods. Used for training of PNLSS model
um, ym = sig.average()

linmodel = Subspace(sig)
models, infodict = linmodel.scan(nvec, maxr, nmax=50, weight=True)
# set model manual, as in matlab program
# linmodel.extract_model()
linmodel.estimate(n=3,r=4)
print(f'linear model: n,r:{linmodel.n},{linmodel.r}.')
print(f'Weighted Cost/nfreq: {linmodel.cost(weight=True)/nfreq}')

# estimate PNLSS
# transient: Add one period before the start of each realization. Note that
# this is for the signal averaged over periods
T1 = np.r_[npp, np.r_[0:(R-1)*npp+1:npp]]

fnsi1 = PNLSS(linmodel)
fnsi1.transient(T1)

fnsi2 = deepcopy(fnsi1)
fnsi3 = deepcopy(fnsi1)
fnsi4 = deepcopy(fnsi1)
fnsi5 = deepcopy(fnsi1)
fnsi6 = deepcopy(fnsi1)
fnsi7 = deepcopy(fnsi1)

fnsi1.nlterms('x', [2], 'statesonly')
fnsi2.nlterms('x', [2,3], 'statesonly')
fnsi3.nlterms('x', [2,3,4], 'statesonly')
fnsi4.nlterms('x', [2,3,4,5], 'statesonly')
fnsi5.nlterms('x', [2,3,4,5,6], 'statesonly')
fnsi6.nlterms('x', [2,3,4,5,6,7], 'statesonly')
fnsi7.nlterms('x', [3,5,7], 'statesonly')

models = [fnsi1, fnsi2, fnsi3, fnsi4, fnsi5, fnsi6, fnsi7]
descrip = ('2','2-3','2-3-4','2-3-4-5','2-3-4-5-6','2-3-4-5-6-7','3-5-7')
opt_path = []
for desc, model in zip(descrip, models):
    #model.nlterms('y', [], 'empty')
    model.optimize(weight=True, nmax=100)

    # get best model on validation data. Change Transient settings, as there is
    # only one realization
    nl_errvec = model.extract_model(yval, uval, T1=npp)
    opt_path.append(nl_errvec)
    with open(f'boucwen_model_{desc}.pkl', 'bw') as f:
        pickler = pickle.Pickler(f)
        pickler.dump(model)
        pickler.dump(nl_errvec)


# add one transient period
Ptr2 = 1
# simulation error
est = np.empty((len(models),len(um)))
val = np.empty((len(models),len(uval)))
test = np.empty((len(models),len(utest)))
for i, model in enumerate(models):
    est[i] = model.simulate(um, T1=T1)[1].T
    val[i] = model.simulate(uval, T1=Ptr2*npp)[1].T
    test[i] = model.simulate(utest, T1=0)[1].T

rms = lambda y: np.sqrt(np.mean(y**2, axis=0))
est_err = np.hstack((ym, (ym.T - est).T))
val_err = np.hstack((yval, (yval.T - val).T))
test_err = np.hstack((ytest, (ytest.T - test).T))
print(descrip)
print(f'rms error est:\n    {rms(est_err[:,1:])}\ndb: {db(rms(est_err[:,1:]))}')
print(f'rms error val:\n    {rms(val_err[:,1:])}\ndb: {db(rms(val_err[:,1:]))}')
print(f'rms error test:\n    {rms(test_err[:,1:])}\ndb: {db(rms(test_err[:,1:]))}')

## Plots ##
# store figure handle for saving the figures later
figs = {}

plt.ion()
# linear and nonlinear model error
resamp = 20
plt.figure()
plt.plot(est_err[::resamp])
plt.xlabel('Time index')
plt.ylabel('Output (errors)')
plt.legend(('Output',) + descrip)
plt.title('Estimation results')
figs['estimation_error'] = (plt.gcf(), plt.gca())

# result on validation data
resamp = 1
plt.figure()
plottime = val_err
N = plottime.shape[0]
freq = np.arange(N)/N*fs
plotfreq = np.fft.fft(plottime, axis=0)/np.sqrt(N)
nfd = plotfreq.shape[0]
plt.plot(freq[:nfd//2:resamp], db(plotfreq[:nfd//2:resamp]), '.')
plt.plot(freq[:nfd//2:resamp], db(np.sqrt(Pest*covY[:nfd//2:resamp].squeeze() / N)), '.')
plt.xlim((5, 150))
plt.xlabel('Frequency')
plt.ylabel('Output (errors) (dB)')
plt.legend(('Output',) + descrip + ('Noise',))
plt.title('Test results')
figs['val_data'] = (plt.gcf(), plt.gca())

# result on test data
resamp = 30
plt.figure()
plottime = test_err
N = plottime.shape[0]
freq = np.arange(N)/N*fs
plotfreq = np.fft.fft(plottime, axis=0)/np.sqrt(N)
nfd = plotfreq.shape[0]
plt.plot(freq[:nfd//2:resamp], db(plotfreq[:nfd//2:resamp]), '.')
plt.plot(freq[:nfd//2:resamp], db(np.sqrt(Pest*covY[:nfd//2:resamp].squeeze() / N)), '.')
plt.xlim((5, 200))
plt.xlabel('Frequency')
plt.ylabel('Output (errors) (dB)')
plt.legend(('Output',) + descrip + ('Noise',))
plt.title('Test results')
figs['val_data'] = (plt.gcf(), plt.gca())

# optimization path for PNLSS
plt.figure()
for err in opt_path:
    plt.plot(db(err))
    imin = np.argmin(err)
    plt.scatter(imin, db(err[imin]))
plt.xlabel('Successful iteration number')
plt.ylabel('Validation error [dB]')
plt.legend(descrip)
plt.title('Selection of the best model on a separate data set')
figs['fnsi_path'] = (plt.gcf(), plt.gca())

# subspace plots
figs['subspace_optim'] = linmodel.plot_info()
figs['subspace_models'] = linmodel.plot_models()

if savefig:
    for k, fig in figs.items():
        fig = fig if isinstance(fig, list) else [fig]
        for i, f in enumerate(fig):
            f[0].tight_layout()
            f[0].savefig(f"boucwen_{k}{i}.pdf")

if savedata:
    with open('boucwen.pkl', 'bw') as f:
        pickler = pickle.Pickler(f)
        pickler.dump(models)
