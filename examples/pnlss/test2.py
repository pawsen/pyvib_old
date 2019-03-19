#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#from pyvib.statespace import StateSpace as linss
from pyvib.signal import Signal
from pyvib.subspace import Subspace
from pyvib.pnlss import PNLSS
from pyvib.common import db
from scipy.linalg import norm
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import pickle

"""PNLSS model of the silverbox system.

The Silverbox system can be seen as an electroninc implementation of the
Duffing oscilator. It is build as a 2nd order linear time-invariant system with
a 3rd degree polynomial static nonlinearity around it in feedback. This type of
dynamics are, for instance, often encountered in mechanical systems.
nonlinearity. The input-output data is synthetic.
See http://www.nonlinearbenchmark.org/#Silverbox

This code correspond to the article
Grey-box state-space identification of nonlinear mechanical vibrations
JP. Noël & J. Schoukens
http://dx.doi.org/10.1080/00207179.2017.1308557
"""

def load(var, amp):
    path = 'data/'
    fname = f"{path}SNJP_{var}m_full_FNSI_{amp}.mat"
    data = sio.loadmat(fname)
    if var == 'u':
        um, fs, flines, P = [data[k] for k in ['um', 'fs', 'flines', 'P']]
        return um, fs.item(), flines.squeeze(), P.item()
    else:
        return data['ym']

# save figures to disk
savefig = True
savedata = True

# estimation data.
# 1 realization, 30 periods of 8192 samples. 5 discarded as transient
amp = 100
u, fs, lines, P = load('u',amp)
y = load('y',amp)

NT, m = u.shape
NT, p = y.shape
npp = NT//P
R = 1
Ptr = 5

# odd multisine till 200 Hz, without DC.
# lines = np.arange(0,2683,2)

# partitioning the data
uest = u.reshape(npp,m,R,P,order='F')[...,Ptr:]
yest = y.reshape(npp,p,R,P,order='F')[...,Ptr:]
uval = uest[...,-1].reshape(-1,m)
yval = yest[...,-1].reshape(-1,p)

# model orders and Subspace dimensioning parameter
n = 2
maxr = 20

sig = Signal(uest,yest)
sig.lines(lines)
# average signal over periods. Used for training of PNLSS model
# Even if there's only 1 period, pnlss expect this to be run first. It also
# reshapes the signal, so um: (npp*m*R)
um, ym = sig.average()
sig.bla()


#linmodel = Subspace(sig)
# # estimate bla, total distortion, and noise distortion
#models, infodict = linmodel.scan(n, maxr, weight=False, nmax=3)


# estimate PNLSS
# transient: Add five periods before the start of each realization. Note that
# this for the signal averaged over periods. Here there is only 1 realization
Ntr = Ptr*npp
T1 = np.r_[Ntr, 0]

model = PNLSS(linmodel)
model.signal = sig
model.nlterms('x', [2,3], 'statesonly')
model.nlterms('y', [2], 'empty')
model.transient(T1)
model.optimize(weight=False, nmax=50)

# compute linear and nonlinear model output on training data
tlin, ylin, xlin = linmodel.simulate(um, T1=T1)
_, ynlin, _ = model.simulate(um)

# get best model on validation data. Change Transient settings, as there is
# only one realization
nl_errvec = model.extract_model(yval, uval, T1=Ntr)

# compute model output on validation data(unseen data)
_, ylval, _ = linmodel.simulate(uval, T1=Ntr)
_, ynlval, _ = model.simulate(uval, T1=Ntr)


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

# subspace plots
figs['subspace_optim'] = linmodel.plot_info()
figs['subspace_models'] = linmodel.plot_models()

if savefig:
    for k, fig in figs.items():
        fig = fig if isinstance(fig, list) else [fig]
        for i, f in enumerate(fig):
            f[0].tight_layout()
            f[0].savefig(f"fig/silverbox_jp_{k}{i}.pdf")

if savedata:
    with open('sn_jp_pnlss.pkl', 'bw') as f:
        pickler = pickle.Pickler(f)
        pickler.dump(linmodel)
        pickler.dump(model)
