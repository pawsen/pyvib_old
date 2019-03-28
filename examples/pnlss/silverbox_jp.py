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

Values from paper:
Estimated nonliner coefficients at different sampling rates
| fs (Hz) |     c1 |   c2 |
|---------+--------+------|
|    2441 | -0.256 | 3.98 |
|   12205 | -0.267 | 3.96 |

Identified at low level (5V)
| Nat freq (Hz) | Damping ratio (%) |
|---------------+-------------------|
|         68.58 |              4.68 |
"""

def load(var, amp, fnsi=True):
    fnsi = 'FNSI_' if fnsi else ''
    path = 'data/'
    fname = f"{path}SNJP_{var}m_full_{fnsi}{amp}.mat"
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
# 1 realization, 30 periods of 8192 samples. 5 discarded as transient (Ptr)
amp = 100
u, fs, lines, P = load('u',amp)
lines = lines[1:]
y = load('y',amp)

NT, R = u.shape
NT, R = y.shape
npp = NT//P
Ptr = 5
m = 1
p = 1

# partitioning the data
u = u.reshape(npp,P,R,order='F').swapaxes(1,2)[:,None,:,Ptr:-1]
y = y.reshape(npp,P,R,order='F').swapaxes(1,2)[:,None,:,Ptr:-1]
uest = u
yest = y
Pest = yest.shape[-1]

# Validation data. 50 different realizations of 3 periods. Use the last
# realization and last period
uval_raw, _, _, Puval = load('u', 100, fnsi=False)
yval_raw = load('y', 100, fnsi=False)
uval_raw = uval_raw.reshape(npp,Puval,50,order='F').swapaxes(1,2)[:,None]
yval_raw = yval_raw.reshape(npp,Puval,50,order='F').swapaxes(1,2)[:,None]
uval = uval_raw[:,:,-1,-1]
yval = yval_raw[:,:,-1,-1]
utest = uval_raw[:,:,1,-1]
ytest = yval_raw[:,:,1,-1]

sig = Signal(uest,yest, fs=fs)
sig.lines = lines
# estimate bla, total distortion, and noise distortion
sig.bla()
um, ym = sig.average()

# model orders and Subspace dimension parameter
n = 2
maxr = 20

# subspace model
linmodel = Subspace(sig)
# models, infodict = linmodel.scan(n, maxr, weight=False)
# ensure we use same dimension as for the fnsi model
linmodel.estimate(n,maxr)
linmodel2 = deepcopy(linmodel)
linmodel2.optimize(weight=False)

pnlss1 = PNLSS(linmodel)
pnlss1.nlterms('x', [2,3], 'statesonly')
pnlss1.nlterms('y', [2], 'empty')
pnlss1.transient(T1=npp)

pnlss2= deepcopy(pnlss1)
pnlss1.optimize(weight=False, nmax=200)
# optimized freq. weighted model
pnlss2.optimize(weight=True, nmax=200)

errvec1 = pnlss1.extract_model(yval, uval, T1=npp)
errvec2 = pnlss1.extract_model(yval, uval, T1=npp)

models = [linmodel, linmodel2, pnlss1, pnlss2]
descrip = ('Subspace','Subspace opt','pnlss', 'pnlss weight')
sca = 1
def print_modal(model):
    # calculate modal parameters
    modal = model.modal
    natfreq = modal['wn']
    dampfreq = modal['wd']
    damping = modal['zeta']
    nw = min(len(natfreq), 8)
    print('Undamped ω: {}'.format(natfreq[:nw]*sca))
    print('damped ω: {}'.format(dampfreq[:nw]*sca))
    print('damping: {}'.format(damping[:nw]))

for model, string in zip(models, descrip):
    print(f'### {string} identified at high level ###')
    modal = print_modal(model)

# add one transient period
Ptr2 = 1
# simulation error
est = np.empty((len(models),len(um)))
val = np.empty((len(models),len(uval)))
test = np.empty((len(models),len(utest)))
for i, model in enumerate(models):
    est[i] = model.simulate(um, T1=Ptr2*npp)[1].T
    val[i] = model.simulate(uval, T1=Ptr2*npp)[1].T
    test[i] = model.simulate(utest, T1=Ptr2*npp)[1].T

rms = lambda y: np.sqrt(np.mean(y**2, axis=0))
est_err = np.hstack((ym, (ym.T - est).T))
val_err = np.hstack((yval, (yval.T - val).T))
test_err = np.hstack((ytest, (ytest.T - test).T))
print(descrip)
print(f'rms error est:\n    {rms(est_err[:,1:])}\ndb: {db(rms(est_err[:,1:]))}')
print(f'rms error val:\n    {rms(val_err[:,1:])}\ndb: {db(rms(val_err[:,1:]))}')
print(f'rms error test:\n    {rms(test_err[:,1:])}\ndb: {db(rms(test_err[:,1:]))}')

# noise estimate over estimation periods
covY = covariance(yest)

## Plots ##
# store figure handle for saving the figures later
figs = {}

plt.ion()
# result on estimation data
resamp = 1
plt.figure()
plt.plot(est_err)
plt.xlabel('Time index')
plt.ylabel('Output (errors)')
plt.legend(('Output',) + descrip)
plt.title('Estimation results')
figs['estimation_error'] = (plt.gcf(), plt.gca())

# optimization path for PNLSS
plt.figure()
plt.plot(db(errvec1))
imin = np.argmin(errvec1)
plt.scatter(imin, db(errvec1[imin]))
plt.xlabel('Successful iteration number')
plt.ylabel('Validation error [dB]')
plt.title('Selection of the best model on a separate data set')
figs['pnlss_path'] = (plt.gcf(), plt.gca())

# optimization path for PNLSS
plt.figure()
plt.plot(db(errvec2))
imin = np.argmin(errvec2)
plt.scatter(imin, db(errvec2[imin]))
plt.xlabel('Successful iteration number')
plt.ylabel('Validation error [dB]')
plt.title('Selection of the best model on a separate data set')
figs['pnlss_weight_path'] = (plt.gcf(), plt.gca())

# result on validation data
plt.figure()
N = len(yval)
freq = np.arange(N)/N*fs
plottime = val_err
plotfreq = np.fft.fft(plottime, axis=0)/np.sqrt(N)
nfd = plotfreq.shape[0]
plt.plot(freq[lines], db(plotfreq[lines]), '.')
plt.plot(freq[lines], db(np.sqrt(Pest*covY[lines].squeeze() / N)), '.')
plt.ylim([-110,10])
plt.xlabel('Frequency')
plt.ylabel('Output (errors) (dB)')
plt.legend(('Output',) + descrip + ('Noise',))
plt.title('Validation results')
figs['val_data'] = (plt.gcf(), plt.gca())

if savefig:
    # subspace plots
    try:
        figs['subspace_optim'] = linmodel.plot_info()
        figs['subspace_models'] = linmodel.plot_models()
    except:
        pass

    for k, fig in figs.items():
        fig = fig if isinstance(fig, list) else [fig]
        for i, f in enumerate(fig):
            f[0].tight_layout()
            f[0].savefig(f"fig/silverbox_jp_{k}{i}.pdf")

if savedata:
    pickle.dump(models, open('sn_jp_pnlss.pkl', 'bw'))
    # with open('sn_jp_pnlss.pkl', 'bw') as f:
    #     pickler = pickle.Pickler(f)
    #     pickler.dump(linmodel)
    #     pickler.dump(model)
