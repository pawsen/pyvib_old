#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from pyvib.signal import Signal
from pyvib.fnsi import FNSI
from pyvib.common import db
from scipy.linalg import norm
from pyvib.modal import modal_ac, frf_mkc
from pyvib.helper.modal_plotting import (plot_knl, plot_frf, plot_svg)
from pyvib.frf import periodic, covariance

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import pickle
from copy import deepcopy

"""FNSI model of the silverbox system.

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
Ptr = 0
m = 1
p = 1
# odd multisine till 200 Hz, without DC.
# lines = np.arange(0,2683,2)

# partitioning the data
u = u.reshape(npp,P,R,order='F').swapaxes(1,2)[:,None,:,Ptr:-1]
y = y.reshape(npp,P,R,order='F').swapaxes(1,2)[:,None,:,Ptr:-1]
# FNSI can only use one realization
uest = u[:,:,0,:]
yest = y[:,:,0,:]
Pest = uest.shape[-1]

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
um, ym = sig.average()
# sig.periodicity()

# model orders and Subspace dimensioning parameter
n = 2
maxr = 20
dof = 0
iu = 0
xpowers = np.array([[2],[3]])

# Linear model
fnsi1 = FNSI(sig)
fnsi1.estimate(n,maxr)
fnsi1.nl_coeff(iu)  # not necessary

# initial nonlinear model
fnsi2 = FNSI(sig)
fnsi2.nlterms('state',xpowers)
fnsi2.estimate(n,maxr)
fnsi2.nl_coeff(iu)

# optimized model
fnsi3 = deepcopy(fnsi2)
fnsi3.transient(T1=npp)
fnsi3.optimize(weight=None, nmax=50, xtol=1e-16, ftol=1e-16, gtol=1e-16)
fnsi3.nl_coeff(iu)

# optimized freq. weighted model
fnsi4 = deepcopy(fnsi2)
fnsi4.transient(T1=npp)
fnsi4.optimize(weight=True, nmax=50, xtol=1e-16, ftol=1e-16, gtol=1e-16)
fnsi4.nl_coeff(iu)

models = [fnsi1, fnsi2, fnsi3, fnsi4]
descrip = ('linear','fnsi init','fnsi optim', 'fnsi weight')
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
    print('# Nonlinear coefficients #')
    print(model.knl_str)

# add one transient period
Ptr2 = 1
# simulation error
val = np.empty((len(models),len(uval)))
est = np.empty((len(models),len(um)))
for i, model in enumerate(models):
    val[i] = model.simulate(uval, T1=Ptr2*npp)[1].T
    est[i] = model.simulate(um, T1=Ptr2*npp)[1].T

rms = lambda y: np.sqrt(np.mean(y**2, axis=0))
est_err = np.hstack((ym, (ym.T - est).T))
val_err = np.hstack((yval, (yval.T - val).T))
print(f'rms error est:\n    {rms(est_err)}\ndb: {db(rms(est_err))}')
print(f'rms error val:\n    {rms(val_err)}\ndb: {db(rms(val_err))}')

# noise estimate over 25 periods
covY = covariance(yest[:,:,None])

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

# result on validation data
plt.figure()
N = len(yval)
freq = np.arange(N)/N*fs
plottime = val_err
plotfreq = np.fft.fft(plottime, axis=0)/np.sqrt(N)
nfd = plotfreq.shape[0]
plt.plot(freq[lines+1], db(plotfreq[lines+1]), '.')
plt.plot(freq[lines+1], db(np.sqrt(Pest*covY[lines+1].squeeze() / N)), '.')
#plt.plot(freq[lines], db(covY[lines].squeeze()), '.')
#plt.plot(freq[lines], db(covY[lines].squeeze()/np.sqrt(25)), '.')
#plt.plot(freq[lines], db(np.sqrt((25-1)*covY[lines].squeeze()))),
#plt.xlim((0, 300))
#plt.ylim([-60,45])
plt.xlabel('Frequency')
plt.ylabel('Output (errors) (dB)')
plt.legend(('Output',) + descrip + ('Noise',))
plt.title('Validation results')
figs['val_data'] = (plt.gcf(), plt.gca())

# subspace plots
#figs['subspace_optim'] = linmodel.plot_info()
#figs['subspace_models'] = linmodel.plot_models()

if savefig:
    figs['periodicity'] = sig.periodicity()
    for k, fig in figs.items():
        fig = fig if isinstance(fig, list) else [fig]
        for i, f in enumerate(fig):
            f[0].tight_layout()
            f[0].savefig(f"fig/silverbox_jp_fnsi_{k}{i}.pdf")

if savedata:
    pickle.dump(models,open('sn_jp_fnsi.pkl', 'bw'))
