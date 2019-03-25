#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pickle
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from scipy.linalg import norm

from pyvib.common import db
from pyvib.fnsi import FNSI
from pyvib.frf import covariance, periodic
from pyvib.helper.modal_plotting import plot_frf, plot_knl, plot_svg
from pyvib.modal import frf_mkc, modal_ac
from pyvib.signal import Signal


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

RMS of validation data
| Unit |   Output |  Lin err | fnsi init | fnsi opt |
|------+----------+----------+-----------+----------|
| V    |     0.16 |     0.09 |     0.002 |    0.001 |
| db   | -15.9176 | -20.9151 |  -53.9794 |     -60. |

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
lines = lines[1:] - 1
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
# FNSI can only use one realization
uest = u[:,:,0,:]
yest = y[:,:,0,:]
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
fnsi1.nl_coeff(iu)

# initial nonlinear model
fnsi2 = FNSI(sig)
fnsi2.nlterms('state',xpowers)
fnsi2.estimate(n,maxr)
fnsi2.nl_coeff(iu)

# optimized models
fnsi3 = deepcopy(fnsi2)
fnsi4 = deepcopy(fnsi2)  # freq. weighted model
weights = (None, True)
for w, model in zip(weights,[fnsi3, fnsi4]):
    model.transient(T1=npp)
    model.optimize(weight=w, nmax=50, xtol=1e-16, ftol=1e-16, gtol=1e-16)
    model.nl_coeff(iu)

# select best model on validation data
errvec3 = fnsi3.extract_model(yval, uval, T1=npp)
errvec4 = fnsi4.extract_model(yval, uval, T1=npp)

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
    print_modal(model)
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
print(descrip)
print(f'rms error est:\n    {rms(est_err[:,1:])}\ndb: {db(rms(est_err[:,1:]))}')
print(f'rms error val:\n    {rms(val_err[:,1:])}\ndb: {db(rms(val_err[:,1:]))}')

# noise estimate over Pest periods
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
plt.plot(freq[lines], db(plotfreq[lines]), '.')
plt.plot(freq[lines], db(np.sqrt(Pest*covY[lines].squeeze() / N)), '.')
#plt.xlim((0, 300))
plt.ylim([-110,10])
plt.xlabel('Frequency')
plt.ylabel('Output (errors) (dB)')
plt.legend(('Output',) + descrip + ('Noise',))
plt.title('Validation results')
figs['val_data'] = (plt.gcf(), plt.gca())

# optimization path
plt.figure()
for err in [errvec3, errvec4]:
    plt.plot(db(err))
    imin = np.argmin(err)
    plt.scatter(imin, db(err[imin]))
plt.xlabel('Successful iteration number')
plt.ylabel('Validation error [dB]')
plt.title('Selection of the best model on a separate data set')
figs['fnsi_path'] = (plt.gcf(), plt.gca())
plt.legend(descrip[2:])

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
