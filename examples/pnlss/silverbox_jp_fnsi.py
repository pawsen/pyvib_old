#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from pyvib.statespace import Signal
from pyvib.fnsi import FNSI
from pyvib.fnsi import NL_force, NL_polynomial, NL_spline
from pyvib.common import db
from scipy.linalg import norm
from pyvib.modal import modal_ac, frf_mkc
from pyvib.helper.modal_plotting import (plot_knl, plot_frf, plot_svg)
from pyvib.frf import periodic, covariance

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
savefig = False
savedata = False

# estimation data.
# 1 realization, 30 periods of 8192 samples. 5 discarded as transient
amp = 100
u, fs, lines, P = load('u',amp)
#lines = lines[1:]
y = load('y',amp)

NT, m = u.shape
NT, p = y.shape
npp = NT//P
R = 1
Ptr = 5

# odd multisine till 200 Hz, without DC.
# lines = np.arange(0,2683,2)

# partitioning the data
u = u.reshape(npp,m,R,P,order='F')[...,Ptr:]
y = y.reshape(npp,p,R,P,order='F')[...,Ptr:]
# FNSI can only use one realization
uest = u[:,:,0,:]
yest = y[:,:,0,:]

# TODO no validation data yet.
# \Users\JP\Documents\Research\Data\Silverbox_VUB_June2013\SilverboxData\CubicNL\BLA'
# um_full_100.mat
# until then, we just use last period of estimation data
uval = uest[...,-1].reshape(-1,m)
yval = yest[...,-1].reshape(-1,p)

# model orders and Subspace dimensioning parameter
n = 2
maxr = 20
dof = 0
iu = 0

xpowers = np.array([[2],[3]])
fnsi = FNSI(flines=lines, u=uest, y=yest, fs=fs, xpowers=xpowers)
fnsi.calc_EY()
fnsi.svd_comp(maxr)
# Do estimation
fnsi.id(n)
#fnsi.nl_coeff(iu, dofs=dof)

sca = 1
def print_modal(fnsi):
    # calculate modal parameters
    modal = modal_ac(fnsi.A, fnsi.C)
    natfreq = modal['wn']
    dampfreq = modal['wd']
    damping = modal['zeta']
    nw = min(len(natfreq), 8)

    print('Undamped ω: {}'.format(natfreq[:nw]*sca))
    print('damped ω: {}'.format(dampfreq[:nw]*sca))
    print('damping: {}'.format(damping[:nw]))
    return modal

print('## nonlinear identified at high level')
#modal = print_modal(fnsi)

# frf_freq, frf_H, covG, covGn = periodic(u,y, fs=fs, fmin=1e-3, fmax=150)

# plt.ion()
# fH1, ax = plt.subplots()
# plot_frf(frf_freq, frf_H, p=dof, sca=sca, ax=ax, ls='-', c='k', label='From high signal')
# fnsi.plot_frf(p=dof, sca=sca, ax=ax, label='nl high', ls='--', c='C1')

# fknl, axknl = plot_knl(fnsi, sca)


nnl = xpowers.shape[0]
E = np.zeros((n, nnl))
um = uest.mean(2)
ym = yest.mean(2)
# linear model simulation. But not really linear.
fnsi.E = E
tm, ylval, xm = fnsi.simulate(uval, T1=Ptr*npp)
_, ylin, _ = fnsi.simulate(um, T1=Ptr*npp)

# add nonlinearity
for i in range(nnl):
    E[:,i] = - fnsi.scaling[i]*fnsi.Bd[:,m+i]
fnsi.E = E

tm, ynlval, xm = fnsi.simulate(uval, T1=Ptr*npp)
_, ynlin, _ = fnsi.simulate(um, T1=Ptr*npp)

covY = covariance(yest[:,:,None])

## Plots ##
# store figure handle for saving the figures later
figs = {}

plt.ion()
# linear and nonlinear model error
resamp = 1
plt.figure()
plt.plot(ym[::resamp])
plt.plot(ym[::resamp]-ylin[::resamp])
plt.plot(ym[::resamp]-ynlin[::resamp])
plt.xlabel('Time index')
plt.ylabel('Output (errors)')
plt.legend(('output','linear error','FNSI error'))
plt.title('Estimation results')
figs['estimation_error'] = (plt.gcf(), plt.gca())

# result on validation data
plt.figure()
N = len(yval)
freq = np.arange(N)/N*fs
plottime = np.hstack((yval, yval-ylval, yval-ynlval))
plotfreq = np.fft.fft(plottime, axis=0)
nfd = plotfreq.shape[0]
plt.plot(freq[lines], db(plotfreq[lines]), '.')
plt.plot(freq[lines], db(np.sqrt((P-Ptr-1)*covY[lines].squeeze())), '.')
plt.xlim((0, 300))
plt.xlabel('Frequency')
plt.ylabel('Output (errors) (dB)')
plt.legend(('Output','Linear error','FNSI error', 'Noise'))
plt.title('Validation results')
figs['val_data'] = (plt.gcf(), plt.gca())

# subspace plots
#figs['subspace_optim'] = linmodel.plot_info()
#figs['subspace_models'] = linmodel.plot_models()

if savefig:
    for k, fig in figs.items():
        fig = fig if isinstance(fig, list) else [fig]
        for i, f in enumerate(fig):
            f[0].tight_layout()
            f[0].savefig(f"fig/silverbox_jp_fnsi_{k}{i}.pdf")

if savedata:
    pickle.dump(fnsi,open('sn_jp_fnsi.pkl', 'bw'))
