#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
from matplotlib import pyplot as plt
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import filter as myfilter

saveplot = True
savedata =  True
forcing = '100'

realpath = os.path.dirname(os.path.realpath(sys.argv[0]))
# plot settings
plt.close('all')
plt.rc('text', usetex=True)  # enable LaTex in plots
plt.rc('font', family='times new roman')  # set font
#plt.rc('lines', markeredgewidth=1.)
plt.rc('lines', markersize=2)

# for report generation: http://wiki.scipy.org/Cookbook/Matplotlib/LaTeX_Examples
fig_width_pt = 422.52  # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0/72.27  # Convert pt to inches
golden_mean = (np.sqrt(5)-1.0)/2.0  # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt  # width in inches
fig_height = fig_width*golden_mean  # height in inches
fig_size = [fig_width, fig_height]
params = {'backend': 'pdf',
          'axes.labelsize': 10,
          'font.size': 12,
          'legend.fontsize': 10,
          'xtick.labelsize': 10,
          'ytick.labelsize': 10,
          'text.usetex': True,
          'figure.figsize': fig_size}
plt.rcParams.update(params)
# in order plot very large datasets, this needs to be set
plt.rcParams['agg.path.chunksize'] = 10000

## import processed data ##
path = realpath + '/../data/duffing_stepped' + forcing
data_inc = np.load(path + '_inc.npz')
data_dec = np.load(path + '_dec.npz')

def tohz(y):
    return y/(2*np.pi)

def steady_amp(data):
    # compute steady state amplitudes
    y = data['y']
    steady_idx = data['steady_idx']
    sweep_idx = data['sweep_idx']
    omegas = data['OMEGA_vec']
    A = []
    for i in range(len(omegas)):
        idx1 = steady_idx[i]
        idx2 = sweep_idx[i]
        ymax = np.max(y[idx1:idx2])
        ymin = np.min(y[idx1:idx2])
        A.append(np.abs(0.5*(ymax-ymin)))
    return A

A_inc = steady_amp(data_inc)
A_dec = steady_amp(data_dec)

# plot amplitude against omega. Ie. FRF
fig1 = plt.figure()
plt.clf()
plt.plot(tohz(data_inc['OMEGA_vec']), A_inc, 'kx',
          label=r'$\Delta \Omega>0$')
plt.plot(tohz(data_dec['OMEGA_vec']), A_dec, 'ro', mfc='none',
          label=r'$\Delta \Omega<0$')

plt.legend()
plt.title(r'FRF for stepped sine')
plt.xlabel(r'$\Omega$ (Hz)')
plt.ylabel(r'$|$Displacement$|$ (m)')
plt.grid(True)

inctype = 'inc'
if inctype == 'inc':
    data = data_inc
else:
    data = data_dec
y = data['y']
t = data['t']
OMEGAvec = data['OMEGA_vec']
sweep_idx = data['sweep_idx']
fs = data['fs']

def tick_function(X):
    return ["%.2f" % z for z in X]

def normalize(x, mini=None, maxi=None):
    """Rescale to 0-1.
    If mini and maxi is given, then they are used as the values that get scaled
    to 0 and 1, respectively
    """
    if mini is None:
        mini = np.min(x)
    if maxi is None:
        maxi = np.max(x)
    return (x - mini) / (maxi - mini)

def steady_time(data):
    # Find the steady state oscillations
    y_data = data['y']
    steady_idx = data['steady_idx']
    sweep_idx = data['sweep_idx']
    omegas = data['OMEGA_vec']
    y = []
    for i in range(len(omegas)):
        idx1 = steady_idx[i]
        idx2 = sweep_idx[i]
        y.extend(y_data[idx1:idx2])
    return y

# Plot 7 Omegas on the second x-axis.
step = len(OMEGAvec)//7
# normalize index of sweep. And fix that the index is actually one ahead. TODO:
# Not working
sweep_idx2 = np.insert(sweep_idx[:-1],0,0)
print(sweep_idx2)
idx_norm =normalize(sweep_idx, 0 ,len(y))
idx_norm2 =normalize(sweep_idx2, 0 ,len(y))
idx_norm2 = idx_norm

# Show only steady state
ys = steady_time(data)
ts = np.arange(len(ys))/fs

# TODO: OMEGAvec tickmarks are only correct for steady state plot. They are
# off(too early) for the full time plot. Ma

fig2 = plt.figure()
ax1 = fig2.add_subplot(111)
ax1.plot(t, y, '-k', label=r'Complete response', rasterized=True)
ax1.legend()
ax1.set_xlabel(r"Time (t)")
ax1.set_ylabel(r"Amplitude (m)")
ax1.set_xlim(left=t[0], right=t[-1])

ax2 = ax1.twiny()
ax2.set_xticks(idx_norm2[::step]) # location of ticks in interval [0,1]
ax2.set_xticklabels(tick_function(tohz(OMEGAvec[+1::step])))
ax2.set_xlabel(r"Excitation frequency $\Omega$ (Hz)")

# plot only steady state
fig3 = plt.figure()
ax1 = fig3.add_subplot(111)
ax1.plot(ts, ys, '-k', label=r'Steady state', rasterized=True)
ax1.legend()
ax1.set_xlabel(r"Time (t)")
ax1.set_ylabel(r"Amplitude (A)")
ax1.set_xlim(left=ts[0], right=ts[-1])

ax2 = ax1.twiny()
ax2.set_xticks(idx_norm[::step])
ax2.set_xticklabels(tick_function(tohz(OMEGAvec[+1::step])))
ax2.set_xlabel(r"Excitation frequency $\Omega (Hz)$")

if saveplot:
    relpath = '/../plots/duffing_stepped' + forcing + '_' + inctype
    path = realpath + relpath
    fig1.savefig(path + 'frf.pdf')
    fig1.savefig(path + 'frf.png')
    fig2.savefig(path + 'time.pdf', dpi=900)
    fig2.savefig(path + 'time.png')
    fig3.savefig(path + 'steady_time.pdf', dpi=900)
    fig3.savefig(path + 'steady_time.png')
    print('Figures saved in {}'.format(relpath))

if savedata:
    relpath = '/../data/duffing_stepped_post' + forcing + '_' + inctype
    filename = realpath + relpath
    np.savez(
        filename,
        fs=fs,
        t=t,
        y=y,
        ts_res=ts,
        ys_res=ys
    )
    print('Data saved as {}'.format(relpath))


plt.show()
