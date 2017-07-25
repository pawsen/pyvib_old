#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
from matplotlib import pyplot as plt
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import filter as myfilter

saveplot = False
saveplot = True
savedata =  True

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
path = realpath + '/../data/'
data_inc = np.load(path + 'duffing_inc.npz')
data_dec = np.load(path + 'duffing_dec.npz')

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

# plt.ion()  # turn on interactive mode
# plot amplitude against omega
fig1 = plt.figure()
plt.clf()

plt.plot(data_inc['OMEGA_vec'], A_inc, 'o', mfc='none',
          label=r'$\Delta \Omega>0$') # hollow circles
plt.plot(data_dec['OMEGA_vec'], A_dec, 'rx',
          label=r'$\Delta \Omega<0$')

plt.legend()
plt.xlabel(r'$\Omega$')
plt.ylabel(r'$a$')
plt.grid(True)
# plt.show()

# Resample data to > 3*omega to capture third harmonic. Cutoff below nyquist
# freq for resample fs.
y = data_inc['y']
t = data_inc['t']
OMEGAvec = data_inc['OMEGA_vec']
sweep_idx = data_inc['sweep_idx']
fs = data_inc['fs']
fs_out = 11 # data_inc['omega2']*3
cutoff = 5

t_res, y_res = myfilter.resample(y,fs, fs_out, cutoff)

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
    # compute steady state amplitudes
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

step = len(OMEGAvec)//7
# normalize index of sweep
idx_norm =normalize(sweep_idx, 0 ,len(y))

# Show only steady state
ys = steady_time(data_inc)
ts, ys = myfilter.resample(ys,fs, fs_out, cutoff)


# plot resampled
fig2 = plt.figure()
ax1 = fig2.add_subplot(111)
# ax1.plot(t, y, '-k', rasterized=True)
ax1.plot(t_res, y_res, '-k', rasterized=True)

ax1.set_xlabel(r"Time (t)")
ax1.set_ylabel(r"Amplitude (A)")
ax1.set_xlim(left=t_res[0], right=t_res[-1])

ax2 = ax1.twiny()
ax2.set_xticks(idx_norm[::step]) # location of ticks in interval [0,1]
ax2.set_xticklabels(tick_function(OMEGAvec[::step]))
ax2.set_xlabel(r"Excitation frequency $\Omega$")


# plot only steady state
fig3 = plt.figure()
ax1 = fig3.add_subplot(111)
# ax1.plot(t, y, '-k', rasterized=True)
ax1.plot(ts, ys, '-k', rasterized=True)

ax1.set_xlabel(r"Time (t)")
ax1.set_ylabel(r"Amplitude (A)")
ax1.set_xlim(left=ts[0], right=ts[-1])

ax2 = ax1.twiny()
ax2.set_xticks(idx_norm[::step]) # location of ticks in interval [0,1]
ax2.set_xticklabels(tick_function(OMEGAvec[::step]))
ax2.set_xlabel(r"Excitation frequency $\Omega$")

# plt.show()

if saveplot:
    path = realpath + '/../plots/'
    fig1.savefig(path + 'duffing_response.pdf')
    fig1.savefig(path + 'duffing_response.png')
    fig2.savefig(path + 'duffing_time_all.pdf', dpi=1200)
    fig2.savefig(path + 'duffing_time_all.png')
    fig3.savefig(path + 'duffing_time.pdf', dpi=1200)
    fig3.savefig(path + 'duffing_time.png')

if savedata:
    filename = realpath + '/../data/' + 'duffing_time_inc.npz'
    np.savez(
        filename,
        fs=fs_out,
        t_res=t_res,
        y_res=y_res,
        ts_res=ts,
        ys_res=ys
    )

plt.show()
