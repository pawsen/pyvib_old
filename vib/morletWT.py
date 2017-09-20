#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.linalg import fft, ifft
from common import next_pow2

def morletWT(x, fs, f0, f1, nf, f00, pad=0):
    """

    Parameters
    ----------
    x: ndarray
        Displacements (or velocities or accelerations)
    fs: float
        Sampling frequency
    """

    dt = 1/fs
    df = (f0 - f1) / nf
    freq = np.linspace(f0, f1, nf)

    a = f00 / (f0 + np.outer(np.arange(nf),df))
    na = len(a) - 1

    k = 2**pad
    NX = len(x)
    NX2 = next_pow2(NX)
    N = 2**NX2
    N = k*N

    time = np.arange(N)*dt
    f = np.linspace(0, fs/2, (N-2)//2)
    omega = f*2*np.pi

    filt = np.sqrt(2*a @ np.ones(N//2)) * \
        np.exp(-0.5*(a @ omega - 2*np.pi*f00)**2)
    filt[np.isnan(filt)] = 0

    X = fft(x, N, axis=1)
    X = np.conj(filt) * (np.ones((na+1,)) @ X[1::N/2])
    y = np.zeros((na+1,N))
    for j in range(na+1):
        y[j] = ifft(X[j], N)

    y = y.T
    mod = abs(y)

    imax = np.argmax(mod, axis=1)
    wtinst = np.max(mod, axis=1)
    finst = f00 / a[imax]

    finst = finst[NX]
    wtinst = wtinst[NX]
    y = y[NX]
    time = time[NX]

    return finst, wtinst, time, y, freq


import matplotlib.pyplot as plt
from common import db

def waveletPlot():

    finst, wtinst, time, y, freq = morletWT(x, fs, f0, f1, nf, f00, pad=0)

    # Some textture settings. Used to reduce the textture size, but not needed
    # for now.
    nmax = len(freq) if len(freq) > len(time) else len(time)

    plot_t = 'time'
    if plot_t == 'freq':
        vx = fss
        xstr = 'Frequency (Hz)'
    else:
        vx = t
        xstr = 'Time (s)'

    n1 = len(freq) // nmax
    n1 = 1 if n1 < 1 else n1
    n2 = len(vx) // nmax
    n2 = 1 if n2 < 1 else n2

    freq = freq[::n1]
    vx = vx[::n2]
    finst = finst[::n2]
    y = y[::n2,::n1]

    T, F = np.meshgrid(vx, freq)
    va = db(y)
    va[va < - 200] = -200

    fig, ax = plt.subplots(111)

    extends = ["neither", "both", "min", "max"]
    cmap = plt.cm.get_cmap("winter")
    cmap.set_under("magenta")
    cmap.set_over("yellow")

    cs = ax.contourf(T, F, va, 10, cmap=cmap, extend=extends[1])
    # Another way of setting invalid data, without specifying cmap
    cs.cmap.set_under('yellow')
    cs.cmap.set_over('cyan')

    cbar = fig.colorbar(cs, ax, shrink=0.9)
    cbar.ax.set_ylabel('verbosity coefficient')
    # Add the contour line levels to the colorbar
    cbar.add_lines(cs)

    ax.set_xlabel(xstr)
    ax.set_ylabel('Instantaneous frequency (Hz)')
