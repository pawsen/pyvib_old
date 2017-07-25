#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from scipy import signal
import numpy as np

def sineSweep(amp, fs, f1, f2, vsweep, inctype='lin', t0=0):
    """Do a linear or logarithmic sinus sweep excitation.

    For a reverse sweep, swap f1 and f2 and set a negative sweep rate.

    Parameters
    ----------
    amp : float
        Amplitude in N
    fs : float
        Sampling frequency
    f1 : float
        Starting frequency in Hz
    f2 : float
        Ending frequency in Hz
    vsweep : float
        Sweep rate in Hz/min
    inctype : str (optional)
        Type of increment. Linear or logarithmic: lin/log
    t0 : float (optional)
        Staring time, default t0=0
    """
    dt = 1/fs

    if inctype == 'log':
        tend = np.log2(f2 / f1 ) * (60/vsweep) + t0
    else:
        tend = (f2 - f1) / vsweep * 60 + t0

    # Because we want to enforce the given fs, arange is used instead of
    # linspace. This means that we might not include tend in t (which would be
    # the case with linspace), but for that we get the desired fs.
    ns = np.floor((tend-t0)*fs)
    t = np.arange(0,ns+1)/fs
    # t = np.linspace(t0, tend, ns +1)

    # Instantaneous frequency
    if inctype == 'log':
        finst = f1 * 2**(vsweep*((t - t0)/60))
    else:
        finst = f1 + vsweep/60*(t-t0)

    if inctype == 'log':
        psi = (2*np.pi * f1*60/(np.log(2)*vsweep)) * (2**(vsweep*((t-t0)/60)) - 1)
    else:
        psi = 2*np.pi *f1*(t-t0) + 2*np.pi*vsweep/60*(t-t0)**2 / 2

    u = amp * np.sin(psi)
    return u, t, finst



def randomPeriodic(arms, fs, f1, f2, ns, nrep=1 ):
    """ Do an random exitaion

    Parameters
    ----------
    arms : float
        Amplitude, rms in N
    fs : float
        Sampling frequency
    f1 : float
        Starting frequency in Hz
    f2 : float
        Ending frequency in Hz
    ns : int
        Number of points
    nrep : int
        Number of times the signal is repeated
    """

    dt = 1/fs

    # uniform distribution
    u = 2*np.random.rand(ns+1) -1
    u = u - np.mean(u)

    freq = np.linspace(0,fs, ns+1)

    # create desired frequency content
    U = np.fft.fft(u)
    U[ freq < f1] = 0
    U[ freq > f2] = 0


    u = np.real(np.fft.ifft(U)) * (ns+1)

    if nrep > 1:
        # repeat signal: 1 2 3 -> 1 2 3 1 2 3 1 2 3
        u = np.tile(u, nrep)
        # prevent the first number from reoccurring: 1 2 3 -> 1 2 3 2 3 2 3
        idx = np.arange(1,nrep) * (ns+1)
        u = np.delete(u, idx)
        t = np.arange(0, ns*nrep+1) / fs
    else:
        t = np.arange(0, ns+1) / fs


    rms = np.sqrt(np.mean(np.square(u)))
    u = arms * u / rms

    return u, t
