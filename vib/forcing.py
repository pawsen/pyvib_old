#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from numpy.fft import ifft

"""
Example of a closure-function. See also partial from ...
def force(A, f, ndof, fdof):
    # Closure function. See also functools.partial
    # fext = self.force(dt, tf=T)
    def wrapped_func(dt, t0=0, tf=1):
        ns = round((tf-t0)/dt)
        fs = 1/dt

        u,_ = sineForce(A, f=f, fs=fs, ns=ns, phi_f=0)
        fext = toMDOF(u, ndof, fdof)
        return fext
    return wrapped_func

"""
def sinesweep(amp, fs, f1, f2, vsweep, nrep=1, inctype='lin', t0=0):
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
    nrep : int
        Number of times the signal is repeated
    inctype : str (optional)
        Type of increment. Linear or logarithmic: lin/log
    t0 : float (optional)
        Staring time, default t0=0

    Notes
    -----
    See scipy.signal.chirp, which does the same
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.chirp.html
    """
    dt = 1/fs

    if inctype == 'log':
        tend = np.log2(f2 / f1) * (60/vsweep) + t0
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
        psi = 2*np.pi * f1*(t-t0) + 2*np.pi*vsweep/60*(t-t0)**2 / 2

    u = amp * np.sin(psi)
    if nrep > 1:
        # repeat signal: 1 2 3 -> 1 2 3 1 2 3 1 2 3
        u = np.tile(u, nrep)
        # prevent the first number from reoccurring: 1 2 3 -> 1 2 3 2 3 2 3
        idx = np.arange(1,nrep) * (ns+1)
        u = np.delete(u, idx)
        t = np.arange(0, ns*nrep+1) / fs

    return u, t, finst



def multisine(f1, f2, fs, N, P=1, M=1, ftype='full',rms=1, ngroup=3):
    """Random periodic excitation

    Generates a zero-mean random phase multisine with specified rms(amplitude).
    Random phase multisine signal is a periodic random signal with a
    user-controlled amplitude spectrum and a random phase spectrum drawn from a
    uniform distribution. If an integer number of periods is measured, the
    amplitude spectrum is perfectly realized, unlike classical Gaussian noise.
    Another advantage is that the periodic nature can help help separate signal
    from noise.

    The amplitude spectrum is flat between f1 and f2.

    Parameters
    ----------
    f1 : float
        Starting frequency in Hz
    f2 : float
        Ending frequency in Hz
    fs : float
        Sample frequency. Must be fs >= 2*f2
    N : int
        Number of points per period
    P  : int, optional
        Number of periods. default = 1
    M  : int, optional
        Number of realizations. default = 1
    ftype : str, {'full', 'odd', 'oddrandom'}, optional
        For characterization of NLs, only selected lines are excited.
    rms : float, optional
        rms(amplitude) of the generated signals. default = 1. Note that since
        the signal is zero-mean, the std and rms is equal.
    ngroup : int, optional
        In case of ftype = oddrandom, 1 out of ngroup odd lines is discarded.

    Returns
    -------
    u: MxNP record of the generated signals
    t: time vector
    lines: excited frequency lines -> 1 = dc, 2 = fs/N
    freq: frequency vector

    Notes
    -----
    J.Schoukens, M. Vaes, and R. Pintelon:
    Linear System Identification in a Nonlinear Setting:
    Nonparametric Analysis of the Nonlinear Distortions and Their Impact on the
    Best Linear Approximation. https://arxiv.org/pdf/1804.09587.pdf

    https://en.wikipedia.org/wiki/Root_mean_square#Relationship_to_other_statistics
    """
    if not fs >= 2*f2:
        raise AssertionError('fs should be fs >= 2*f2. fs={}, f2={}'.format(fs,f2))

    valid_ftype = {'full', 'odd', 'oddrandom'}
    # frequency resolution
    f0 = fs/N
    # lines selection - select which frequencies to excite
    lines_min = np.ceil(f1/f0).astype('int')
    lines_max = np.floor(f2/f0).astype('int')
    lines = np.arange(lines_min, lines_max + 1, dtype=int)

    # remove dc
    if lines[0] == 0:
        lines = lines[1:]

    if ftype == 'full':
        pass  # do nothing
    elif ftype == 'odd':
        # remove even lines
        if np.remainder(lines[0],2):  # lines[0] is even
            lines = lines[::2]
        else:
            lines = lines[1::2]
    elif ftype == 'oddrandom':
        if np.remainder(lines[0],2):
            lines = lines[::2]
        else:
            lines = lines[1::2]
        # remove 1 out of ngroup lines
        nlines = len(lines)
        nremove = np.floor(nlines/ngroup).astype('int')
        idx = np.random.randint(ngroup, size=nremove)
        idx = idx + ngroup*np.arange(nremove)
        lines = np.delete(lines, idx)
    else:
        raise ValueError('Invalid ftype {}'.format(repr(ftype)))

    nlines = len(lines)

    # multisine generation - frequency domain implementation
    U = np.zeros((M,N),dtype=complex)
    # excite the selected frequencies
    U[:,lines] = np.exp(2j*np.pi*np.random.rand(M,nlines))

    u = np.real(ifft(U,axis=1))  # go to time domain
    u = rms*u / np.std(u[0])  # rescale to obtain desired rms/std

    # Because the ifft is for [0,2*pi[, there is no need to remove any point
    # when the generated signal is repeated.
    u = np.tile(u,(1,P))  # generate P periods
    t = np.arange(N*P)/fs
    freq = np.linspace(0, fs, N)

    return u, t, lines, freq


def sineForce(A, f=None, omega=None, t=None, fs=None, ns=None, phi_f=0):
    """
    Parameters
    ----------
    A: float
        Amplitude in N
    f: float
        Forcing frequency in (Hz/s)
    t: ndarray
        Time array
    n: int
        Number of DOFs
    fdofs: int or ndarray of int
        DOF location of force(s)
    phi_f: float
        Phase in degree
    """

    if t is None:
        t = np.arange(ns)/fs
    if f is not None:
        omega = f * 2*np.pi

    phi = phi_f / 180 * np.pi
    u = A * np.sin(omega*t + phi)

    return u, t

def toMDOF(u, ndof, fdof):

    fdofs = np.atleast_1d(np.asarray(fdof))
    if any(fdofs) > ndof-1:
        raise ValueError('Some fdofs are greater than system size(n-1), {}>{}'.
                         format(fdofs, ndof-1))

    ns = len(u)
    f = np.zeros((ndof, ns))
    # add force to dofs
    for dof in fdofs:
        f[dof] = f[dof] + u

    return f
