#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

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
def sineSweep(amp, fs, f1, f2, vsweep, nrep=1, inctype='lin', t0=0):
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



def randomPeriodic(arms, fs, f1, f2, ns, nrep=1):
    """Random periodic excitation

    Random phase multisine signal is a periodic random signal with a
    user-controlled amplitude spectrum and a random phase spectrum drawn from a
    uniform distribution. If an integer number of periods is measured, the
    amplitude spectrum is perfectly realized, unlike classical Gaussian noise.
    Another advantage is that the periodic nature can help help separate signal
    from noise.

    Here the amplitude spectrum is flat between f1 and f2.

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
        Number of points per sample
    nrep : int
        Number of times the signal is repeated

    Notes
    -----
    R. Pintelon and J. Schoukens. System Identification: A Frequency Domain
    Approach. IEEE Press, Piscataway, NJ, 2001
    """
    dt = 1/fs

    # uniform distribution
    u = 2*np.random.rand(ns+1) - 1
    u = u - np.mean(u)

    freq = np.linspace(0,fs, ns+1)

    # create desired frequency content, by modifying the phase.
    U = np.fft.fft(u)
    U[freq < f1] = 0
    U[freq > f2] = 0

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