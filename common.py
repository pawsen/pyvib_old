#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

def db(x):
    """relative value in dB

    TODO: Maybe x should be rescaled to ]0..1].?
    log10(0) = inf.
    """
    return 20*np.log10( np.abs(x **2 ))

def rescale(x, mini=None, maxi=None):
    """Rescale x to 0-1.

    If mini and maxi is given, then they are used as the values that get scaled
    to 0 and 1, respectively

    Notes
    -----
    To 0..1:
    z_i = (x_i− min(x)) / (max(x)−min(x))

    Or custom range:
    a = (maxval-minval) / (max(x)-min(x))
    b = maxval - a * max(x)
    z = a * x + b

    """
    if hasattr(x, "__len__") is False:
        return x

    if mini is None:
        mini = np.min(x)
    if maxi is None:
        maxi = np.max(x)
    return (x - mini) / (maxi - mini)

def meanVar(Y):
    """
    Y = fft(y)/nsper

    Parameters
    ----------
    Y : ndarray (ndof, nsper, nper)
        Y is the fft of y
    """

    # number of periods
    p = Y.shape[2]

    # average over periods
    Ymean = np.sum(Y,axis=2) / p

    # subtract column mean from y in a broadcast way. Ie: y is 3D matrix and
    # for every 2D slice we subtract y_mean. Python automatically
    # broadcast(repeat) y_mean.
    # https://scipy.github.io/old-wiki/pages/EricsBroadcastingDoc
    Y0 = Y - Ymean[...,None]

    # weights. Only used if the signal is noisy
    W = np.sum(np.abs(Y0)**2, axis=2)/( p-1)

    return Ymean, W

def modal_properties(A, C=None):
    """Calculate eigenvalues and modes from A and C

    Parameters
    ----------


    Returns
    -------
    cpxmode : complex ndarray. (modes, nodes)
        Complex mode(s) shape
    realmode : real ndarray. (nodes, nodes)
        Real part of cpxmode. Normalized to 1.
    natfreq : real ndarray. (modes)
        Natural frequency (Hz)
    freq : real ndarray. (modes)
        Damped frequency (Hz)
    ep : real ndarray. (modes)
        Damping factor
    sd : dict
        Keys are the names written above.
    """
    from copy import deepcopy
    from scipy import linalg
    egval, egvec = linalg.eig(A)
    lda = egval
    # throw away very small values
    idx1 = np.where(np.imag(lda) > 1e-8)
    lda = lda[idx1]
    # sort after eigenvalues
    idx2 = np.argsort(np.imag(lda))
    lda = lda[idx2]
    freq = np.imag(lda) / (2*np.pi)
    natfreq = np.abs(lda) / (2*np.pi)

    ep = -100* np.real(lda) / np.abs(lda)

    # TODO:
    # cannot calculate mode shapes if C is not given
    # if C is None:

    # Transpose so cpxmode has format: (modes, nodes)
    cpxmode = (C @ egvec).T
    cpxmode = cpxmode[idx1]
    cpxmode = cpxmode[idx2]
    # np.real returns a view. Thus scaling realmode, will also scale the part
    # of cpxmode that is part of the view (ie the real part)
    realmode = deepcopy(np.real(cpxmode))

    # normalize realmode
    nmodes = realmode.shape[0]
    for i in range(nmodes):
        realmode[i] = realmode[i] / linalg.norm(realmode[i])
        if realmode[i,0] < 0:
            realmode[i] =  -realmode[i]

    sd = {
        'cpxmode': cpxmode,
        'realmode': realmode,
        'freq': freq,
        'natfreq': natfreq,
        'ep': ep,
    }
    return sd

def ModalAC(M1, M2):
    """ Calculate MAC value for real valued mode shapes

    Returns
    -------
    MAC : float
        MAC value in range [0-1]. 1 is perfect fit.
    """
    if M1.ndim != 2:
        M1 = M1.reshape(-1,M1.shape[0])
    if M2.ndim != 2:
        M2 = M2.reshape(-1,M2.shape[0])

    nmodes1 = M1.shape[0]
    nmodes2 = M2.shape[0]

    MAC = np.zeros((nmodes1, nmodes2))

    for i in range(nmodes1):
        for j in range(nmodes2):
            num = M1[i].dot(M2[j])
            den = linalg.norm(M1[i]) * linalg.norm(M2[j])
            MAC[i,j] = (num/den)**2

    return MAC

def ModalACX(M1, M2):
    """Calculate MAC value for complex valued mode shapes

    M1 and M2 can be 1D arrays. Then they are recast to 2D.

    Parameters
    ----------
    M1 : ndarray (modes, nodes)
    M1 : ndarray (modes, nodes)

    Returns
    -------
    MACX : ndarray float (modes_m1, modes_m2)
        MAC value in range [0-1]. 1 is perfect fit.
    """
    if M1.ndim != 2:
        M1 = M1.reshape(-1,M1.shape[0])
    if M2.ndim != 2:
        M2 = M2.reshape(-1,M2.shape[0])

    nmodes1 = M1.shape[0]
    nmodes2 = M2.shape[0]

    MACX = np.zeros((nmodes1, nmodes2))
    for i in range(nmodes1):
        for j in range(nmodes2):
            num = (np.abs( np.vdot(M1[i],M2[j])) + np.abs(M1[i] @ M2[j]))**2
            den = np.real( np.vdot(M1[i],M1[i]) + np.abs(M1[i] @ M1[i]) ) * \
                  np.real( np.vdot(M2[j],M2[j]) + np.abs(M2[j] @ M2[j]) )

            MACX[i,j] = num / den

    return MACX
