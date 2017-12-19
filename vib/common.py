#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.linalg import eig, norm, inv, solve
import math
import itertools

class color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def next_pow2(i):
    """
    Find the next power of two

    >>> int(next_pow2(5))
    8
    >>> int(next_pow2(250))
    256
    """
    # do not use NumPy here, math is much faster for single values
    exponent = math.ceil(math.log(i) / math.log(2))
    # the value: int(math.pow(2, exponent))
    return exponent

def factors(n):
    """Find the prime factorization of n

    Efficient implementation. Find the factorization by trial division, using
    the optimization of dividing only by two and the odd integers.

    An improvement on trial division by two and the odd numbers is wheel
    factorization, which uses a cyclic set of gaps between potential primes to
    greatly reduce the number of trial divisions. Here we use a 2,3,5-wheel

    Factoring wheels have the same O(sqrt(n)) time complexity as normal trial
    division, but will be two or three times faster in practice.

    >>> list(factors(90))
    [2, 3, 3, 5]
    """
    f = 2
    increments = itertools.chain([1,2,2], itertools.cycle([4,2,4,2,4,6,2,6]))
    for incr in increments:
        if f*f > n:
            break
        while n % f == 0:
            yield f
            n //= f
        f += incr
    if n > 1:
        yield n

def db(x):
    """relative value in dB

    TODO: Maybe x should be rescaled to ]0..1].?
    log10(0) = inf.
    """

    # dont nag if x=0
    with np.errstate(divide='ignore', invalid='ignore'):
        return 20*np.log10(np.abs(x**2))

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

def meanVar(Y, isnoise=False):
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

    W = []
    # weights. Only used if the signal is noisy and multiple periods are
    # used
    if p > 1 and isnoise:
        W = np.sum(np.abs(Y0)**2, axis=2)/(p-1)

    return Ymean, W

def frf_mkc(M, K, fmin, fmax, fres, C=None, idof=None, odof=None):
    """Compute the frequency response for a FEM model, given a range of
    frequencies.

    Parameters
    ----------
    M: array
        Mass matrix
    K: array
        Stiffness matrix
    C: array, optional
        Damping matrix
    fmin: float
        Minimum frequency used 
    fmax: float
        Maximum frequency used 
    fres: float
        Frequency resolution 
    idof: array[int], optional
        Array of in dofs/modes to use. If None, use all.
    odof: array[int], optional
        Array of out dofs/modes to use. If None, use all.

    Returns
    -------
    freq: ndarray
        The frequencies where H is calculated.
    H: ndarray, [idof, odof, len(freq)]
        The transfer function. H[0,0] gives H1 for DOF1, etc.

    Examples
    --------
    >>> M = np.array([[1, 0],
    ...               [0, 1]])
    >>> K = np.array([[2, -1],
    ...               [-1, 6]])
    >>> C = np.array([[0.3, -0.02],
    ...               [-0.02, 0.1]])
    >>> freq, H = frf_mkc(M, K,  C)
    """

    n, n = M.shape
    if C is None:
        C = np.zeros(M.shape)
    # in/out DOFs to use
    if idof is None:
        idof = np.arange(n)
    if odof is None:
        odof = np.arange(n)
    n1 = len(idof)
    n2 = len(odof)

    # Create state space system, A, B, C, D. D=0
    Z = np.zeros((n, n))
    I = np.eye(n)
    A = np.vstack((
        np.hstack((Z, I)),
        np.hstack((-solve(M, K, assume_a='pos'),
                   -solve(M, C, assume_a='pos')))))
    B = np.vstack((Z, inv(M)))
    C = np.hstack((I, Z))
    

    F = int(np.ceil((fmax-fmin) / fres))
    freq = np.linspace(fmin, fmax, F+1) # + F*fres

    mat = np.zeros((n1,n2,F+1), dtype=complex)
    for k in range(F+1):
        mat[...,k] = solve(((1j*2*np.pi*freq[k] * np.eye(2*n) - A)).T,
                           C[odof].T).T @ B[:,idof]

    # Map to right index.
    H = np.zeros((n1,n2,F+1), dtype=complex)
    for i in range(n2):
        il = odof[i]
        for j in range(n1):
            ic = odof[j]
            H[il,ic] = np.squeeze(mat[i,j,:]).T

    return freq, H

def modal_properties_MKC(M, K, C=None, neigs=6):
    """Calculate natural frequencies, damping ratios and mode shapes.

    If the dampind matrix C is none or if the damping is proportional,
    wd and zeta are None.

    Parameters
    ----------
    M: array
        Mass matrix
    K: array
        Stiffness matrix
    C: array
        Damping matrix
    neigs: int, optional
        Number of eigenvalues to calculate

    Returns
    -------


    Examples
    --------
    >>> M = np.array([[1, 0],
    ...               [0, 1]])
    >>> K = np.array([[2, -1],
    ...               [-1, 6]])
    >>> C = np.array([[0.3, -0.02],
    ...               [-0.02, 0.1]])
    >>> sd = modes_system(M, K, C)
    """

    # Damping is non-proportional, eigenvectors are complex.
    if (C is not None and not np.all(C == 0)):
        n = len(M)
        Z = np.zeros((n, n))
        I = np.eye(n)
        # creates state space matrices
        A = np.vstack([np.hstack([Z, I]),
                    np.hstack([-solve(M, K, assume_a='pos'),
                               -solve(M, C, assume_a='pos')])])
        C = np.hstack((I, Z))
        sd = modal_properties(A, C)
        return sd

    # Damping is proportional or zero, eigenvectors are real
    egval, egvec = eig(K,M)
    lda = np.real(egval)
    idx = np.argsort(lda)
    lda = lda[idx]
    # In Hz
    wn = np.sqrt(lda) / (2*np.pi)
    realmode = np.real(egvec.T[idx])
    # normalize realmode
    nmodes = realmode.shape[0]
    for i in range(nmodes):
        realmode[i] = realmode[i] / norm(realmode[i])
        if realmode[i,0] < 0:
            realmode[i] = -realmode[i]

    zeta = []
    cpxmode = []
    wd = []
    sd = {
        'wn': wn,
        'wd': wd,
        'zeta': zeta,
        'cpxmode': cpxmode,
        'realmode': realmode,
    }
    return sd

def modal_properties(A, C=None):
    """Calculate eigenvalues and modes from state space matrices A and C

    Parameters
    ----------


    Returns
    -------
    wn: real ndarray. (modes)
        Natural frequency (Hz)
    wd: real ndarray. (modes)
        Damped frequency (Hz)
    zeta: real ndarray. (modes)
        Damping factor
    cpxmode : complex ndarray. (modes, nodes)
        Complex mode(s) shape
    realmode : real ndarray. (nodes, nodes)
        Real part of cpxmode. Normalized to 1.
    sd : dict
        Keys are the names written above.
    """
    from copy import deepcopy
    egval, egvec = eig(A)
    lda = egval
    # throw away very small values. Note this only works for state-space
    # systems including damping. For undamped system, imag(lda) == 0!
    idx1 = np.where(np.imag(lda) > 1e-8)
    lda = lda[idx1]
    # sort after eigenvalues
    idx2 = np.argsort(np.imag(lda))
    lda = lda[idx2]
    wd = np.imag(lda) / (2*np.pi)
    wn = np.abs(lda) / (2*np.pi)

    # Definition: np.sqrt(1 - (freq/natfreq)**2)
    zeta = - np.real(lda) / np.abs(lda)

    # cannot calculate mode shapes if C is not given
    if C is not None:

        # Transpose so cpxmode has format: (modes, nodes)
        cpxmode = (C @ egvec).T
        cpxmode = cpxmode[idx1][idx2]
        # np.real returns a view. Thus scaling realmode, will also scale the part
        # of cpxmode that is part of the view (ie the real part)
        realmode = deepcopy(np.real(cpxmode))
    else:
        cpxmode = []
        realmode = egvec[idx1][idx2]

    # normalize realmode
    nmodes = realmode.shape[0]
    for i in range(nmodes):
        realmode[i] = realmode[i] / norm(realmode[i])
        if realmode[i,0] < 0:
            realmode[i] = -realmode[i]

    sd = {
        'wn': wn,
        'wd': wd,
        'zeta': zeta,
        'cpxmode': cpxmode,
        'realmode': realmode,
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
            den = norm(M1[i]) * norm(M2[j])
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
            num = (np.abs(np.vdot(M1[i],M2[j])) + np.abs(M1[i] @ M2[j]))**2
            den = np.real(np.vdot(M1[i],M1[i]) + np.abs(M1[i] @ M1[i])) * \
                  np.real(np.vdot(M2[j],M2[j]) + np.abs(M2[j] @ M2[j]))

            MACX[i,j] = num / den

    return MACX
