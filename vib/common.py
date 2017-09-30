#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.linalg import eig, norm
import math

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
    buf = math.ceil(math.log(i) / math.log(2))
    return int(math.pow(2, buf))


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
    W = np.sum(np.abs(Y0)**2, axis=2)/(p-1)

    return Ymean, W

def frf_mck(M, C, K , fmin, fmax, fres):
    """Compute the transfer function for a FEM model.

    Returns
    -------
    freq: ndarray
        The frequencies where H is calculated.
    H: ndarray, [n, n, len(freq)]
        The transfer function. H[0,0] gives H1 for DOF1, etc.
    """
    from scipy.linalg import inv, lstsq

    n, n = M.shape
    if C is None:
        C = np.zeros(M.shape)
    # in/out DOFs to use
    idof = np.arange(n)
    odof = np.arange(n)
    n1 = len(idof)
    n2 = len(odof)

    # TODO: Use better state-space formulation
    iM = inv(M)
    A = np.vstack((
        np.hstack((np.zeros((n,n)),np.eye(n))),
        np.hstack((-iM @ K, -iM @ C))))
    B = np.vstack((np.zeros((n,n)), iM))
    C = np.hstack((np.eye(n), np.zeros((n,n))))

    N = int(np.ceil((fmax-fmin) / fres))
    freq = np.linspace(fmin, fmax, N+1) # + N*fres

    mat = np.zeros((n1,n2,N+1), dtype=complex)
    for k in range(N+1):
        sol = lstsq((( 1j *2*np.pi*freq[k] * np.eye(2*n) - A)).T, C[odof].T)[0]
        mat[...,k] = sol.T @ B[:,idof]

    # Map to right index.
    H = np.zeros((n1,n2,N+1), dtype=complex)
    for i in range(n2):
        il = odof[i]
        for j in range(n1):
            ic = odof[j]
            H[il,ic] = np.squeeze(mat[i,j,:]).T

    return freq, H

def undamp_modal_properties(M,K):
    """Calculate undamped modal properties"""
    egval, egvec = eig(K,b=M)
    egval = np.real(egval)
    idx = np.argsort(egval)
    w = egval[idx]
    f = w / (2*np.pi)
    X = egvec[idx]

    return w, f, X

def modal_properties_MKC(M, K, C=None):
    """Calculate damped and undamped eigenfrequencies in (rad/s).

    See Jakob S. Jensen & Niels Aage: "Lecture notes: FEMVIB", eq. 3.50 for the
    state space formulation of the EVP.

    Returns
    -------
    vals : float[neigs]
        The eigenvalues, each repeated according to its multiplicity. The
        eigenvalues are not necessarily ordered.
    vesc: float[neqs,neigs]
        the mass normalized (unit “length”) eigenvectors. The column
        vals[:,i] is the eigenvector corresponding to the eigenvalue
        vecs[i].

    Undamped frequencies can be found directly from (matlab syntax). Remove
    diag when using python.
    * [phi, w2] = eigs(K,M,6,'sm');
    * f = sort(sqrt(diag(w2))) / (2*pi)

    The generalized eigenvalue problem can be stated in different ways.
    scipy.sparse.linalg.eigs requires B to positive definite. A common
    formulation[1]:
    A = [C, K; -eye(size(K)), zeros(size(K))]
    B = [M, zeros(size(K)); zeros(size(K)), eye(size(K))]

    with corresponding eigenvector
    z = [lambda x; x]
    We can then take the first neqs components of z as the eigenvector x

    Note that the system coefficient matrices are always positive definite for
    elastic problems. Thus the formulation:
    A = [zeros(size(K)),K; K, C];
    B = [K, zeros(size(K)); zeros(size(K)), -M];
    does not work with scipy.sparse. But does work with scipy.linalg.eig. See
    also [2] and [3]

    [1]
    https://en.wikipedia.org/wiki/Quadratic_eigenvalue_problem
    [2]
    https://mail.python.org/pipermail/scipy-dev/2011-October/016670.html
    [3]
    https://scicomp.stackexchange.com/questions/10940/solving-a-generalised-eigenvalue-problem

    """
    from scipy.sparse.linalg import eigs
    from scipy.sparse import csr_matrix, vstack, hstack, identity
    from scipy.sparse import issparse
    from copy import deepcopy

    n,n = K.shape
    if C is None:
        A = K
        B = M
    else:
        if issparse(K):
            A = hstack([C, K], format='csr')
            A = vstack((A, hstack([-identity(n), csr_matrix((n,n))])), format='csr')
            B = hstack([M, csr_matrix((n,n))], format='csr')
            B = vstack((B, hstack([csr_matrix((n,n)), identity(n)])), format='csr')
        else:
            A = np.column_stack([C, K])
            A = np.row_stack((A, np.column_stack([-np.eye(n), np.zeros((n,n))])))
            B = np.column_stack([M, np.zeros(dim)])
            B = np.row_stack((B, np.column_stack([np.zeros((n,n)), np.eye(n)])))


    if n < 12 and not issparse(K):
        egval, egvec  = linalg.eig(A, b=B)

    else:
        egval, egvec = eigs(A, k=neigs, M=B, which='SR')
        if C is not None:
            # extract eigenvectors as the first half
            egvec = np.split(egvec, 2)[0]

    # TODO: dont repeat code! XXX. The following system can call modal_properties
    # dof = size( M, 1 );
    # A = [ zeros( dof, dof ), eye( dof ); - inv( M ) * K,  - inv( M ) * Cv ];
    # C = [ eye( dof ), zeros( dof, dof ) ];

    # Rest is copy from modal_properties below:
    lda = egval
    # throw away very small values
    idx1 = np.where(np.imag(lda) > 1e-8)
    lda = lda[idx1]
    # sort after eigenvalues
    idx2 = np.argsort(np.imag(lda))
    lda = lda[idx2]
    freq = np.imag(lda) / (2*np.pi)
    natfreq = np.abs(lda) / (2*np.pi)

    # Definition: np.sqrt(1 - (freq/natfreq)**2)
    ep = - np.real(lda) / np.abs(lda)

    # TODO: do some calculation/scaling of modes:
    cpxmode = []
    realmode = egvec[idx1][idx2]
    # normalize realmode
    nmodes = realmode.shape[0]
    for i in range(nmodes):
        realmode[i] = realmode[i] / norm(realmode[i])
        if realmode[i,0] < 0:
            realmode[i] = -realmode[i]

    sd = {
        'cpxmode': cpxmode,
        'realmode': realmode,
        'freq': freq,
        'natfreq': natfreq,
        'ep': ep,
    }
    return sd

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
    egval, egvec = eig(A)
    lda = egval
    # throw away very small values
    idx1 = np.where(np.imag(lda) > 1e-8)
    lda = lda[idx1]
    # sort after eigenvalues
    idx2 = np.argsort(np.imag(lda))
    lda = lda[idx2]
    freq = np.imag(lda) / (2*np.pi)
    natfreq = np.abs(lda) / (2*np.pi)

    # Definition: np.sqrt(1 - (freq/natfreq)**2)
    ep = - np.real(lda) / np.abs(lda)

    # cannot calculate mode shapes if C is not given
    if C is not None:

        # Transpose so cpxmode has format: (modes, nodes)
        cpxmode = (C @ egvec).T
        cpxmode = cpxmode[idx1]
        cpxmode = cpxmode[idx2]
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
