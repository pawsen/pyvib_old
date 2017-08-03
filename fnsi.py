#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import scipy as sp
from scipy import linalg, io
from common import meanVar, db
from signal2 import Signal
from collections import defaultdict
# just for debugging
from pprint import pprint
from scipy.linalg import norm

import os
import sys
abspath =  os.path.dirname(os.path.realpath(sys.argv[0]))

def force_nl(x, inl, enl, knl, idof):
    """Calculate force contribution for polynomial nonlinear stiffness or
    damping, see eq(2)

    Parameters
    ----------
    x : ndarray (ndof, ns)
        displacement or velocity.
    inl : ndarray (nbln, 2)
        Matrix with the locations of the nonlinearities,
        ex: inl = np.array([[7,0],[7,0]])
    enl : ndarray
        List of exponents of nonlinearity
    knl : ndarray (nbln)
        Array with nonlinear coefficients. ex. [1,1]
    idof : ndarray
        Array with node mapping for x.

    Returns
    -------
    f_nl : ndarray (nbln, ns)
        Nonlinear force
    """

    inl = np.asarray(inl)

    # return empty array in case of no nonlinearities
    if inl.size == 0:
        return np.array([])

    nbln = inl.shape[0]
    nsper = x.shape[1]
    fnl = np.zeros((nbln, nsper))

    for j in range(nbln):
        # connected from
        i1 = inl[j,0]
        # conencted to
        i2 = inl[j,1]

        # Convert to the right index
        idx1 = np.where(i1==idof)
        x1 = x[idx1]

        # if connected to ground
        if i2 == -1:
            x2 = 0
        else:
            idx2 = np.where(i2==idof)
            x2 = x[idx2]
        x12 = x1 - x2
        # in case of even functional
        if (enl[j] % 2 == 0):
            x12 = np.abs(x12)

        fnl[j,:] = knl[j] * np.abs(x12)**enl[j] * np.sign(x12)

    return fnl


def calc_EY(signal, inl, enl, knl, iu, nldof, isnoise, idof):
    """Calculate FFT of the extended input vector e(t) and the measured output
    y.

    The concatenated extended input vector e(t), is e=[u(t), g(t)].T, see eq
    (5). (E is called the Extended input spectral matrix and used for forming
    Ei, eq. (12)). Notice that the stacking order is reversed here.
    u(t) is the input force and g(y(t),ẏ(t)) is the functional nonlinear force
    calculated from the specified polynomial nonlinearity, see eq.(2)

    Returns
    ------
    E : ndarray (complex)
        FFT of the concatenated extended input vector e(t)
    Y : ndarray (complex)
        FFT of y.

    Notes
    -----
    Method by J.P Noel. Described in article
    "Frequency-domain subspace identification for nonlinear mechanical systems"
    http://dx.doi.org/10.1016/j.ymssp.2013.06.034
    Equation numbers refers to this article
    """
    print('E and Y comp.')

    # nper : number of periods
    # nsper : number of samples per period
    # ns : total samples in periodic signal
    y = signal.y_per
    u = signal.u_per
    if u.ndim != 2:
         u = u.reshape(-1,u.shape[0])
    nper = signal.nper
    nsper = signal.nsper

    # if all arrays are int, then resulting array is also int. But if some are
    # empty, the resulting array is float
    idof = np.unique(np.hstack((idof, iu, nldof)))
    idof = idof.astype(int)
    y = y[idof,:]

    fdof, ns = u.shape
    ndof, ns = y.shape

    # some parameters. Dont know what..
    p1 = 0
    p2 = 0
    npp = nper - p1 - p2


    # TODO make C-order. Should just be swap nsper and npp
    u = np.reshape(u[:,p1*nsper:(nper-p2)*nsper], (fdof, nsper, npp), order='F')
    y = np.reshape(y[:,p1*nsper:(nper-p2)*nsper], (ndof, nsper, npp), order='F')

    U = np.fft.fft(u,axis=1) / np.sqrt(nsper)
    Y = np.fft.fft(y,axis=1) / np.sqrt(nsper)

    Umean, WU = meanVar(U)
    Ymean, WY = meanVar(Y)

    # Set weights to none, if the signal is not noisy
    if not isnoise:
        WY = None

    # In case of no nonlinearities
    if inl.size == 0:
        return Umean, Ymean, WY, []

    pprint(inl)
    ynl = y
    # average displacement
    ynl = np.sum(ynl, axis=2) / npp
    nl = force_nl(ynl, inl, enl, knl, idof)
    nnl = nl.shape[0]

    scaling = np.zeros(nnl)
    for j in range(nnl):
      scaling[j] = np.std(u[0,:]) / np.std(nl[j,:])
      nl[j,:] = nl[j,:] * scaling[j]

    NL = np.fft.fft(nl, axis=1) / np.sqrt(nsper)
    # concatenate to form extended input spectra matrix
    E = np.vstack((Umean, -NL))

    return E, Ymean, WY, scaling


def svd_comp(signal, E, Y, W, i, flines):
    """
    i: Number of matrix block rows. Determining the size of matrices used.
    W : some weights

    # flines : ndarray
    #     Vector of frequency lines where the nonlinear coefficients are
    #     computed.
    """
    print('FNSI analysis part 1')

    nper = signal.nper
    nsper = signal.nsper
    ns = nper * nsper
    fs = signal.fs

    # dimensions
    m = np.shape(E)[0]
    l = np.shape(Y)[0]
    F = len(flines)

    # z-transform variable
    # Remember that e^(a+ib) = e^(a)*e^(ib) = e^(a)*(cos(b) + i*sin(b))
    # i.e. the exponential of a imaginary number is complex.
    zvar = np.empty(flines.shape, dtype=complex)
    zvar.real = np.zeros(flines.shape)
    zvar.imag = 2 * np.pi * flines / nsper
    zvar = np.exp(zvar)


    # dz is an array containing powers of zvar. Eg. the scaling z's in eq. (10)
    # (ξ is not used, but dz relates to ξ)
    dz = np.zeros((i+1, F), dtype=complex)
    for j in range(i+1):
        dz[j,:] = zvar**j

    # 2:
    # Concatenate external forces and nonlinearities to form the extended input
    # spectra Ei
    # Initialize Ei and Yi
    Emat = np.empty((m * (i + 1), F), dtype=complex)
    Ymat = np.empty((l * (i + 1), F), dtype=complex)
    for j in range(F):
        # Implemented as the formulation eq. (10), not (11) and (12)
        Emat[:,j] = np.kron(dz[:,j], E[:, flines[j]])
        Ymat[:,j] = np.kron(dz[:,j], Y[:, flines[j]])
    print('dtype: {}'.format(Emat.dtype))

    # Emat is implicitly recreated as dtype: float64
    Emat = np.hstack([np.real(Emat), np.imag(Emat)])
    Ymat = np.hstack([np.real(Ymat), np.imag(Ymat)])
    print('dtype: {}'.format(Emat.dtype))

    # 3:
    # Compute the orthogonal projection P = Yi/Ei using QR-decomposition, eq. (20)
    print('QR decomposition')
    P = np.vstack([Emat, Ymat])
    R, = linalg.qr(P.T, mode='r')
    Rtr = R[:(i+1)*(m+l),:(i+1)*(m+l)].T
    R22 = Rtr[(i+1)*m:i*(m+l)+m,(i+1)*m:i*(m+l)+m]

    # Calculate weight CY from filter W if present.
    if W is None:
        CY = np.eye(l*i)
    else:
        Wmat = np.zeros((l*i,F))
        for j in range(F):
            Wmat[:,j] = np.sqrt(np.kron(dz[:i,j], W[:, flines[j]]))
        CY = np.real(Wmat @ Wmat.T)


    # full_matrices=False is equal to matlabs economy-size decomposition.
    # gesvd is the driver used in matlab,
    UCY, scy, _ = linalg.svd(CY, full_matrices=False, lapack_driver='gesvd')
    SCY = np.diag(scy)

    sqCY = UCY.dot(np.sqrt(SCY).dot(UCY.T))
    isqCY = UCY.dot(np.diag(np.diag(SCY)**(-0.5)).dot(UCY.T))
    print('nan value: ', np.argwhere(np.isnan(isqCY)))

    # 4:
    # Compute the SVD of P
    print('SV decomposition')

    Un, sn, _ = linalg.svd(isqCY.dot(R22), full_matrices=False,
                           lapack_driver='gesvd')
    Sn = np.diag(np.diag(sn))

    #plt.ion()
    plt.figure(1)
    plt.clf()
    plt.semilogy(Sn/np.sum(Sn),'sk', markersize=6)
    plt.xlabel('Singular value')
    plt.ylabel('Magnitude')
    #plt.show()

    return Un, Sn, Rtr, sqCY, m, l ,F

def fnsi_id(Un, Sn, Rtr, sqCY, ims, nmodel,  fs, l, m, bd_method=None):
    """Frequency-domain Nonlinear Subspace Identification (FNSI)
    """


    # 5:
    # Truncate Un and Sn based on the model order n. The model order can be
    # determined by inspecting the singular values in Sn or using stabilization
    # diagram.
    if not nmodel:
        print('System order not set.')
    else:
        U1 = Un[:,:nmodel]
        S1 = Sn[:nmodel]

    # 6:
    # Estimation of the extended observability matrix, Γi, eq (21)
    # Here np.diag(np.sqrt(S1)) creates an diagonal matrix from an array
    G = sqCY.dot(U1.dot(np.diag(np.sqrt(S1))))

    # 7:
    # Estimate A from eq(24) and C as the first block row of G.
    # Recompute G from A and C, eq(13). G plays a major role in determining B and D,
    # thus Noel suggest that G is recalculated

    A = linalg.pinv(G[:-l,:]).dot(G[l:,:])
    C = G[:l,:]

    G1 = np.empty(G.shape)
    G1[:l,:] = C
    for j in range(1,ims):
        G1[j*l:(j+1)*l,:] = C.dot(np.linalg.matrix_power(A,j))

    G = G1

    print('(B,D) estimation using no optimisation')
    # 8:
    # Estimate B and D
    ## Start of (B,D) estimation using no optimisation ##

    # R_U: Ei+1, R_Y: Yi+1
    R_U = Rtr[:m*(ims+1),:(m+l)*(ims+1)]
    R_Y = Rtr[m*(ims+1):(m+l)*(ims+1),:(m+l)*(ims+1)]

    # eq. 30
    G_inv = linalg.pinv(G)
    Q = np.vstack([
        G_inv.dot(np.hstack([np.zeros((l*ims,l)), np.eye(l*ims)]).dot(R_Y)),
        R_Y[:l,:]]) - \
        np.vstack([
            A,
            C]).dot(G_inv.dot(np.hstack([np.eye(l*ims), np.zeros((l*ims,l))]).dot(R_Y)))

    Rk = R_U

    # eq (34) with zeros matrix appended to the end. eq. L1,2 = [L1,2, zeros]
    L1 = np.hstack([A.dot(G_inv), np.zeros((nmodel,l))])
    L2 = np.hstack([C.dot(G_inv), np.zeros((l,l))])

    # The pseudo-inverse of G. eq (33), prepended with zero matrix.
    # eq. MM = [zeros, G_inv]
    MM = np.hstack([np.zeros((nmodel,l)), G_inv])

    # The reason for appending/prepending zeros in L and MM, is to easily form the
    # submatrices of N, given by eq. 40. Thus ML is equal to first row of N1
    ML = MM - L1

    # rhs multiplicator of eq (40)
    Z = np.vstack([
        np.hstack([np.eye(l), np.zeros((l,nmodel))]),
        np.hstack([np.zeros((l*ims,l)),G])
    ])

    # Assemble the kron_prod in eq. 44.
    for kk in range(ims+1):
        # Submatrices of N_k. Given by eq (40).
        # eg. N1 corresspond to first row, N2 to second row of the N_k's submatrices
        N1 = np.zeros((nmodel,l*(ims+1)))
        N2 = np.zeros((l,l*(ims+1)))

        N1[:, :l*(ims-kk+1)] = ML[:, kk*l:l*(ims+1)]
        N2[:, :l*(ims-kk)] = -L2[:, kk*l:l*ims]

        if kk == 0:
            # add the identity Matrix
            N2[:l, :l] = np.eye(l) + N2[:l, :l]

        # Evaluation of eq (40)
        Nk = np.vstack([
            N1,
            N2
        ]).dot(Z)


        if kk == 0:
            kron_prod = np.kron(Rk[kk*m:(kk+1)*m,:].T, Nk)
        else:
            kron_prod = kron_prod + np.kron(Rk[kk*m:(kk+1)*m,:].T, Nk)

    # like flatten, just faster. Flatten row wise.
    Q = Q.ravel(order='F')
    Q_real = np.hstack([
        np.real(Q),
        np.imag(Q)
    ])

    kron_prod_real = np.vstack([
        np.real(kron_prod),
        np.imag(kron_prod)
    ])

    # Solve for DB, eq. (44)
    DB = linalg.pinv(kron_prod_real).dot(Q_real)
    DB = DB.reshape(nmodel+l,m, order='F')
    D = DB[:l,:]
    B = DB[l:l+nmodel,:]
    # B = DB[l:l+n,:]

    ## End of (B,D) estimation using no optimisation ##

    # 9:
    # Convert A, B, C, D into continous-time arrays using eq (8)
    # Ad, Bd is the discrete time(frequency domain) matrices.
    # A, B is continuous time
    Ad = A
    A = fs * linalg.logm(Ad)

    Bd = B
    B = A.dot(linalg.solve(Ad - np.eye(len(Ad)),Bd))

    return A, B, C, D


def stabilisation_diagram(fs, U, S, sqCY, nlist, i, m, l, F, fmin, fmax):
    """

    Returns
    -------
    SD : defaultdict of defaultdict(list)
        Stabilization data. Key is model number(int), v is the properties for the
        given model number.
    """
    print( '\nFNSI stabilisation diagram\n' );


    SD = [None]*len(nlist)
    for k, n in enumerate(nlist):
        U1 = U[:,:n]
        S1 = S[:n]
        # Estimation of the extended observability matrix, Γi, eq (21)
        G = sqCY.dot(U1.dot(np.diag(np.sqrt(S1))))

        # Estimate A from eq(24) and C as the first block row of G.
        G_up = G[l:,:]
        G_down = G[:-l,:]
        A = linalg.pinv(G_down).dot(G_up)
        C = G[:l, :]
        # Convert A into continous-time arrays using eq (8)
        A = fs * linalg.logm(A)
        SD[k] = modal_properties(A, C)

    #return SD

    # postprocessing
    tol_freq =1
    tol_damping = 5
    tol_mode = 0.98
    macchoice = 'complex'

    # Initialize SDout as 2x nested defaultdict
    SDout = defaultdict(lambda: defaultdict(list))
    # loop over model orders
    for ior, nval in enumerate(nlist[:-1]):
        # loop over frequencies for current model order
        for ifr, natfreq in enumerate( SD[ior]['natfreq']):
            if natfreq < fmin or natfreq > fmax:
                continue

            # compare with frequencies from one model order higher.
            nfreq = SD[ior+1]['natfreq']
            tol_low = (1 - tol_freq / 100) * natfreq
            tol_high = (1 + tol_freq / 100) * natfreq
            ifreqS, = np.where( (nfreq >= tol_low) & (nfreq <= tol_high) )
            if ifreqS.size == 0:  # ifreqS is empty
                # the current natfreq is not stabilized
                SDout[nval]['stab'].append(False)
                SDout[nval]['freq'].append(natfreq)
                SDout[nval]['ep'].append(False)
                SDout[nval]['mode'].append(False)
            else:
                # Stabilized in natfreq
                SDout[nval]['stab'].append(True)
                SDout[nval]['freq'].append(natfreq)
                # Only in very rare cases, ie multiple natfreqs are very close,
                # is len(ifreqS) != 1
                for ii in ifreqS:
                    nep = SD[ior+1]['ep'][ii]
                    tol_low = (1 - tol_damping / 100) * SD[ior]['ep'][ifr]
                    tol_high = (1 + tol_damping / 100) * SD[ior]['ep'][ifr]
                    # TODO: matlab have find(cond ,1). Ie only return the first match
                    iepS, = np.where( (nep >= tol_low) & (nep <= tol_high) )
                    if iepS.size == 0:
                        SDout[nval]['ep'].append(False)
                    else:
                        SDout[nval]['ep'].append(True)
                if macchoice == 'complex':
                    m1 = SD[ior]['cpxmode'][ifr]
                    m2 = SD[ior+1]['cpxmode'][ifreqS]
                    MAC = ModalACX(m1, m2)
                else:
                    m1 = SD[ior]['realmode'][ifr]
                    m2 = SD[ior+1]['realmode'][ifreqS]
                    MAC = ModalAC(m1, m2)
                if np.max(MAC) >= tol_mode:
                    SDout[nval]['mode'].append(True)
                else:
                    SDout[nval]['mode'].append(False)


    orUNS = []    # Unstabilised model order
    freqUNS = []  # Unstabilised frequency for current model order
    orSfreq = []  # stabilized model order, frequency
    freqS = []    # stabilized frequency for current model order
    orSep = []    # stabilized model order, damping
    freqSep = []  # stabilized damping for current model order
    orSmode = []  # stabilized model order, mode
    freqSmode = []# stabilized damping for current model order
    orSfull = []  # full stabilized
    freqSfull = []
    for k, v in SDout.items():
        # Short notation for the explicit for-loop
        # values = zip(v.values())
        # for freq, ep, mode, stab in zip(*values):
        for freq, ep, mode, stab in zip(v['freq'], v['ep'],
                                        v['mode'], v['stab']):
            if stab:
                if ep and mode:
                    orSfull.append(k)
                    freqSfull.append(freq)
                elif ep:
                    orSep.append(k)
                    freqSep.append(k)
                elif mode:
                    orSmode.append(k)
                    freqSmode.append(freq)

                else:
                    orSfreq.append(k)
                    freqS.append(freq)
            else:
                orUNS.append(k)
                freqUNS.append(freq)

    plt.figure(2)
    plt.clf()
    # Avoid showing the labels of empty plots
    if len(freqUNS) != 0:
        plt.plot(freqUNS, orUNS, 'xr', ms= 7, label='Unstabilized')
    if len(freqS) != 0:
        plt.plot(freqS, orSfreq, 'xk', ms=7, label='Stabilized in natural frequency')
    if len(freqSep) != 0:
        plt.plot(freqSep, orSep, 'sk', ms=7, mfc='none', label='Extra stabilized in damping ratio')
    if len(freqSmode) != 0:
        plt.plot(freqSmode, orSmode, 'ok', ms=7, mfc='none', label='Extra stabilized in MAC')
    if len(freqSfull) != 0:
        plt.plot(freqSfull, orSfull, '^k', ms= 7, mfc='none', label='Full stabilization')

    plt.xlim([fmin, fmax])
    plt.ylim([nlist[0]-2, nlist[-1]])
    ax = plt.gca()
    step = round(nlist[-2]/5)
    major_ticks = np.arange(0, nlist[-2]+1, step)
    ax.set_yticks(major_ticks)

    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Model order')
    plt.title('Stabilization diagram')
    plt.legend(loc='lower right')
    #plt.show()

    return SDout

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


def NLCoefficients(fs, nsper, flines, A, B, C, D, inl, iu, scaling, dofs):
    """Form the extended FRF (transfer function matrix) He(ω) and ectract nonlinear
    coefficients

    H(ω) is the linear FRF matrix, eq. (46)
    He(ω) is formed using eq (47)

    Parameters
    ----------
    fs : float
        Sampling frequency.
    N : int
        The number of time-domain samples.
    flines : ndarray(float)
        Vector of frequency lines where the nonlinear coefficients are
        computed.
    'A,B,C,D' : ndarray x ndarray
        The continuous-time state-space matrices.
    inl : ndarray
        A matrix contaning the locations of the nonlinearities in the system.
    iu : int
        The location of the force.
    dofs : ndarray (int)
        DOFs to calculate H(ω) for.

    Returns
    -------
    knl : ndarray(complex)
        The nonlinear coefficients (frequency-dependent and complex-valued).
    H(ω) : ndarray(complex)
        The extended FRF (transfer function matrix)
    """

    freq = np.arange(0,nsper)*fs/nsper
    F = len(flines)


    l, m = D.shape

    # just return in case of no nonlinearities
    if inl.size == 0:
        #return np.array([]), np.array([])
        knl = np.array([])
        nnl = 0
    else:
        nnl = inl.shape[0]
        # connected from
        inl1 = inl[:,0]
         # connected to
        inl2 = inl[:,1]
        # if connected to ground. For 1 based numbering: +1
        inl2[np.where(inl2 == -1)] = l

        knl = np.empty((F, nnl), dtype=complex)

    m = m - nnl

    H = np.empty((len(dofs), F),dtype=complex)
    He = np.empty((l+1, m+nnl, F),dtype=complex)
    He[-1,:,:] = 0
    for k in range(F):
        He[:-1,:,k] = C.dot(linalg.pinv(np.eye(*A.shape,dtype=complex)*1j*2*np.pi*
                                        freq[flines[k]] - A)).dot(B) + D

        for i in range(nnl):
            knl[k,i] = scaling[i] * He[iu, m+i, k] / (He[inl1[i],0,k] -
                                                      He[inl2[i],0,k] )

        for j, dof in enumerate(dofs):
            H[j,k] = He[dof, 0, k]

    # TODO: I dont remember which of these
    # for k in range(F):
    #     He[:-2,:,k] = C.dot(linalg.pinv(
    #         np.eye(*A.shape,dtype=complex)*1j*2*np.pi* freq[flines[k] + 1])).dot(B) + D

    #     for i in range(nnl):
    #         knl[k,i] = He[iu, m+i, k].dot(
    #             linalg.pinv( He[inl1[i],0,k] - He[inl2[i],0,k] ))

    return knl, H, He

def plot_knl(knl):

    if inl.size == 0:
        return
    freq = np.arange(0,nsper)*fs/nsper
    freq_plot = freq[flines + 1]  # Hz

    # machine precision for float 64. See also
    # https://stackoverflow.com/a/25155518 for an interesting way of doing it.
    eps = np.finfo(float).eps

    mu1 = knl[:,0]
    mu2 = knl[:,1]
    mu1_mean = np.zeros(2)
    mu2_mean = np.zeros(2)
    mu1_mean[0] = np.mean(np.real(mu1))
    mu1_mean[1] = np.mean(np.imag(mu1))
    mu2_mean[0] = np.mean(np.real(mu2))
    mu2_mean[1] = np.mean(np.imag(mu2))

    print('mu1_real {:e}\nmu1_imag {:e}'.format(*mu1_mean))
    print('Ratio mu1 {}'.format(np.log10(np.abs(mu1_mean[0]/mu1_mean[1]))))
    print('mu2_real {:e}\nmu2_imag {:e}'.format(*mu2_mean))
    print('Ratio mu2 {}'.format(np.log10(np.abs(mu2_mean[0]/mu2_mean[1]))))

    mu1_exact = 8.0035e9
    mu2_exact = -1.0505e-7

    # idx = np.where(np.abs(freq_plot - 16.55) < .05 )
    # scaling1 = mu1_exact/np.real(mu1[idx])
    # scaling2 = mu2_exact/np.real(mu2[idx])
    # scaling3 = -2.5e5/np.imag(mu1[idx])
    # scaling4 = 700/np.imag(mu2[idx])
    scaling1=1
    scaling2=1
    scaling3=1
    scaling4=1

    plt.figure(3)
    plt.clf()
    plt.suptitle('Frequency dependence of calculated nonlinear parameters',
                 fontsize=20)

    plt.subplot(2, 2, 1)
    plt.title('Cubic stiffness {:e}'.format(mu1_mean[0]))
    plt.plot(freq_plot, scaling1 * np.real(mu1),label='fnsi')
    plt.axhline(mu1_exact, c='r', ls='--', label='Exact')
    plt.axhline(mu1_mean[0], c='k', ls='--', label='mean')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel(r'Real($\mu$) $(N/m^3)$')
    plt.legend()

    ymin = np.min(np.real(mu1))
    ymax = np.max(np.real(mu1))
    if np.abs(ymax - ymin) <= 1e-6:
      ymin = 0.9 * kmean
      ymax = 1.1 * kmean
      plt.ylim([ymin-eps, ymax+eps])

    #plt.ylim([mu1_exact - abs(0.01*mu1_exact), mu1_exact + abs(0.01*mu1_exact)])

    plt.subplot(2, 2, 3)
    plt.plot(freq_plot, scaling3 * np.imag(mu1))
    plt.axhline(mu1_mean[1], c='k', ls='--', label='mean')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel(r'Imag($\mu$) $(N/m^3)$')

    plt.subplot(2, 2, 2)
    plt.title('Quadratic stiffness {:e}'.format(mu2_mean[0]))
    plt.plot(freq_plot, scaling2*np.real(mu2),label='fnsi')
    plt.axhline(mu2_exact, c='r', ls='--', label='Exact')
    plt.axhline(mu2_mean[0], c='k', ls='--', label='mean')
    plt.legend()
    plt.xlabel('Frequency (Hz)')
    plt.ylabel(r'Real($\mu$) $(N/m^2)$')
    ymin = np.min(np.real(mu2))
    ymax = np.max(np.real(mu2))
    if np.abs(ymax - ymin) <= 1e-6:
      ymin = 0.9 * kmean
      ymax = 1.1 * kmean
      plt.ylim([ymin-eps, ymax+eps])



    plt.subplot(2, 2, 4)
    plt.plot(freq_plot, scaling4* np.imag(mu2))
    plt.axhline(mu2_mean[1], c='k', ls='--', label='mean')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel(r'Imag($\mu$) $(N/m^2)$')
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.show()

def plot_modes(sd):

    plt.figure(4)
    plt.clf()
    plt.title('Linear modes')
    plt.xlabel('Node id')
    plt.ylabel('Displacement (m)')
    # display max 8 modes
    nmodes = min(len(sd['freq']), 8)
    for i in range(nmodes):
        natfreq = sd['natfreq'][i]
        plt.plot(idof, sd['realmode'][i],'-*', label='{:0.2f} Hz'.format(natfreq))
    plt.axhline(y=0, ls='--', lw='0.5',color='k')
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.legend()

def plot_linfrf(H,dofs):

    freq = np.arange(0,nsper)*fs/nsper
    freq_plot = freq[flines]  # Hz

    plt.figure(5)
    plt.clf()

    for i, dof in enumerate(dofs):
        # TODO Why +1?
        plt.plot(freq_plot+1, db(np.abs(H[i])),label='dof: {:d}'.format(dof))

    plt.title('Nonparametric linear FRF')
    plt.xlabel('Frequency (Hz)')
    # For linear scale: 'Amplitude (m/N)'
    plt.ylabel('Amplitude (dB)')
    plt.legend()


abspath='/home/paw/ownCloud/speciale/code/python/vib/'
relpath = 'data/T03b_Data/'
path = abspath + relpath
mat_u =  io.loadmat(path + 'u_15.mat')
mat_y =  io.loadmat(path + 'y_15.mat')
#mat_u =  io.loadmat(path + 'u_01.mat')
#mat_y =  io.loadmat(path + 'y_01.mat')
fs = mat_u['fs'].item() # 3000
fmin = mat_u['fmin'].item()  # 5
fmax = mat_u['fmax'].item()  # 500
iu = mat_u['iu'].item()  # 2 location of force
nper = mat_u['P'].item()
nsper = mat_u['N'].item()
u = mat_u['u'].squeeze()
y = mat_y['y']

signal = Signal(u, y, fs)
per = [4,5,6,7,8,9]
signal.cut(nsper, per)
nper = signal.nper

#inl = np.array([[1,0]])
isnoise = False

inl = np.array([[6,-1], [6,-1], [6,-1]])
inl = np.array([[6,-1], [6,-1]])
#inl = np.array([[]])
enl = np.array([3,2,5])
knl = np.array([1,1,1])

# zero-based numbering of dof
# idof are selected dofs.
# iu are dofs of force
iu = iu-1
idof = [6]
idof = [0,1,2,3,4,5,6]
# dof where nonlinearity is
nldof = []


relpath2 = 'data/fnsi_mats/'
mat = io.loadmat(abspath+relpath2 + 't3b_correct.mat')
# E = mat['E']
# Ymean = mat['Y']
# W = None
# ims = 22
# nmax = 6

# ims: matrix block order. At least n+1
# nmax: max model order for stabilisation diagram
# ncur: model order for erstimation
ims = 22
nmax = 20
ncur = 6

nlist = np.arange(2,nmax+3,2)

f1 = int(np.floor(fmin/fs * nsper))
f2 = int(np.ceil(fmax/fs*nsper))
flines = np.arange(f1,f2+1)

E, Ymean, W, scaling = calc_EY(signal, inl, enl, knl, iu, nldof, isnoise, idof)
print(np.linalg.norm(E), np.linalg.norm(Ymean))
Un, Sn, Rtr, sqCY, m, l, F = svd_comp(signal, E, Ymean, W, ims, flines)

sd = stabilisation_diagram(fs, Un, Sn, sqCY, nlist, ims, m, l, F, fmin, fmax)
plt.show()
A, B, C, D = fnsi_id(Un, Sn, Rtr, sqCY, ims, ncur,  fs, l, m, bd_method=None)
sd = modal_properties(A,C)
plot_modes(sd)

frfdof = np.array([6])
knl, H, He = NLCoefficients(fs, nsper, flines, A, B, C, D, inl, iu, scaling, frfdof)

plot_linfrf(H, frfdof)

plot_knl(knl)

plt.show()


def SimulateLinearStateSpace(A, B, C, D, n, m, l, E, flines, N):

    F = len(flines)
    z = np.exp(np.complex(0, 2*np.pi*flines/N))

    H = np.zeros((l, n, F))
    Y = np.zeros((F, l))
    for f in range(F):
        H[:, :, f] = C / (np.eye(n) * z[f] - A)
        Y[f, :] = E[:, flines[f]].T * np.transpose(H[:,:,f] * B + D, [1, 0, 2])

    J = np.zeros((l*F, n*m + l*m))

    for i in range(n*m):
        I = np.zeros((n, m))
        I[i] = 1
        Jf = np.zeros((F, l))
        for f in range(F):
            Jf[f, :] = E[:, flines[f]].T * np.transpose(H[:,:,f] * I, [1, 0, 2])

        J[:, i] = Jf.flatten()

    for i in range(l*m):
        I = np.zeros((l, m))
        I[i] = 1
        Jf = E[:, flines].T * I.T
    J[:, n*m + i] = Jf.flatten()


    return Y, J
