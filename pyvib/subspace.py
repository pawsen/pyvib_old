#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .common import matrix_square_inv, mmul_weight, normalize_columns
import numpy as np
# qr(mode='r') returns r in economic form. This is not the case for scipy
# svd and solve allows broadcasting when imported from numpy
from numpy.linalg import qr, svd, solve
from scipy.signal import dlsim
from scipy.linalg import (lstsq, norm, eigvals)
from numpy import kron


def is_stable(A, domain='z'):
    """Determines if a linear state-space model is stable from eigenvalues of `A`

    Parameters
    ----------
    A : ndarray(n,n)
        state matrix
    domain : str, optional {'z', 's'}
        'z' for discrete-time, 's' for continuous-time state-space models

    returns
    -------
    bool
    """

    if domain == 'z':  # discrete-time
        # Unstable if at least one pole outside unit circle
        if any(abs(eigvals(A)) > 1):
            return False
    elif domain == 's':  # continuous-time
        # Unstable if at least one pole in right-half plane
        if any(np.real(eigvals(A)) > 0):
            return False
    else:
        raise ValueError(f"{domain} wrong. Use 's' or 'z'")
    return True

def jacobian_freq(A,B,C,z):
    """Compute Jacobians of the unweighted errors wrt. model parameters.

    Computes the Jacobians of the unweighted errors ``e(f) = Ĝ(f) - G(f)``
    w.r.t. the elements in the ``A``, ``B``, and ``C`` state-space matrices.
    The Jacobian w.r.t. the elements of ``D`` is a zero matrix where one
    element is one. ``Ĝ(f) = C*(z(f)*I - A)^(-1)*B + D`` is the estimated and
    ``G(f)`` is the measured frequency response matrix (FRM).

    The structure of the Jacobian is: ``JX[f,p,m,i]`` where ``p`` and ``m`` are
    inputs and outputs and ``f`` the frequency line. ``i`` is the index
    mapping, relating the matrix element ``(k,l)`` of ``X`` to the linear index
    of the vector ``JX[p,m,:,f]``. This mapping is given by, fx for ``A``:
    ``i = np.ravel_multi_index((k,l) ,(n,n))`` and the reverse is
    ``k, l = np.unravel_index(i, (n,n))``. Thus ``JA(f,:,:,i)`` contains the
    partial derivative of the unweighted error ``e(f)`` at frequency `f` wrt.
    ``A(k,l)``

    Parameters
    ----------
    A : ndarray(n,n)
        state matrix
    B : ndarray(n,m)
        input matrix
    C : ndarray(p,n)
        output matrix
    z : ndarray(F)
        ``z = exp(2j*pi*freq)``, where freq is a vector of normalized
        frequencies at which the Jacobians are computed (0 < freq < 0.5)

    Returns
    -------
    JA : ndarray(F,p,m,n*n)
        JA(f,:,:,i) contains the partial derivative of the unweighted error
        e(f) at frequency f wrt. A(k,l)
    JB : ndarray(p,m,n*m,F)
        JB(f,:,:,i) contains the partial derivative of e(f) w.r.t. B(k,l)
    JC : ndarray(p,m,p*n,F)
        JC(f,:,:,i) contains the partial derivative of e(f) w.r.t. C(k,l)

    Notes
    -----
    See eq. (5-103) in :cite:pauldart2008

    """

    F = len(z)          # Number of frequencies
    n = np.shape(A)[0]  # Number of states
    m = np.shape(B)[1]  # Number of inputs
    p = np.shape(C)[0]  # Number of outputs

    JA = np.empty((F,p,m,n*n),dtype=complex)
    JB = np.empty((F,p,m,n*m),dtype=complex)
    JC = np.empty((F,p,m,n*p),dtype=complex)

    # get rows and columns in A for a given index: A(i)=A(k(i),ell(i))
    k, ell = np.unravel_index(np.arange(n**2), (n,n))
    # Note that using inv(A) implicitly calls solve and creates an identity
    # matrix. Thus it is faster to allocate In once and then call solve.
    In = np.eye(n)
    Im = np.eye(m)
    Ip = np.eye(p)
    # TODO must vectorize...
    # see for calling lapack routines directly
    # https://stackoverflow.com/a/11999063/1121523
    # see for multicasting
    # https://docs.scipy.org/doc/numpy/reference/routines.linalg.html#linear-algebra-on-several-matrices-at-once
    for f in range(F):
        temp1 = solve((z[f]*In - A),In)
        temp2 = C @ temp1
        temp3 = temp1 @ B

        # Jacobian w.r.t. all elements in A, A(i)=A(k(i),ell(i))
        # Note that the partial derivative of e(f) w.r.t. A(k(i),ell(i)) is
        # equal to temp2*fOne(n,n,i)*temp3, and thus
        # JA(:,:,i,f) = temp2(:,k(i))*temp3(ell(i),:)
        for i in range(n**2): # Loop over all elements in A
            JA[f,:,:,i] = np.outer(temp2[:,k[i]], temp3[ell[i],:])

        # Jacobian w.r.t. all elements in B
        # Note that the partial derivative of e(f) w.r.t. B(k,l) is equal to
        # temp2*fOne(n,m,sub2ind([n m],k,l)), and thus
        # JB(:,l,sub2ind([n m],k,l),f) = temp2(:,k)
        JB[f] = np.reshape(kron(Im, temp2), (p,m,m*n))

        # Jacobian w.r.t. all elements in C
        # Note that the partial derivative of e(f) w.r.t. C(k,l) is equal to
        # fOne(p,n,sub2ind([p n],k,l))*temp3, and thus
        # JC(k,:,sub2ind([p n],k,l),f) = temp3(l,:)
        JC[f] = np.reshape(kron(temp3.T, Ip), (p,m,n*p))

    # JD does not change over iterations
    JD = np.zeros((p,m,p*m))
    for f in range(p*m):
        np.put(JD[...,f], f, 1)
    JD = np.tile(JD, (F,1,1,1))

    return JA, JB, JC, JD

def subspace(G, covarG, freq, n, r):
    """Estimate state-space model from Frequency Response Function (or Matrix).

    The linear state-space model is estimated from samples of the frequency
    response function (or frequency response matrix). The frequency-domain
    subspace method in `McKelvey1996`_ is applied with the frequency weighting
    in `Pintelon2002`_, i.e. weighting with the sampled covariance matrix.

    `p`: number of outputs, `m`: number of inputs, `F`: number of frequencies.

    Parameters
    ----------
    G : complex ndarray(p, m, F)
        Frequency Response Matrix (FRM)
    covarG : ndarray(p*m, p*m, F)
        σ²_G, Covariance tensor on G (None if no weighting required)
    freq : ndarray(F)
        Vector of normalized frequencies at which the FRM is given (0 < freq < 0.5)
    n : int
        Model order
    r : int
        Number of block rows in the extended observability matrix (r > n)

    Returns
    -------
    A : ndarray(n, n)
        state matrix
    B : ndarray(n, m)
        input matrix
    C : ndarray(p, n)
        output matrix
    D : ndarray(p, m)
        feed-through matrix
    unstable : boolean
        Indicating whether or not the identified state-space model is unstable

    Notes
    -----
    Algorithm: (see p. 119 `Paduart2008`_ for details)
    From a DFT of the state space eqs., and recursive use of the two equations
    give the relation: ``Gmat = OᵣX + SᵣU``. From this ``A`` and ``C`` are
    determined. ``B`` and ``D`` are found by minimizing the weighted error
    ``e(f) = W*(Ĝ(f) - G(f))`` where ``Ĝ(f) = C*(z(f)*I - A)^(-1)*B + D`` is
    the estimated- and ``G(f)`` is the measured frequency response matrix(FRM).
    The weight, ``W=1/σ_G``, is chosen in :cite:pinleton2002, sec. 5, to almost
    eliminate the bias resulting from observing the inputs and outputs ``U``
    and ``Y`` with errors.

    In ``Gmat``, ``Sᵣ`` is a lower triangular block toeplitz matrix and ``Oᵣ``,
    ``U`` are extended matrices and found as:
      1. Construct Extended observability matrix Oᵣ
          a. Construct Wᵣ with z
          b. Construct Hmat with H and Wᵣ
          c. Construct Umat with Wᵣ (U=eye(m))
          d. Split real and imaginary parts of Umat and Hmat
          e. Z=[Umat; Hmat]
          f. Calculate CY
          g. QR decomposition of Zᵀ
          h. CY^(-1/2)*RT22=USV'
          i. Oᵣ=U(:,1:n)
      2. Estimate A and C from the shift property of Oᵣ
      3. Estimate B and D given A,C and H

    References
    ----------
    .. _McKelvey1996:
       McKelvey T., Akcay, H., and Ljung, L. (1996).
       Subspace-Based Multivariable System Identification From Frequency
       Response Data. IEEE Transactions on Automatic Control, 41(7):960-979

    .. _Pintelon2002:
       Pintelon, R. (2002). Frequency-domain subspace system identification
       using non-parametric noise models. Automatica, 38:1295-1311

    .. _Paduart2008:
       Paduart J. (2008). Identification of nonlinear systems using polynomial
       nonlinear state space models. PhD thesis, Vrije Universiteit Brussel.

    """
    #  number of outputs/inputs and number of frequencies
    F,p,m = G.shape

    # 1.a. Construct Wr with z
    z = np.exp(2j*np.pi*freq)

    Wr = (np.tile(z[:,None], r)**np.tile(np.arange(r),(len(z),1))).T

    # 1.b. and 1.c. Construct Gmat and Umat
    Gmat = np.empty((r*p,F*m), dtype=complex)
    Umat = np.empty((r*m,F*m), dtype=complex)
    for f in range(F):
        Gmat[:,f*m:(f+1)*m] = kron(Wr[:,f,None], G[f])
        Umat[:,f*m:(f+1)*m] = kron(Wr[:,f,None], np.eye(m))

    # 1.e. and 1.f: split into real and imag part and stack into Z
    # we do it in a memory efficient way and avoids intermediate memory copies.
    # (Just so you know: It is more efficient to stack the result in a new
    # memory location, than overwriting the old). Ie.
    # Gre = np.hstack([Gmat.real, Gmat.imag]) is more efficient than
    # Gmat = np.hstack([Gmat.real, Gmat.imag])
    Z = np.empty((r*(p+m), 2*F*m))
    Z[:r*m,:F*m] = Umat.real
    Z[:r*m,F*m:] = Umat.imag
    Z[r*m:,:F*m] = Gmat.real
    Z[r*m:,F*m:] = Gmat.imag

    # 1.f. Calculate CY from σ²_G
    if covarG is None:
        CY = np.eye(p*Wr.shape[0])
        covarG = np.tile(np.eye(p*m), (F,1,1))
    else:
        CY = np.zeros((p*Wr.shape[0],p*Wr.shape[0]))
        for f in range(F):
            # Take sum over the diagonal blocks of cov(vec(H)) (see
            # paduart2008(5-93))
            temp = np.zeros((p,p),dtype=complex)
            for i in range(m):
                temp += covarG[f, i*p:(i+1)*p, i*p:(i+1)*p]
                CY += np.real(kron(np.outer(Wr[:,f], Wr[:,f].conj()),temp))

    # 1.g. QR decomposition of Z.T, Z=R.T*Q.T, to eliminate U from Z.
    R = qr(Z.T, mode='r')
    RT = R.T
    RT22 = RT[-r*p:,-r*p:]

    # 1.h. CY^(-1/2)*RT22=USV', Calculate CY^(-1/2) using svd decomp.
    # Use gesdd as driver. Matlab uses gesvd. The only reason to use dgesvd
    # instead is for accuracy and workspace memory. The former should only be
    # important if you care about high relative accuracy for tiny singular
    # values. Mem: gesdd: O(min(m,n)^2) vs O(max(m,n)) gesvd
    UC, sc, _ = svd(CY, full_matrices=False)

    # it is faster to work on the diagonal scy, than the full matrix SCY
    # Note: We work with real matrices here, thus UC.conj().T -> UC.T
    sqrtCY = UC * np.sqrt(sc) @ UC.conj().T
    invsqrtCY = UC * 1/np.sqrt(sc) @ UC.conj().T

    # Remove noise. By taking svd of CY^(-1/2)*RT22
    Un, sn, _ = svd(invsqrtCY @ RT22)

    ## RETURN HERE

    # 1.i. Estimate extended observability matrix
    Or = sqrtCY @ Un[:,:n]

    # 2. Estimate A and C from shift property of Or
    # Recompute G from A and C. G plays a major role in determining B
    # and D, thus J.P. Noel suggest that G is recalculated
    # A = pinv(Or[:-p,:]) @ Or[p:,:]
    A, *_ = lstsq(Or[:-p,:], Or[p:,:])
    C = Or[:p,:].copy()

    # 3. Estimate B and D given A,C and H: (W)LS estimate
    # Compute weight, W = sqrt(σ²_G^-1)
    # TODO this is calculated multiple places
    weight = np.zeros_like(covarG) # .transpose((2,0,1))
    for f in range(F):
        weight[f] = matrix_square_inv(covarG[f])

    # Compute partial derivative of the weighted error, e = W*(Ĝ - G)
    # Ĝ(f) = C*inv(z(f)*I - A)*B + D and W = 1/σ_G.
    # wrt. B
    B = np.zeros((n,m))
    # unweighted jacobian
    _, JB, _, _ = jacobian_freq(A,B,C,z)
    # add weight
    JB.shape =(F, p*m, m*n)
    JB = mmul_weight(JB, weight).reshape((p*m*F, m*n))

    # The jacobian wrt. elements of D is a zero matrix where one element is one
    # times the weight
    # TODO this is already calculated in jacobian
    JD = np.zeros((p*m*F,p*m),dtype=complex)
    for f in range(F):
        JD[p*m*f:(f+1)*p*m,:] = weight[f]

    # Compute e = -W*(0 - G), i.e. minus the weighted error(the residual) when
    # considering zero initial estimates for Ĝ
    resG = mmul_weight(G, weight).ravel()

    # TODO: test performance vs preallocating. This makes some intermediate copies
    resGre = np.hstack((resG.real, resG.imag))
    # B and D as one parameter vector => concatenate Jacobians
    # We do: J = np.hstack([JB, JD]), Jre = np.vstack([J.real, J.imag]), but faster
    Jre = np.empty((2*p*m*F, m*(n+p)))
    Jre[:p*m*F, :m*n] = JB.real
    Jre[:p*m*F, m*n:] = JD.real
    Jre[p*m*F:, :m*n] = JB.imag
    Jre[p*m*F:, m*n:] = JD.imag

    # Normalize columns of Jacobian with their rms value
    Jre, scaling = normalize_columns(Jre)
    # Compute Gauss-Newton parameter update. For small residuals e(Θ), the
    # Hessian can be approximated by the Jacobian of e. See (5.140) in
    # paduart2008.
    dtheta, res, rank, s = lstsq(Jre, resGre, check_finite=False)
    dtheta /= scaling

    # Parameter estimates = parameter update, since zero initial estimates considered
    B.flat[:] = dtheta[:n*m]
    D = np.zeros((p,m))
    D.flat[:] = dtheta[n*m:]

    # Check stability of the estimated model
    isstable = is_stable(A)

    return A, B, C, D, z, isstable


def ss2frf(A,B,C,D,freq):
    """Compute frequency response function from state-space parameters
    (discrete-time)

    Computes the frequency response function (FRF) or matrix (FRM) Ĝ at the
    normalized frequencies `freq` from the state-space matrices `A`, `B`, `C`,
    and `D`. ```̂G(f) = C*inv(exp(2j*pi*f)*I - A)*B + D```

    Returns
    -------
    Gss : ndarray(F,p,m)
        frequency response matrix

    """

    # Z-transform variable
    z = np.exp(2j*np.pi*freq)
    In = np.eye(*A.shape)
    # Use broadcasting. Much faster than for loop.
    Gss = C @ solve((z*In[...,None] - A[...,None]).transpose((2,0,1)), B[None]) + D

    return Gss


def jacobian(x0, system, weight=None):

    n, m, p, npar = system.n, system.m, system.p, system.npar
    F = system.signal.F

    A, B, C, D = extract_ss(x0, system)
    JA, JB, JC, JD = jacobian_freq(A,B,C,system.z)

    tmp = np.empty((F,p,m,npar),dtype=complex)
    tmp[...,:n**2] = JA
    tmp[...,n**2 +       np.r_[:n*m]] = JB
    tmp[...,n**2 + n*m + np.r_[:n*p]] = JC
    tmp[...,n**2 + n*m + n*p:] = JD
    tmp = tmp.reshape((F,m*p,npar))

    if weight is not None:
        tmp = mmul_weight(tmp, weight)
    tmp = tmp.reshape((F*p*m,npar))

    jac = np.empty((2*F*p*m, npar))
    jac[:F*p*m] = tmp.real
    jac[F*p*m:] = tmp.imag

    return jac

def costfcn(x0, system, weight=None):
    """Compute the vector of residuals such that the function to mimimize is

    res = ∑ₖ e[k]ᴴ*e[k], where the error is given by
    e = weight*(Ĝ - G)
    and the weight is the square inverse of the covariance matrix of `G`,
    weight = \sqrt(σ_G⁻¹) Ĝ⁻¹

    """

    freq, fs = system.signal.freq, system.signal.fs
    A, B, C, D = extract_ss(x0, system)

    # frf of the state space model
    Gss = ss2frf(A,B,C,D,freq/fs)
    err = Gss - system.G
    if weight is not None:
        err = mmul_weight(err, weight)
    err_w = np.hstack((err.real.ravel(), err.imag.ravel()))

    return err_w  # err.ravel()

def extract_ss(x0, system):

    n, m, p = system.n, system.m, system.p
    A = x0.flat[:n**2].reshape((n,n))
    B = x0.flat[n**2 + np.r_[:n*m]].reshape((n,m))
    C = x0.flat[n**2+n*m + np.r_[:p*n]].reshape((p,n))
    D = x0.flat[n*(p+m+n):].reshape((p,m))

    return A, B, C, D

def extract_model(models, y, u, dt, t=None, x0=None):
    """extract the best model using validation data"""

    dictget = lambda d, *k: [d[i] for i in k]
    err_old = np.inf
    err_vec = np.empty(len(models))
    for i, (k, model) in enumerate(models.items()):

        A, B, C, D = dictget(model, 'A', 'B', 'C', 'D')
        system = (A, B, C, D, dt)
        tout, yout, xout = dlsim(system, u, t, x0)
        err_rms = np.sqrt(np.mean((y - yout)**2))
        err_vec[i] = err_rms
        if err_rms < err_old:
            n = k
            err_old = err_rms

    return models[n], err_vec
