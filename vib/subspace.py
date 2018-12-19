
import numpy as np
from scipy.linalg import (lstsq, qr, svd, logm, inv, norm, eigvals)
from numpy import kron
from numpy.linalg import solve, pinv

def _matrix_square_inv(A):
    """Calculate the inverse of the matrix square root of `A`

    Calculate `X` such that XX = inv(A)
    `A` is assumed positive definite, thus the all singular values are strictly
    positive. Given the svd decomposition A=UsVᴴ, we see that
    AAᴴ = Us²Uᴴ (remember (UsV)ᴴ = VᴴsUᴴ) and it follows that
    (AAᴴ)⁻¹/² = Us⁻¹Uᴴ

    Returns
    -------
    X : ndarray(n,n)
       Inverse of matrix square root of A

    Notes
    -----
    See the comments here.
    https://math.stackexchange.com/questions/106774/matrix-square-root

    """
    U, s, _ = svd(A, full_matrices=False)
    return U * 1/np.sqrt(s) @ U.conj().T

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

    if domain == 'z': # discrete-time
        # Unstable if at least one pole outside unit circle
        if any(abs(eigvals(A)) > 1):
            return False
    elif domain == 's': # continuous-time
        # Unstable if at least one pole in right-half plane
        if any(np.real(eigvals(A)) > 0):
            return False
    else:
        raise ValueError('{domain} wrong. Use "s" or "z"'.
                         format(domain=repr(domain)))
    return True

def jacobian_freq_ss(A,B,C,z):
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

    F = len(z) # Number of frequencies
    n = np.shape(A)[0]  # Number of states
    m = np.shape(B)[1]  # Number of inputs
    p = np.shape(C)[0]  # Number of outputs

    JA = np.empty((F,p,m,n*n),dtype=complex)
    JB = np.empty((F,p,m,n*m),dtype=complex)
    JC = np.empty((F,p,m,n*p),dtype=complex)

    # get rows and columns in A for a given index: A(i)=A(k(i),ell(i))
    k, ell = np.unravel_index(np.arange(n**2), (n,n))
    In = np.eye(n)
    Im = np.eye(m)
    Ip = np.eye(p)
    # TODO must vectorize...
    # see for calling lapack routines directly
    # https://stackoverflow.com/a/11999063/1121523
    # see for multicasting
    # https://docs.scipy.org/doc/numpy/reference/routines.linalg.html#linear-algebra-on-several-matrices-at-once
    # Note that using inv(A) implicitly calls solve and creates an identity
    # matrix. Thus it is faster to allocate In once and then call solve.
    for f in range(F):
        temp1 = solve((z[f]*In - A),In)
        temp2 = C @ temp1
        temp3 = temp1 @ B

        # Jacobian w.r.t. all elements in A, A(i)=A(k(i),ell(i))
        # Note that the partial derivative of e(f) w.r.t. A(k(i),ell(i)) is
        # equal to temp2*fOne(n,n,i)*temp3, and thus
        # JA(:,:,i,f) = temp2(:,k(i))*temp3(ell(i),:)
        for i in range(n**2): # Loop over all elements in A
            JA[f,:,:,i] = temp2[:,k[i]] @ temp3[ell[i],:]

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

    return JA, JB, JC

def weight_jacobian_ss(jac, W):
    """Computes the Jacobian of the weighted error ``e_W(f) = W(f,:,:)*e(f)``


    This uses broadcasting.
    """

    # np.einsum('ijk,kl',W, jac) or
    # np.einsum('ijk,kl->ijl',W, jac) or
    # np.einsum('ijk,jl->ilk',W,jac)
    # np.tensordot(W, jac, axes=1)
    # np.matmul(W, jac)
    return np.matmul(W, jac)


def normalize_columns(jac):

    # Rms values of each column
    scaling = np.sqrt(np.mean(jac**2,axis=0))
    # or scaling = 1/np.sqrt(jac.shape[0]) * np.linalg.norm(jac,ord=2,axis=0)
    # Robustify against columns with zero rms value
    scaling[scaling == 0] = 1
    # Scale columns with 1/rms value
    jac /= scaling
    return jac, scaling


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
        Gmat[:,f*m:(f+1)*m] = kron(Wr[:,f], G[f]).T
        Umat[:,f*m:(f+1)*m] = kron(Wr[:,f], np.eye(m)).T

    # 1.e. and 1.f: split into real and imag part and stack into Z
    # we do it in a memory efficient way and avoids intermediate memory copies.
    # (Just so you know: It is more efficient to stack the result in a new
    # memory location, than overwriting the old). Ie.
    # Gre = np.hstack([Gmat.real, Gmat.imag]) is more efficient than
    # Gmat = np.hstack([Gmat.real, Gmat.imag])
    Z = np.empty((r*(p+m), 2*F*m))
    Z[:r*p,:F*m] = Umat.real
    Z[:r*p,F*m:] = Umat.imag
    Z[r*p:,:F*m] = Gmat.real
    Z[r*p:,F*m:] = Gmat.imag

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
    _, R = qr(Z.T, mode='economic')
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
    weight = np.zeros_like(covarG) # .transpose((2,0,1))
    for f in range(F):
        weight[f] = _matrix_square_inv(covarG[f])

    # Compute partial derivative of the weighted error, e = W*(Ĝ - G)
    # Ĝ(f) = C*inv(z(f)*I - A)*B + D and W = 1/σ_G.
    # wrt. B
    B = np.zeros((n,m))
    # unweighted jacobian
    _, JB, _ = jacobian_freq_ss(A,B,C,z)
    # add weight
    JB.shape =(F, p*m, m*n)
    JB = weight_jacobian_ss(JB, weight).reshape((p*m*F, m*n))

    # The jacobian wrt. elements of D is a zero matrix where one element is
    # one times the weight
    JD = np.zeros((p*m*F,p*m),dtype=complex)
    for f in range(F):
        JD[p*m*f:(f+1)*p*m,:] = weight[f]

    # Compute e = -W*(0 - G), i.e. minus the weighted error(the residual) when
    # considering zero initial estimates for Ĝ
    resG = weight_jacobian_ss(G, weight).ravel()

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

    return A, B, C, D, isstable


def fss2frf(A,B,C,D,freq):
    """Compute frequency response function from state-space parameters
    (discrete-time)

    Computes the frequency response function (FRF) or matrix (FRM) GSS at the
    normalized frequencies `freq` from the state-space matrices `A`, `B`, `C`,
    and `D`. ```GSS(f) = C*inv(exp(1j*2*pi*f)*I - A)*B + D```

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

from vib.frf import periodic
import scipy.io as sio
from vib.common import db, import_npz
import matplotlib.pyplot as plt

data = sio.loadmat('data.mat')

Y = data['Y'].transpose((1,2,3,0))
U = data['U'].transpose((1,2,3,0))
G_data = data['G']
covGML_data = data['covGML']
covGn_data = data['covGn']

lines = data['lines'].squeeze()
non_exc_even = data['non_exc_even'].squeeze()
non_exc_odd = data['non_exc_odd'].squeeze()

N = 1024
freq = (lines-1)/N  # Excited frequencies (normalized)

G, covG, covGn = periodic(U, Y)
G = G.transpose((2,0,1))
covG = covG.transpose((2,0,1))

n = 2
r = 3
fs = 1
optimize = True
Kss = np.zeros((n,r))

# weight
if covG is None:
    covGinv = np.tile(np.eye((m*p)),(F,1,1))
else:
    covGinv = pinv(covG)

A, B, C, D, isstable = subspace(G, covG, freq/fs, n, r)

# frf of the state space model
Gss = fss2frf(A,B,C,D,freq/fs)
errSS = G - Gss
# cost = ∑ₖ e[k]ᴴ*σ_G⁻¹*e[k]
cost = np.real(np.einsum('ki,kij,kj',errSS.conj().reshape(errSS.shape[0],-1),
                         covGinv,errSS.reshape(errSS.shape[0],-1)) )

# Normalize with the number of excited frequencies
cost /= F
Kss[n-1,r-1] = cost

# Do Levenberg-Marquardt optimizations
if optimize:
    pass

""" Optimize state-space matrices using Levenberg-Marquardt.

Optimize state-space matrices A, B, C, and D using at most nmax
Levenberg-Marquardt runs starting from initial values `A0`, `B0`, `C0`, and `D0`. The
cost function that is optimized is the weighted sum of the mean-square
differences between the provided frequency response matrix G at the normalized
frequencies freq and the estimated frequency response matrix ``Ĝ = C*(zₖ*Iₙ -
A)⁻¹*B + D``. The cost function is weighted with the inverse of
the covariance matrix of G, if this covariance matrix covG is provided (no
weighting if covG = 0).

	Output parameters:
		A : n x n optimized state matrix
       B : n x m optimized input matrix
       C : p x n optimized output matrix
       D : p x m optimized feed-through matrix

	Input parameters:
       freq : vector of normalized frequencies at which the FRM is given
              (0 < freq < 0.5)
		G : p x m x F array of FRF (or FRM) data
		covG : p*m x p*m x F covariance tensor on G
              (0 if no weighting required)
		A0 : n x n initial state matrix
       B0 : n x m initial input matrix
       C0 : p x n initial output matrix
       D0 : p x m initial feed-through matrix
		MaxCount : maximum number of Levenberg-Marquardt optimizations
                  (maximum 1000, if MaxCount = Inf, then the algorithm
                  stops if the cost functions of the last 10 successful
                  steps differ by less than 0.1% or if 1000 iterations
                  were performed)
"""

# Number of parameters
npar = n**2 + n*m + p*n + p*m

info = False
