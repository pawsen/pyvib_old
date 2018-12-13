
import numpy as np
from scipy.linalg import (lstsq, solve, qr, svd, logm, pinv, inv, kron)

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


def jacobian_freq_ss(A,B,C,z):
    """Compute Jacobians of the unweighted errors wrt. model parameters.

    Computes the Jacobians of the unweighted errors ``e(f) = Ĝ(f) - G(f)``
    w.r.t. the elements in the ``A``, ``B``, and ``C`` state-space matrices.
    The Jacobian w.r.t. the elements of ``D`` is a zero matrix where one
    element is one. ``Ĝ(f) = C*(z(f)*I - A)^(-1)*B + D`` is the estimated and
    ``G(f)`` is the measured frequency response matrix (FRM).

    The structure of the Jacobian is: ``JX[p,m,i,f]`` where ``p`` and ``m`` are
    inputs and outputs and ``f`` the frequency line. ``i`` is the index
    mapping, relating the matrix element ``(k,l)`` of ``X`` to the linear index
    of the vector ``JX[p,m,:,f]``. This mapping is given by, fx for ``A``:
    ``i = np.ravel_multi_index((k,l) ,(n,n))`` and the reverse is
    ``k, l = np.unravel_index(i, (n,n))``. Thus ``JA(:,:,i,f)`` contains the
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
    JA : ndarray(p,m,n*n,F)
        JA(:,:,i,f) contains the partial derivative of the unweighted error
        e(f) at frequency f wrt. A(k,l)
    JB : ndarray(p,m,n*m,F)
        JB(:,:,i,f) contains the partial derivative of e(f) w.r.t. B(k,l)
    JC : ndarray(p,m,p*n,F)
        JC(:,:,i,f) contains the partial derivative of e(f) w.r.t. C(k,l)

    Notes
    -----
    See eq. (5-103) in :cite:pauldart2008

    """

    F = len(z) # Number of frequencies
    n = np.shape(A)[0]  # Number of states
    m = np.shape(B)[1]  # Number of inputs
    p = np.ahape(C)[0]  # Number of outputs

    JA = np.empty(p,m,n*n,F)
    JB = np.empty(p,m,n*m,F)
    JC = np.empty(p,m,n*p,F)

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
        temp1 = solve((z(f)*In - A),In, check_finte+False)
        temp2 = C @ temp1
        temp3 = temp1 @ B

        # Jacobian w.r.t. all elements in A, A(i)=A(k(i),ell(i))
        # Note that the partial derivative of e(f) w.r.t. A(k(i),ell(i)) is
        # equal to temp2*fOne(n,n,i)*temp3, and thus
        # JA(:,:,i,f) = temp2(:,k(i))*temp3(ell(i),:)
        for i in range(n**2): # Loop over all elements in A
            JA[:,:,i,f] = temp2[:,k[i]] @ temp3[ell[i],:]

        # Jacobian w.r.t. all elements in B
        # Note that the partial derivative of e(f) w.r.t. B(k,l) is equal to
        # temp2*fOne(n,m,sub2ind([n m],k,l)), and thus
        # JB(:,l,sub2ind([n m],k,l),f) = temp2(:,k)
        JB[:,:,:,f] = np.reshape(kron(Im, temp2), (p,m,m*n))

        # Jacobian w.r.t. all elements in C
        # Note that the partial derivative of e(f) w.r.t. C(k,l) is equal to
        # fOne(p,n,sub2ind([p n],k,l))*temp3, and thus
        # JC(k,:,sub2ind([p n],k,l),f) = temp3(l,:)
        JC[:,:,:,f] = np.reshape(kron(temp3.T, Ip), (p,m,n*p))

    return JA, JB, JC

def weight_jacobian_ss():
    pass

def subspace(H, covarH, freq, n, r):
    """Estimate state-space model from Frequency Response Function (or Matrix).

    The linear state-space model is estimated from samples of the frequency
    response function (or frequency response matrix). The frequency-domain
    subspace method in `McKelvey1996`_ is applied with the frequency weighting
    in `Pintelon2002`_, i.e. weighting with the sampled covariance matrix.

    `p`: number of outputs, `m`: number of inputs, `F`: number of frequencies.

    Parameters
    ----------
    H : complex ndarray(p, m, F)
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
    p,m,F = H.shape

    # 1.a. Construct Wr with z
    z = np.exp(2j*np.pi*freq)
    Wr = (np.tile(z[:,None], r)**np.tile(np.arange(r),(len(z),1))).T

    # 1.b. and 1.c. Construct Hmat and Umat
    Hmat = np.empty(r*p,F*m, dtype=complex)
    Umat = np.empty(r*m,F*m, dtype=complex)
    for f in range(F):
        Hmat[:,f*m:(f+1)*m] = kron(Wr[:,f], H[:,:,f])
        Umat[:,f*m:(f+1)*m] = kron(Wr[:,f], np.eye(m))

    # 1.e. and 1.f: split into real and imag part and stack into Z
    # we do it in a memory efficient way and avoids intermediate memory copies.
    # (Just so you know: It is more efficient to stack the result in a new
    # memory location, than overwriting the old). Ie.
    # Hre = np.hstack([Hmat.real, Hmat.imag]) is more efficient than
    # Hmat = np.hstack([Hmat.real, Hmat.imag])
    Z = np.empty((r*(p+m), 2*F*m))
    Z[:r*p,:F*m] = Hmat.real
    Z[:r*p,F*m:] = Hmat.imag
    Z[r*p:,:F*m] = Umat.real
    Z[r*p:,F*m:] = Umat.imag

    # 1.f. Calculate CY from σ²_G
    if covarH is None:
        CY = np.eye(p*Wr.shape[0])
        covarH = np.tile(np.eye(p*m), (1,1,F))
    else:
        CY = np.zeros((p*Wr.shape[0],p*Wr.shape[0]))
        for f in range(F):
            # Take sum over the diagonal blocks of cov(vec(H)) (see
            # paduart2008(5-93))
            temp = np.zeros((p,p))
            for i in range(m-1):
                temp += covarH[i*p:(i+1)*p, i*p:(i+1)*p, f]
                CY += np.real(kron(np.outer(Wr[:,f], Wr[:,f].conj(),temp)))

    # 1.g. QR decomposition of Z.T, Z=R.T*Q.T, to eliminate U from Z.
    _, R = qr(Z.T, mode='economic')
    RT = R.T
    RT22 = RT[-r*p:,-r*p]

    # 1.h. CY^(-1/2)*RT22=USV', Calculate CY^(-1/2)
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
    Or = sqrtCY*Un[:,:n]

    # 2. Estimate A and C from shift property of Or
    # Recompute G from A and C. G plays a major role in determining B
    # and D, thus J.P. Noel suggest that G is recalculated
    # A = pinv(Or[:-p,:]) @ Or[p:,:]
    A, *_ = lstsq(Or[:-p,:], Or[p:,:])
    C = Or[:p,:].copy()

    # 3. Estimate B and D given A,C and H: (W)LS estimate
    # Compute weight, W = sqrt(σ²_G^-1)
    weight = np.zeros(covarH.shape)
    for f in range(F):
        weight[:,:,f] = _matrix_square_inv(covarH[:,:,f])

    # Compute partial derivative of the weighted error, e = W*(Ĝ - G)
    # Ĝ(f) = C*inv(z(f)*I - A)*B + D and W = 1/σ_G.
    # wrt. B
    B = np.zeros((n,m))
    # unweighted jacobian
    _, JB, _ = jacobian_freq_ss(A,B,C,z)
    # add weight


    # The jacobian wrt. elements of D is a zero matrix where one element is
    # one times the weight
    JD = np.zeros((p*m*F,p*m))
    for f in range(F):
        JD[p*m*k:(f+1)*p*m,:] = weight[:,:,f]

    # Compute e = -W*(0 - G), i.e. minus the weighted error(the residual) when
    # considering zero initial estimates for Ĝ
    resG = reshape(fWeightJacobSubSpace(reshape(H,p*m,F),c,p*m,F,1),[p*m*F,1]);

    # TODO: test performance vs preallocating. This makes a lot of intermediate copies
    resGre = np.vstack([resG.real, resG.imag])
    # B and D as one parameter vector => concatenate Jacobians
    J = np.hstack([JB, JD])
    Jre = np.vstack([J.real, J.imag])

    # Normalize columns of Jacobian with their rms value
    Jre, scaling = fNormalizeColumns(Jre)
    # Compute Gauss-Newton parameter update. For small residuals e(Θ), the
    # Hessian can be approximated by the Jacobian of e. See (5.140) in
    # paduart2008.
    dtheta, res, rnk, s = lstsq(Jre, resGre, check_finte=False)
    dtheta /= scaling

    # Parameter estimates = parameter update, since zero initial estimates considered
    B[:] = dtheta[:n*m]
    D = np.zeros((p,m))
    D[:p*m] = dtheta[n*m:]

    # Check stability of the estimated model
    isstable = True

    return A, B, C, D, isstable


import timeit

s1 = """\
z = np.exp(2j*np.pi*freq)
"""
s2 = """\
zvar = np.empty(freq.shape, dtype=complex)
zvar.real = np.zeros(freq.shape)
zvar.imag = 2 * np.pi * freq
zvar = np.exp(zvar)
"""
setup_statement = ';'.join([
    'import numpy as np',
    'n = int(1e5)',
    'freq = np.array(n)',
])

t1 = timeit.Timer(s1, setup=setup_statement)
t2 = timeit.Timer(s2, setup=setup_statement)
print(t1.timeit(number=5))
print(t2.timeit(number=5))

setup_statement2 = ';'.join([
    'import numpy as np',
    'n = int(1e3)',
    'Hmat = np.ones((n,n)) + 1j*np.ones((n,n))',
    'Umat = np.ones((n,n)) + 1j*np.ones((n,n))',
])

s3 = """\
Hmat = np.hstack([np.real(Hmat), np.imag(Hmat)])
Umat = np.hstack([np.real(Umat), np.imag(Umat)])
Z = np.vstack([Hmat, Umat])
"""

s4 = """\
Hre = np.hstack([np.real(Hmat), np.imag(Hmat)])
Ure = np.hstack([np.real(Umat), np.imag(Umat)])
Z = np.vstack([Hre, Ure])
"""

s5 = """\
Z = np.empty((2*n,2*n))
Z[:n,:n] = Hmat.real
Z[:n,n:] = Hmat.imag
Z[n:,:n] = Umat.real
Z[n:,n:] = Umat.imag
"""


t3 = timeit.Timer(s3, setup=setup_statement2)
t4 = timeit.Timer(s4, setup=setup_statement2)
t5 = timeit.Timer(s5, setup=setup_statement2)
print(t3.timeit(number=3))
print(t4.timeit(number=3))
print(t5.timeit(number=3))


# import numpy as np
# n = int(1e3)
# freq = np.array(n)
# Emat = np.empty((n,n), dtype=complex)
# Mmat = np.hstack([np.real(Emat), np.imag(Emat)])


import numpy as np
A = np.random.rand(1000,3,3)
def slow_inverse(A):
    Ainv = np.zeros_like(A)

    for i in range(A.shape[0]):
        Ainv[i] = np.linalg.inv(A[i])
    return Ainv

def fast_inverse(A):
    identity = np.identity(A.shape[2], dtype=A.dtype)
    Ainv = np.zeros_like(A)

    for i in range(A.shape[0]):
        Ainv[i] = np.linalg.solve(A[i], identity)
    return Ainv

%timeit -n 20 aI11 = slow_inverse(A)
