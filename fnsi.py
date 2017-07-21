#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
#def fnsi_id( E, Y, W, i, n, flines, M, fs, bd_method ):

    Frequency-domain Nonlinear Subspace Identification (FNSI)

    Estimates system parameters from measured signals: acceleration, ü(t), and
    forcing, p(t).

    Given p(t) and u(t), the FNSI method determines the five output matrices.
    The estimation of nonlinear coefficients μ and FRFs is subsequently carried
    out thanks to the conversion from state space to physical space.

    In time domain, the state space formulation is (c denotes continuous time)
    ẋ = Ac*x + Bc*e   : state equation
    y = Cc*x + Dc*e   : measurement equation

    where x is the state vector x = (q, q̇) and e is the concatenation of g(t)
    and p(t), e = (g, p). g(t) is the assumed functional form of the
    nonlinearities. Thus e is known.

    In frequency domain we instead have (d denotes discrete time)
    z_k X = Ad*X(K) + Bd*E(K)
        Y = Cd*X(K) + Dd*E(K)
 
    where X, E, Y are DFTs of x, e and y.
    z_k = exp(j2πk / M) is the z-transform variable.
    We have Cd = Cc, Dd = Dc and
    Ad = expm( Ac / fs)
    Bd = (expm( Ac/fs ) - I)Ac^-1 Bc, I=[n, n]

    E: Size[σ, M]
    Y: Size[l, M]


    Yi defines the 'measured output frequency spectra matrix'
    Yi = (Y^T (Yξ)^T (Yξ^2)^T ... (Yξ^(i-1))^T)^T,  Size[l*i, N]

    Ei defines the 'extended input frequency spectra matrix'
    Ei = (E^T (Eξ)^T (Eξ^2)^T ... (Eξ^(i-1))^T)^T,  Size[σ*i, N]

    Γi defines the 'extended observability matrix'
    Γi = (C^T (CA)^T (CA^2)^T ... (CA^(i-1))^T)^T,  Size[l*i, N]

    where ξ = diag(z1, z2, ..., zN),  Size[N, N]



    Dimensions:
    s: number of nonlinearities/basis functions
    r: ndof
    n: 2*ndof
    m: dofs where applied force is observed/measured, i.e. m ⋜ r
    l: dofs where displacement is observed/measured, i.e. l ⋜ r
    σ: s + m
    N: number of (non-necessarily equidistant) frequency lines used in the
       identification. N = len(flines)
    M: points per period

    fs: sampling frequency
    M: points per period (the number of time-domain samples)
    n: System order used for the FNSI process. The higher the slower
    i: Number of matrix block rows. Determining the size of matrices used.
    flines: is a vector of frequency lines where the nonlinear coefficients are computed.


    For most experiments just assume
    l = r

    Output:
        A
        - State matrix. Size[2*ndof, 2*ndof]
        Bc^nonlin
        - Nonlinear coefficient. Size[2*ndof, s]
        Bc
        - Input feed through matrix. Size[2*ndof, ndof]
        C
        - Output feed through  matrix. Size[ndof, 2*ndof]
        D
        - Direct feed through  matrix. Size[ndof, ndof]



    Method by J.P Noel. Described in article
    "Frequency-domain subspace identification for nonlinear mechanical systems"
    http://dx.doi.org/10.1016/j.ymssp.2013.06.034
    Equations refers to this article

    See http://ipython-books.github.io/featured-01/
    for numpy performance tips.

    TODO:
    Use C-order arrays.
"""
import numpy as np
from scipy import linalg
import matplotlib.pylab as plt
from scipy.io import loadmat

# s = 3
# r = 5
# N = 10

# flines = np.random.rand(6)
# E = np.random.rand(s+1,M)
# Y = np.random.rand(r,M)

sys = loadmat('fnsi_mats/fnsi_1.mat')
sys = loadmat('fnsi_mats/t3b_correct.mat')
fs = sys['fs'][0][0]
i = sys['i'][0][0]
n = sys['n'][0][0]
M = sys['N'][0][0]
Y = sys['Y']
E = sys['E']
W = sys['W']
flines = sys['flines'][0]

# n = 6

print('FNSI analysis')

# dimensions
m = np.shape(E)[0]
l = np.shape(Y)[0]
N = len(flines)

# z-transform variable
# Remember that e^(a+ib) = e^(a)*e^(ib) = e^(a)*(cos(b) + i*sin(b))
# i.e. the exponential of a imaginary number is complex.
zvar = np.empty(flines.shape, dtype=complex)
zvar.real = np.zeros(flines.shape)
zvar.imag = 2 * np.pi * flines / M
zvar = np.exp(zvar)


# dz is an array containing powers of zvar. Eg. the scaling z's in eq. (10)
# (ξ is not used, but dz relates to ξ)
dz = np.zeros((i+1, N), dtype=complex)
for j in range(i+1):
    dz[j,:] = zvar**j

# 2:
# Concatenate external forces and nonlinearities to form the extended input
# spectra Ei
# Initialize Ei and Yi
Emat = np.empty((m * (i + 1), N), dtype=complex)
Ymat = np.empty((l * (i + 1), N), dtype=complex)
for j in range(N):
    # Implemented as the formulation eq. (10), not (11) and (12)
    Emat[:,j] = np.kron(dz[:,j], E[:, flines[j]])
    Ymat[:,j] = np.kron(dz[:,j], Y[:, flines[j]])
print('dtype: {}'.format(Emat.dtype))

# Emat is implicitly recreated as dtype: float64
Emat = np.hstack([np.real(Emat), np.imag(Emat)])
Ymat = np.hstack([np.real(Ymat), np.imag(Ymat)])
print('dtype: {}'.format(Emat.dtype))

P = np.vstack([Emat, Ymat])

# 3:
# Compute the orthogonal projection P = Yi/Ei using QR-decomposition
print('QR decomposition')

R, = linalg.qr(P.T, mode='r')
Rtr = R[:(i+1)*(m+l),:(i+1)*(m+l)].T
R22 = Rtr[(i+1)*m:i*(m+l)+m,(i+1)*m:i*(m+l)+m]

## TODO: Insert conditon for empty W ##

CY = np.eye(l*i)
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
Sn = np.diag(sn)

plt.ion()
plt.figure(1)
plt.clf()
plt.semilogy(np.diag(Sn)/np.sum(np.diag(Sn)),'sk', markersize=6)
plt.xlabel('Singular value')
plt.ylabel('Magnitude')
# plt.show()

# 5:
# Truncate Un and Sn based on the model order n. The model order can be
# determined by inspecting the singular values in Sn or using stabilization
# diagram.
if not n:
    print('System order not set.')
else:
    U1 = Un[:,:n]
    S1 = np.diag(Sn[:n,:n])

# 6:
# Estimation of the extended observability matrix, Γi, eq (21)
G = sqCY.dot(U1.dot(np.diag(np.sqrt(S1))))

# 7:
# Estimate A from eq(24) and C as the first block row of G.
# Recompute G from A and C, eq(13). G plays a major role in determining B and D,
# thus Noel suggest that G is recalculated

A = linalg.pinv(G[:-l,:]).dot(G[l:,:])
C = G[:l,:]

G1 = np.empty(G.shape)
G1[:l,:] = C
for j in range(1,i):
    G1[j*l:(j+1)*l,:] = C.dot(np.linalg.matrix_power(A,j))



G = G1

print('(B,D) estimation using no optimisation')
# 8:
# Estimate B and D
## Start of (B,D) estimation using no optimisation ##

# R_U: Ei+1, R_Y: Yi+1
R_U = Rtr[:m*(i+1),:(m+l)*(i+1)]
R_Y = Rtr[m*(i+1):(m+l)*(i+1),:(m+l)*(i+1)]

# eq. 30
G_inv = linalg.pinv(G)
Q = np.vstack([
    G_inv.dot(np.hstack([np.zeros((l*i,l)), np.eye(l*i)]).dot(R_Y)),
    R_Y[:l,:]]) - \
    np.vstack([
        A,
        C]).dot(G_inv.dot(np.hstack([np.eye(l*i), np.zeros((l*i,l))]).dot(R_Y)))

Rk = R_U

# eq (34) with zeros matrix appended to the end. eq. L1,2 = [L1,2, zeros]
L1 = np.hstack([A.dot(G_inv), np.zeros((n,l))])
L2 = np.hstack([C.dot(G_inv), np.zeros((l,l))])

# The pseudo-inverse of G. eq (33), prepended with zero matrix.
# eq. MM = [zeros, G_inv]
MM = np.hstack([np.zeros((n,l)), G_inv])

# The reason for appending/prepending zeros in L and MM, is to easily form the
# submatrices of N, given by eq. 40. Thus ML is equal to first row of N1
ML = MM - L1

# rhs multiplicator of eq (40)
Z = np.vstack([
    np.hstack([np.eye(l), np.zeros((l,n))]),
    np.hstack([np.zeros((l*i,l)),G])
])


# Assemble the kron_prod in eq. 44.
for kk in range(i+1):
    # Submatrices of N_k. Given by eq (40).
    # eg. N1 corresspond to first row, N2 to second row of the N_k's submatrices
    N1 = np.zeros((n,l*(i+1)))
    N2 = np.zeros((l,l*(i+1)))

    N1[:, :l*(i-kk+1)] = ML[:, kk*l:l*(i+1)]
    N2[:, :l*(i-kk)] = -L2[:, kk*l:l*i]

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
DB = DB.reshape(n+l,m, order='F')
D = DB[:l,:]
B = DB[l:l+n,:]
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


N = M

inl = np.array([[7,0],[7,0]])
iu = 2
freq = np.arange(0,N-1)*fs/N

F = len(flines)

nnl = inl.shape[0]
l, m = D.shape
m = m - nnl

# connected from
inl1 = inl[:,0]
# connected to
inl2 = inl[:,1]
# if connected to ground
inl2[np.where(inl2 == 0)] = l + 1

knl = np.empty((F, nnl),dtype=complex)
He = np.empty((l+1, m+nnl, F),dtype=complex)
He[-1,:,:] = 0
for k in range(F):
    He[:-1,:,k] = C.dot(linalg.pinv(np.eye(*A.shape,dtype=complex)*1j*2*np.pi* freq[flines[k]] - A)).dot(B) + D

    for i in range(nnl):
        # -1 in index because DOF specifying is 1-based.
        knl[k,i] = He[iu-1, m+i, k] / (He[inl1[i]-1,0,k] - He[inl2[i]-1,0,k] )

# scale back
mu1 = 8.0035e9
mu2 = -1.0505e-7

freq_plot = freq[flines]  # Hz
idx = np.where(np.abs(freq_plot - 16.55) < .05 )
scaling1 = mu1/np.real(knl[idx,0])
scaling2 = mu2/np.real(knl[idx,1])
scaling3 = -2.5e5/np.imag(knl[idx,0])
scaling4 = 700/np.imag(knl[idx,1])

params = {'backend': 'pdf',
          'axes.labelsize': 18,
          'font.size': 18,
          'legend.fontsize': 18,
          'xtick.labelsize': 18,
          'ytick.labelsize': 18,
          'text.usetex': True
}
plt.rcParams.update(params)

plt.figure(2, figsize=(16.53,11.69)) #(11.69,8.27))
plt.clf()
plt.suptitle('Frequency dependence of calculated nonlinear parameters',
             fontsize=20)
# fig, axes = plt.subplots(nrows=2, ncols=2)
# for i, ax in enumerate(axes.flat, start=1):
#     print(i)
#     ax.plot(freq_plot,np.real(knl[:,0]))

plt.subplot(2, 2, 1)
plt.title('Cubic stiffness')
plt.plot(freq_plot, *scaling1*np.real(knl[:,0]),label='fnsi')
plt.plot([freq_plot[0],freq_plot[-1]],[mu1, mu1],'--r', label='Exact')
plt.xlabel('Frequency (Hz)')
plt.ylabel(r'Real($\mu$) $(N/m^3)$')
plt.legend()
plt.ylim([mu1 - abs(0.01*mu1), mu1 + abs(0.01*mu1)])

plt.subplot(2, 2, 3)
plt.plot(freq_plot,*scaling3 * np.imag(knl[:,0]))
plt.xlabel('Frequency (Hz)')
plt.ylabel(r'Imag($\mu$) $(N/m^3)$')

plt.subplot(2, 2, 2)
plt.title('Quadratic stiffness')
plt.plot(freq_plot,*scaling2 * np.real(knl[:,1]),label='fnsi')
plt.plot([freq_plot[0],freq_plot[-1]],[mu2, mu2],'--r', label='Exact')
plt.legend()
plt.xlabel('Frequency (Hz)')
plt.ylabel(r'Real($\mu$) $(N/m^2)$')
plt.ylim([mu2 - abs(0.01*mu2), mu2 + abs(0.01*mu2)])


plt.subplot(2, 2, 4)
plt.plot(freq_plot,*scaling4 *np.imag(knl[:,1]))
plt.xlabel('Frequency (Hz)')
plt.ylabel(r'Imag($\mu$) $(N/m^2)$')
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.tight_layout()
plt.subplots_adjust(top=0.85)
plt.show()

plt.savefig('non_coeff.pdf', orientation='landscape')
plt.savefig('non_coeff.png', orientation='landscape')


# Form the extended FRF (transfer function matrix) H(ω) using eq (47)
def NLCoefficients(fs,N, flines, A, B,C, D):
    """
    'fs' is the sampling frequency.
    'N' is the number of time-domain samples.
    'flines' is a vector of frequency lines where the nonlinear coefficients
    are computed.
    'A,B,C,D' are the continuous-time state-space matrices.
    'inl' a matrix contaning the locations of the nonlinearities in the system.
    'iu' is the location of the force.

    'knl' are the nonlinear coefficients (frequency-dependent and complex-valued).
    """

    inl = np.array([[6, 0], [6, 0]])
    freq = np.arange(0,N-1)*fs/N
    F = len(freq)

    nnl = inl.shape[0]
    l, m = D.shape
    m = m - nnl

    inl1 = inl[:,0]
    inl2 = inl[:,1]
    inl2[np.where(inl2 == 0)] = l + 1

    knl = np.empty((F, nnl))
    He = np.empty((l+1, m+nnl, F))
    for k in range(F):
        He[:-2,:,k] = C.dot(linalg.pinv(
            np.eye(A.shape,dtype=complex)*1j*2*np.pi* freq[flines[k] + 1])).dot(B) + D

    for i in range(nnl):
        knl[k,i] = He[iu, m+i, k].dot(
            linalg.pinv( He[inl1[i],0,k] - He[inl2[i],0,k] ))

    return knl
# for k = 1:F,
    
#     He(1:end-1,:,k) = C/(eye(size(A))*complex(0,2*pi*freq(flines(k)+1)) - A)*B + D;
    
#     for i = 1:nnl,
#         knl(k,i) = He(iu,m+i,k)/(He(inl1(i),1,k)-He(inl2(i),1,k));
#     end  
    
# end

# for omega in range(10):
#     # np.diag([omega*1j]*n)
#     He = C.dot(linalg.inv(np.eye(n,dtype=complex)*1j*omega - A).dot(B)) + D
#     # H = He[:n,:]
#     # take spectral mean to find value
