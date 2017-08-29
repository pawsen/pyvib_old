#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.linalg import block_diag, kron, solve, lstsq
from scipy.linalg import norm
from scipy.fftpack import fft, ifft
from scipy import linalg
from scipy import sparse

import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from scipy import io
from forcing import sineForce
from hb_plots import Anim


class HB():

    def __init__(self, M0, C0, K0, nonlin,
                 NH=3, npow2=8, nu=1, scale_x=1, scale_t=1,
                 amp0=1e-3, tol_NR=1e-6, max_it_NR=15,
                 stability=True, rcm_permute=False):
        """Because frequency(in rad/s) and amplitude have different orders of
        magnitude, time and displacement have to be rescaled to avoid
        ill-conditioning.

        Parameters
        ----------
        Harmonic balance parameters:
        NH: number of harmonic retained in the Fourier series
        npow: number of time samples in the Fourier transform, ie. 2^8=256
              samples
        nu: accounts for subharmonics of excitation freq w0
        amp0: amplitude of first guess

        """

        self.NH = NH
        self.npow2 = npow2
        self.nu = nu
        self.scale_x = scale_x
        self.scale_t = scale_t
        self.amp0 = amp0
        self.stability = stability
        self.tol_NR = tol_NR * scale_x
        self.max_it_NR = max_it_NR
        self.rcm_permute = rcm_permute
        self.nt = 2**npow2

        self.M = M0 * scale_x / scale_t**2
        self.C = C0 * scale_x / scale_t
        self.K = K0 * scale_x
        self.n = M0.shape[0]

        self.nonlin = nonlin

        # number of unknowns in Z-vector, eq (4)
        self.nz = self.n * (2 * NH + 1)

        self.z_vec = []
        self.xamp_vec = []
        self.omega_vec = []
        self.stab_vec = []
        self.lamb_vec = []

    def periodic(self, f0, f_amp, fdofs):
        """Find periodic solution

        NR iteration to solve:
        # Solve h(z,ω)=A(ω)-b(z)=0 (ie find z that is root of this eq), eq. (21)
        # NR solution: (h_z is the derivative of h wrt z)
        # zⁱ⁺¹ = zⁱ - h(zⁱ,ω)/h_z(zⁱ,ω)
        # h_z = A(ω) - b_z(z) = A - Γ⁺ ∂f/∂x Γ
        # where the last exp. is in time domain. df/dx is thus available analytical.
        # See eq (30)

        Parameters
        ----------
        f0: float
            Forcing frequency in Hz
        f_amp: float
            Forcing amplitude
        """
        self.f0 = f0
        self.f_amp = f_amp
        self.fdofs = fdofs

        nu = self.nu
        NH = self.NH
        n = self.n
        nz = self.nz
        scale_x = self.scale_x
        scale_t = self.scale_t
        amp0 = self.amp0
        stability = self.stability
        tol_NR = self.tol_NR
        max_it_NR = self.max_it_NR

        w0 = f0 *2*np.pi
        omega = w0*scale_t
        omega2 = omega / nu

        t = assemblet(self, omega2)
        nt = len(t)

        force = sineForce(f_amp, omega, t, n, fdofs)

        # Assemble A, describing the linear dynamics. eq (20)
        A = assembleA(self, omega2)

        # Form Q(t), eq (8). Q is orthogonal trigonometric basis(harmonic terms)
        # Then use Q to from the kron product Q(t) ⊗ Iₙ, eq (6)
        mat_func_form = np.empty((n*nt, nz))
        Q = np.empty((NH*2+1,1))
        for i in range(nt):
            Q[0] = 1
            for ii in range(1,NH+1):
                Q[ii*2-1] = np.sin(omega * t[i]*ii)
                Q[ii*2] = np.cos(omega * t[i]*ii)

            # Stack the kron prod, so each block row is for time(i)
            mat_func_form[i*n:(i+1)*n,:] = kron(Q.T, np.eye(n))
        self.mat_func_form_sparse = sparse.csr_matrix(mat_func_form)

        if stability:
            hills_stability_init(self)

        # Initial guess for x. Here calculated by broadcasting. np.outer could
        # be used instead to calculate the outer product
        amp = np.ones(n)*amp0
        x_guess = amp[:,None] * np.sin(omega * t)/scale_x

        # Initial guess for z, z = Γ⁺x as given by eq (26). Γ is found as
        # Q(t)⊗Iₙ, eq 6
        # TODO use mat_func_form_sparse
        z_guess, *_ = lstsq(mat_func_form, x_guess.ravel())

        # Solve h(z,ω)=A(ω)-b(z)=0 (ie find z-root), eq. (21)
        print('Newton-Raphson iterative solution')

        Obj = 1
        it_NR = 1
        z = z_guess
        while (Obj > tol_NR ) and (it_NR <= max_it_NR):
            H = state_sys(self, z, A, force)
            H_z = hjac(self, z, A)

            zsol, *_ = lstsq(H_z,H)
            z = z - zsol

            Obj = linalg.norm(H) / (eps + linalg.norm(z))
            print('It. {} - Convergence test: {:e} ({:e})'.
                  format(it_NR, Obj, tol_NR))
            it_NR = it_NR + 1

        if it_NR > max_it_NR:
            raise ValueError("Number of NR iterations exceeded {}."
                             "Change Harmonic Balance parameters".
                             format(max_it_NR))

        if stability:
             B, stab = hills_stability(self, omega, H_z)

        # amplitude of hb components
        c, cnorm, phi = hb_components(scale_x*z, n, NH)
        x = hb_signal(omega, t, c, phi)
        xamp = np.max(x, axis=1)

        self.z_vec.append(z)
        self.xamp_vec.append(xamp)
        self.omega_vec.append(omega)
        self.stab_vec.append(stab)
        self.lamb_vec.append(B)

        return omega, z, stab, B


    def continuation(self, omega_min, omega_max,
                     step=0.1,step_min=0.1, step_max=20, cont_dir=1,
                     opt_it_NR=3, it_cont_max=1e4, adaptive_stepsize=True,
                     angle_max_pred=90):
        """ Do continuation of periodic solution.

        Based on tangent prediction and Moore-Penrose correction.
        """

        z = self.z
        omega = self.omega
        scale_x = self.scale_x
        scale_y = self.scale_y
        n = self.n
        NH = self.NH

        step_min = step_min * scale_t
        step_max = step_max * scale_t
        omega_min = omega_min*2*np.pi
        omega_max = omega_max*2*np.pi
        angle_max_pred = angle_max_pred*np.pi/180


    def get_components(self, omega=None, z=None):
        scale_x = self.scale_x
        scale_t = self.scale_t
        n = self.n
        NH = self.NH
        nu = self.nu

        if omega is None:
            omega = self.omega_vec[-1]
            z = self.z_vec[-1]

        c, phi, cnorm = hb_components(scale_x*z, n, NH)

        omega2 = omega/nu
        iw = np.arange(NH+1) * omega2
        cd = c*iw
        cd[:,0] = 0
        phid = phi + np.pi/2

        cdd = -c * iw**2
        cdd[:,0] = 0
        phidd = phi

        t = assemblet(self, omega2)*scale_t
        # make t one full period + dt, ie include first point of next period
        t = np.append(t, t[-1] + t[1] - t[0])
        omegap = omega/scale_t
        omega2p = omega2/scale_t
        zp = z*scale_x

        return t, omegap, zp, cnorm, (c, phi), (cd, phid), (cdd, phidd)



def state_sys(self, z, A, force):
    """Calculate the system matrix h(z,ω) = A(ω)z - b(z), given by eq. (21).

    b is calculated from an alternating Frequency/Time domain (AFT) technique,
    eq. (23).

        not known
      z +--------> b(z)
      +            ^
      | FFT⁻¹      | FFT
      v            +
     x(t)+---->f(x,ẋ,ω,t)

    The Fourier coefficients of the nonlinear forces b(z) are found
    """

    n = self.n
    nt = self.nt
    NH = self.NH
    scale_x = self.scale_x
    nonlin = self.nonlin

    x = ifft_coeff(z, n, nt, NH)
    fnl = force_nl(x*scale_x, *nonlin)

    b = fft_coeff(force - fnl, NH)
    # eq. 21
    H = A @ z - b
    return H

def hjac(self, z, A, force=None):
    """ Computes the jacobian matrix of h wrt Fourier coefficients z, denoted
    h_z.

    From eq. (30)
    h_z = ∂h/∂z = A - ∂b/∂z = A - Γ⁺ ∂f/∂x Γ
    ∂f/∂x is available analytical in time domain

    It is only necessary to calculate the jacobian ∂b/∂z for nonlinear forces
    (ie. it is zero for linear forces).

    """
    nz  =self.nz
    nt = self.nt
    NH = self.NH
    n = self.n
    scale_x = self.scale_x
    mat_func_form_sparse = self.mat_func_form_sparse
    nonlin = self.nonlin

    if inl.size == 0:
        return A
    else:
        x = ifft_coeff(z, n, nt, NH)
        dFnl_dx_tot_mat = der_force_nl(x*scale_x, *nonlin) * scale_x
        # the derivative ∂b/∂z
        bjac = np.empty((nz, nz))
        full_rhs = dFnl_dx_tot_mat @ mat_func_form_sparse

        # TODO try use linear solver instead! Check if mat_func is square.
        for j in range(nz):
            rhs = np.squeeze(np.asarray(full_rhs[:,j].todense()))
            return_values = sparse.linalg.lsmr(mat_func_form_sparse, rhs)
            x = - return_values[0]
            bjac[:,j] = x

        hjac = A - bjac
        return hjac


def hjac_omega(self, omega, z):
    """ Calculate the derivative of h wrt ω, h_ω.

    From eq. 32,
    h_ω = ∂A/∂ω*z
    """
    M = self.M
    C = self.C
    K = self.K
    NH  = self.NH
    A = np.zeros(K.shape)
    for i in range(1,NH+1):
        blk = np.vstack((
            np.hstack((-2*i**2 * omega * M, -i * C)),
            np.hstack((i * C, -2*i**2 * omega * M)) ))
        A = block_diag( A, blk )

    return A @ z

def fft_coeff(x, NH):
    """ Extract FFT-coefficients from X=fft(x)
    """
    # n: dofs
    n, nt = x.shape
    # Format of X after transpose: (nt, n)
    X = fft(x).T / nt

    re_fft_im_fft = np.hstack([-2*np.imag(X[1:NH+1]),
                               2* np.real(X[1:NH+1])])

    # X[0] only contains real numbers (it is the dc/0-frequency-part), but we
    # still need to extract the real part. Otherwise z is casted to complex128
    z = np.hstack([np.real(X[0]), re_fft_im_fft.ravel()])

    return z

def ifft_coeff(z, n, nt, NH):
    """ extract iFFT-coefficients from x=ifft(X)
    """

    X = np.zeros((n, nt), dtype='complex')
    X[:,0] = nt*z[:n]

    for i in range(NH):
        X[:,i+1] = 2 * (nt/2   * z[(n*(2*i+1)+n): (n*(2*i+1)+2*n)] - \
                        nt/2*1j* z[(n*(2*i+1))  : (n*(2*i+1)+n)])

    x = np.real(ifft(X))

    return x

def hb_components(z, n, NH):
    z = np.hstack([np.zeros(n), z])
    # reorder so first column is zeros, then one column for each dof
    z = np.reshape(z, (n,2*(NH+1)), order='F')

    # first column in z is zero, thus this will a 'division by zero' warning.
    # Instead we just set the first column in phi to pi/2 (arctan(inf) = pi/2)
    # phi = np.arctan(z[:,1::2] / z[:,::2])
    phi = np.empty((n, NH+1))
    phi[:,1:] =np.arctan(z[:,3::2] / z[:,2::2])
    phi[:,0] = np.pi/2

    c = z[:,::2] / np.cos(phi)
    c[:,0] = z[:,1]

    cnorm = np.abs(c) / (np.max(np.abs(c)))

    return c, phi, cnorm

def hb_signal(omega, t, c, phi):
    n = c.shape[0]
    NH = c.shape[1]-1
    nt = len(t)
    tt = np.arange(1,NH+1)[:,None] * omega * t

    x = np.zeros((n, nt))
    for i in range(n):

        tmp = tt + np.outer(phi[i,1:],np.ones(nt))
        tmp = c[i,0]*np.ones(nt) + c[i,1:] @ np.sin(tmp)
        x[i] = tmp #np.sum(tmp, axis=0)

    return x

def assembleA(self, omega2):
    """Assemble A, describing the linear dynamics. eq (20)
    """
    M = self.M
    C = self.C
    K = self.K
    NH = self.NH

    A = K
    for i in range(1,NH+1):
        blk = np.vstack((
            np.hstack((K - ( i * omega2 )**2 * M, -i * omega2 * C)),
            np.hstack((i * omega2 * C, K - (i * omega2)**2 * M)) ))
        A = block_diag( A, blk )
    return A

def assemblet(self, omega2):
    npow2 = self.npow2
    return np.arange(2**npow2) / 2**npow2 * 2*np.pi / omega2


def hills_stability_init(self):
    """Estimate Floquet multipliers from Hills method.

    If one of the Floquet exponent have a positive real part, the solution is
    unstable.
    Only the 2n eigenvalues with lowest imaginary modulus approximate the
    Floquet multipliers λ. The rest are spurious. (n is the number of DOFs)

    The Hill matrix is simply a by-product of a harmonic-balance continuation
    method in the frequency domain. Thus there is no need to switch from one
    domain to another for computing stability.
    (The only term that needs to be evaluated when z varies is h_z)
    The monodromy matrix might be faster computationally-wise, but this is a
    by-product of the shooting continuation method in the time-domain approach.
    Hills method is effective for large systems.[1]_

    Assemble Δs of the linear eigenvalue problem, eq. 44. B2 is assembled now.
    B1 requires h_z and is thus assembled after the steady state solution is
    found.

    Notes
    -----
    [1]_: Peletan, Loïc, et al. "A comparison of stability computational
    methods for periodic solution of nonlinear problems with application to
    rotordynamics." Nonlinear dynamics 72.3 (2013): 671-682.
    """
    scale_t = self.scale_t
    scale_x = self.scale_x
    NH = self.NH
    M0 = self.M * scale_t**2 / scale_x
    C0 = self.C * scale_t / scale_x
    K0 = self.K / scale_x

    Delta2 = M0
    M_inv = linalg.inv(M0)
    Delta2_inv = M_inv

    for i in range(1,NH+1):
        Delta2 = block_diag(Delta2, M0, M0)
        Delta2_inv = block_diag(Delta2_inv, M_inv, M_inv)

    # eq. 45
    b2 = np.vstack((
        np.hstack((Delta2, np.zeros(Delta2.shape))),
        np.hstack((np.zeros(Delta2.shape), np.eye(Delta2.shape[0]))) ))

    b2_inv = - np.vstack((
        np.hstack((Delta2_inv, np.zeros(Delta2_inv.shape))),
        np.hstack((np.zeros(Delta2_inv.shape), np.eye(Delta2.shape[0]))) ))

    self.Delta2 = Delta2
    self.b2_inv = b2_inv
    #return Delta2, b2_inv

def hills_stability(self, omega, H_z, it = None):#, Delta2, b2_inv):
    scale_x = self.scale_x
    scale_t = self.scale_t
    n = self.n
    rcm_permute = self.rcm_permute
    Delta2 = self.Delta2
    b2_inv = self.b2_inv
    NH = self.NH
    nu = self.nu

    omega2 = omega/nu
    # eq. 38
    Delta1 = C0
    for i in range(1,NH+1):
        blk = np.vstack((
            np.hstack((C0, - 2*i * omega2/scale_t * M0)),
            np.hstack((2*i * omega2/scale_t * M0, C0)) ))
        Delta1 = block_diag(Delta1, blk)

    # eq. 45
    A0 = H_z/scale_x
    A1 = Delta1
    A2 = Delta2
    b1 = np.vstack((
        np.hstack((A1, A0)),
        np.hstack((-np.eye(A0.shape[0]), np.zeros(A0.shape))) ))
    # eq. 46
    mat_B = b2_inv @ b1
    if rcm_permute:
        # permute B to get smaller bandwidth which gives faster linalg comp.
        p = sparse.csgraph.reverse_cuthill_mckee(mat_B)
        B_tilde = mat_B[p]
    else:
        B_tilde = mat_B
    # Get 2n eigenvalues with lowest imaginary modulus
    B = B_reduc(B_tilde, n)

    if np.max(np.real(B)) <= 0:
        stab = True
    else:
        stab = False

    return B, stab

def B_reduc(B_tilde, n):
    """Find the 2n first eigenvalues with lowest imaginary part"""

    w = linalg.eigvals(B_tilde)
    idx = np.argsort(np.abs(np.imag(w)))[:2*n]

    return w[idx]

def der_force_nl(x, inl, knl, enl):
    """Derivative of nonlinear functional
    """
    if inl.size == 0:
        return np.array([])

    ndof, ns = x.shape
    nbln = inl.shape[0]
    idof = np.arange(ndof)

    dfnl = np.zeros((ndof+1, ns*(ndof+1)))
    x = np.vstack((x, np.zeros((1,ns)) ))
    for j in range(nbln):
        # connected from
        i1 = inl[j,0]
        # conencted to
        i2 = inl[j,1]

        # Convert to the right index
        idx1 = np.where(i1==idof)[0][0]
        x1 = x[idx1]

        # if connected to ground
        if i2 == -1:
            idx2 = ndof
            x2 = 0
        else:
            idx2 = np.where(i2==idof)[0][0]
            x2 = x[idx2]
        x12 = x1 - x2

        df12 = knl[j] * enl[j] * np.abs(x12)**(enl[j]-1)

        # in case of even functional
        if (enl[j] % 2 == 0):
            idx = np.where(x12 < 0)
            df12[idx] = -df12[idx]
            #x12 = np.abs(x12)

        # add the nonlinear force to the right dofs
        dfnl[idx1, idx1::ndof+1] += df12
        dfnl[idx2, idx1::ndof+1] -= df12
        dfnl[idx1, idx2::ndof+1] -= df12
        dfnl[idx2, idx2::ndof+1] += df12

    if ns == 1:
        return dfnl[:dof,:dof]


    # create sparse structure from dfnl
    # TODO: dont create dfnl in the first place...:)
    ind = np.arange(ns*(ndof+1))
    ind = np.delete(ind, np.s_[ndof::ndof+1])
    dfnl = dfnl[:ndof, ind]
    dfnl = np.reshape(dfnl, (ndof**2,ns), order='F')
    # dont ask...
    ind = np.outer(np.ones(ndof), np.arange(ndof)) * ns*ndof + \
          np.outer(np.arange(ndof), np.ones(ndof))
    ind = np.outer(ind.T, np.ones(ns)) + \
          ns*ndof * np.outer(np.ones(ndof**2), np.arange(0,(ns-1)*ndof+1, ndof)) + \
          ndof * np.outer(np.ones(ndof**2), np.arange(ns))

    ind = ind.ravel(order='F').astype('int')

    #https://stackoverflow.com/questions/28995146/matlab-ind2sub-equivalent-in-python
    arr = ns*ndof*np.array([1,1])
    ii, jj = np.unravel_index(ind, tuple(arr), order='F')
    dfnl_s = sparse.coo_matrix((dfnl.ravel(order='F'), (ii, jj)),
                               shape=(ndof*ns, ndof*ns)).tocsr()

    return dfnl_s


def force_nl(x, inl, knl, enl):
    if inl.size == 0:
        return np.array([0])

    # ns = Nt
    ndof, ns = x.shape
    nbln = inl.shape[0]
    idof = np.arange(ndof)

    fnl = np.zeros((ndof+1, ns))
    x = np.vstack((x, np.zeros((1,ns)) ))
    nbln = inl.shape[0]

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
            idx2 = ([ndof],)
            x2 = 0
        else:
            idx2 = np.where(i2==idof)
            x2 = x[idx2]
        x12 = x1 - x2
        # in case of even functional
        if (enl[j] % 2 == 0):
            x12 = np.abs(x12)

        f12 = knl[j] * x12**enl[j]
        fnl[idx1] += f12
        fnl[idx2] -= f12
    fnl = fnl[:ndof,:]
    return fnl



M0 = 1
C0 = 0.02
K0 = 1
M0, C0, K0 = np.atleast_2d(M0,C0,K0)
f_amp = 0.1
f0 = 1e-3
fdofs = 0

inl = np.array([[0,-1]])
# inl = np.array([])
enl = np.array([3])
knl = np.array([1])

par_hb ={
    'NH': 5,
    'npow2': 9,
    'nu': 1,
    'stability': True,
    'rcm_permute': False,
    'tol_NR': 1e-6,
    'max_it_NR': 15,
    'scale_x': 1,
    'scale_t': 1
}
par_cont = {
    'omega_cont_min': 1e-5*2*np.pi,
    'omega_cont_max': 0.5*2*np.pi,
    'cont_dir': 1,
    'opt_it_NR': 3,
    'step': 0.1,
    'step_min': 0.001,
    'step_max': 0.1,
    'angle_max_pred': 90,
    'it_cont_max': 1e4,
    'adaptive_stepsize': True
}


abspath='/home/paw/ownCloud/speciale/code/python/vib/'
relpath = 'data/T05_Data/'
path = abspath + relpath
mat =  io.loadmat(path + 'NLBeam.mat')
# M0 = mat['M']
# C0 = mat['C']
# K0 = mat['K']

## Force parameters
# location of harmonic force
# fdofs = 7
# f_amp = 3
# # Excitation frequency. lowest sine freq in Hz
# f0 = 25
# par_hb ={
#     'NH': 5,
#     'npow2': 9,
#     'nu': 1,
#     'stability': True,
#     'rcm_permute': False,
#     'tol_NR': 1e-6,
#     'max_it_NR': 15,
#     'scale_x': 5e-6, # == 5e-12
#     'scale_t': 3000
# }

# par_cont = {
#     'omega_cont_min': 25*2*np.pi,
#     'omega_cont_max': 40*2*np.pi,
#     'cont_dir': 1,
#     'opt_it_NR': 3,
#     'step': 0.1,
#     'step_min': 0.1,
#     'step_max': 20,
#     'angle_max_pred': 90,
#     'it_cont_max': 1e4,
#     'adaptive_stepsize': True
# }

# inl = np.array([[27,-1], [27,-1]])
# # inl = np.array([])
# enl = np.array([3,2])
# knl = np.array([8e9,-1.05e7])


nonlin = (inl, knl, enl)
# machine precision for float
eps = np.finfo(float).eps


hb = HB(M0,C0,K0,nonlin,**par_hb)
omega, z, stab, B = hb.periodic(f0, f_amp, fdofs)
tp, omegap, zp, cnorm, c, cd, cdd = hb.get_components()

#hb.continuation(,**par_cont)

import matplotlib.pyplot as plt
plt.ion()
from hb_plots import plots
dof = 0
plot = False
if plot:
    plots(tp, omegap, cnorm, c, cd, cdd, dof, B, inl, gtype='displ', savefig=False)


#!!! CONT !!!#

self = hb
f0 = self.f0
f_amp = self.f_amp
fdofs = self.fdofs
nu = self.nu
NH = self.NH
n = self.n
nz = self.nz
scale_x = self.scale_x
scale_t = self.scale_t
amp0 = self.amp0
stability = self.stability
tol_NR = self.tol_NR
max_it_NR = self.max_it_NR

omega_cont_min = par_cont['omega_cont_min']
omega_cont_max = par_cont['omega_cont_max']
cont_dir = par_cont['cont_dir']
opt_it_NR = par_cont['opt_it_NR']
step = par_cont['step']
step_min = par_cont['step_min']
step_max = par_cont['step_max']
angle_max_pred = par_cont['angle_max_pred']
it_cont_max = par_cont['it_cont_max']
adaptive_stepsize = par_cont['adaptive_stepsize']

xamp_vec =[]
omega_vec = []
z_vec = []
stab_vec = []
lamb_vec = []
step_vec = []

omega2 = omega/nu
t = assemblet(self, omega2)
x = hb_signal(omega, t, *c)
xamp = np.max(x, axis=1)
xamp_vec.append(xamp)
omega_vec.append(omega)
stab_vec.append(stab)
lamb_vec.append(B)
z_vec.append(z)

anim = Anim(hb, omega_vec, xamp_vec, omega_cont_min, omega_cont_max)

def test_func(G, p0, q0, bif_type, f=None):
    """
    If f=None, then the test_func is returned.
    Else the extended state sys is returned.

    Parameters
    ----------

    """

    nG = G.shape[1]

    if f is None:
        q0 = np.zeros(q0.shape)
        q0[0] = 1
        p0 = np.zeros(p0.shape)
        p0[0] = 1

    if bif_type == 'LP' or bif_type == 'NS':

        # data = np.diag(G)
        # data.append(p0)
        # data.append(q0)
        # ii = np.arange(0,nG*nG+nG-1,nG+2)
        #M = sparse.coo_matrix((data, (ii,jj)), shape=(nG+1,nG+1))
        M = sparse.coo_matrix((nG+1, nG+1))
        idx = np.diag_indices(nG+1, ndim=2)
        M[idx] = np.diag(G)
        M[nG,:nG] = p0
        M[:nG,nG] = q0

        Mb = np.zeros(nG+1)
        Mb[nG] = 1

        wg = sparse.linalg.spsolve(M.tocsr(), Mb)

    elif bif_type == 'BP' or bif_type == 'LP2':

        M = np.vstack((
            np.hstack((G, p0[:,None])),
            np.hstack((q0, 0))
        ))

        wg = solve(M, np.append(np.zeros(nG),1))

    w = wg[:nG]
    g = wg[nG]
    if f is None:
        return g, q0, p0

    f = np.append(f,g)
    return f, M, w


def null_approx(A, bif_type):
    if bif_type == 'LP2' or bif_type == 'BP':

        U, s, V = linalg.svd(A, lapack_driver='gesvd')
        # In case of multiple occurrences of the minimum values, the indices
        # corresponding to the first occurrence are returned.
        idx = np.argmin(np.abs(s))
        eig_sm = np.abs(s[idx])
        if eig_sm > 5e-3:
            print('The approximation of the bifurcation point has to be improved'
                  'The smallest eigenvalue of the matrix is {:.2e}'.format(eig_sm))
            return

        # V: Unitary matrix having right singular vectors as rows.
        nullvec = V[idx]
    else:

        if not np.all(A == np.diag(np.diagonal(A))):
            raise ValueError('The matrix A should be diagonal')

        eig_A = eigvals(A)
        idx = np.argmin(np.abs(eig_A))

        nullvec = np.zeros(A.shape[0])
        nullvec[idx] = 1

    return nullvec

def extended_jacobian5_detect(self, omega, z, A, B, M_ext, w, ind_f, bif_type):
    n = self.n
    nG = M_ext.shape[1] -1
    nz = self.nz

    J_z_f = jac_sys(self, A, omega, z)
    J_w = jac_w(self, omega)
    # jacobian_param2
    btronc = np.zeros(nz)
    btronc[2*n+ind_f] += 1
    J_w2 = -btronc
    J_w_f = np.hstack((J_w, J_w2))

    vh = solve(M_ext,np.append(np.zeros(nG),1))
    v = vh[:nG]

    J_part = np.empty(nz)
    # TODO parallel loop
    for i in range(nz):
        if i in ind_nl_ext:
            hess = hessian4_detect(self, omega, z, A, B, i, 'z', bif_type)
            J_g_A =- v[:,None] @ hess * w
        else:
            J_g_A = 0
        J_part[i] = np.real(J_g_A)

    # TODO DONT pass 0!!!
    hess = hessian4_detect(self, omega, z, A, B, 0,'omega', bif_type)
    J_g_p1 = - v[:,None] @ hess * w

    hess = hessian4_detect(self, omega, z, A, B, 0, 'f', bif_type)
    J_g_p2 = - v[:,None] @ hess * w

    J_part = np.append(J_part, np.real(J_g_p1), np.real(J_g_p2))

    J = np.vstack((
        np.hstack((J_z_f, J_w_f)),
        J_part
    ))
    return J


def hessian4_detect(self, omega, z, A, B, idx, gtype, bif_type):
    scale_x = self.scale_x
    nu = self.nu

    eps = 1e-5
    if gtype == 'z':
        z_pert = z

        if z_pert[idx] != 0:
            eps = eps * abs(z_pert[idx])

        z_pert[idx] = z_pert[idx] + eps
        J_z_pert = jac_sys(self, z_pert, A) / scale_x

        if bif_type == 'LP2':
            return (J_z_pert - B) / eps
        elif bif_type == 'BP':
            J_w_pert = jac_w(self, omega, z_pert)
            dG_dalpha = (np.vstack((np.hstack((J_z_pert, J_w_pert[:,None])),
                                   tangent_prev)) - B) / eps
            return dG_dalpha
        elif bif_type == 'NS':
            pass

    elif gtype == 'omega':

        if omega != 0:
            eps = eps * abs(omega)
        omega_pert = omega + eps

        A = assembleA(self, omega/nu)
        J_z_pert = jac_sys(self, z_pert, A) / scale_x

        if bif_type == 'LP2':
            return  (J_z_pert - B ) / eps
        elif bif_type == 'BP':
            J_w_pert = jac_w(self, omega_pert, z_pert)
            dG_dalpha = np.vstack((np.hstack((J_z_pert, J_w_pert[:,None])),
                                   tangent_prev))
            return dG_dalpha
        elif bif_type == 'NS':
            pass

    elif gtype == 'f':
         Hess = np.zeros(B.shape)
         if bif_type == 'LP2' or bif_type == 'BP':
             return Hess
         elif bif_type == 'LP':
             pass
         elif bif_type == 'NS':
             pass


class Bifurcation():
    def __init__(self, hb, max_it_secant=10, tol_sec=1e-5):

        self.max_it_secant = max_it_secant
        self.tol_sec = tol_sec

        self.hb = hb
        self.idx = [0]
        self.nbif = 0
        self.isbif = False

class Fold(Bifurcation):

    def __init__(self,*args, **kwargs):
        Bifurcation.__init__(self, *args, **kwargs)

    def detect(self, omega, z, A, J_z, B, ind_f, force, it_cont):
        """Fold bifurcation traces the lotus of the frequency response peaks

        At a fold bifurcation these four conditions are true:
        * h(z,ω) = 0
        * Rank h_z(z,ω) = nz-1
        * h_ω(z,ω) ∉ range h_z(z,ω). Ie:  Rank [h_z,h_ω] = nz
        * There is a parametrization z(σ) and ω(σ), with z(σ₀) = z₀,
          ω(σ₀) = z₀ and d²ω(σ)/d²σ ≠ 0

        Ie fold bifurcations are detected when h_z is singular and Rank
        [h_z,h_ω] = nz. It is however more efficient to detect fold
        bifurcations, when the component of the tangent vector related to ω
        changes sign.
        """
        nz = self.hb.nz
        max_it_secant = self.max_it_secant
        tol_sec = self.tol_sec
        nu = self.hb.nu
        scale_x = self.hb.scale_x
        n = self.hb.n
        f_amp = self.hb.f_amp
        fdofs = self.hb.fdofs

        G = J_z;
        Gq0 = null_approx(G, 'LP2')
        Gp0 = null_approx(G.T, 'LP2')

        print('-----> Detecting LP bifurcation...')
        H = state_sys(self.hb, z, A, force)
        # really extended_state_sys
        F_it_NR, M_ext, w = test_func(G, Gp0, Gq0, 'LP2', H)

        print('Test function = {}'.format(norm(F_it_NR) / norm(z)))

        it = 1
        while (it < max_it_secant and norm(F_it_NR)/norm(z) > tol_sec):

            J_ext = extended_jacobian5_detect(omega, z, A, B, M_ext, w, ind_f,
                                              'LP2')

            #idx = np.s_[:,:nz,nz]
            delta_NR = pinv(J_ext[:,:nz+1]) @ -F_it_NR
            z = z + delta_NR[:nz]
            omega = omega + delta_NR[nz+1]
            omega2 = omega / nu
            t = assemblet(self.hb, omega2)
            force = sineForce(f_amp, omega, t, n, fdofs)
            A = assembleA(self.hb, omega2)
            J_z = jac_sys(self.hb, z, A) / scale_x
            G = J_z
            Gq0 = null_approx(G, 'LP2')
            Gp0 = null_approx(G.T, 'LP2')
            H = state_sys(self.hb, z, A, force)
            F_it_NR, M_ext, w = test_func(G, Gp0, Gq0, 'LP2', H)
            print('Test function = {}'.format(norm(F_it_NR)))
            it = it + 1

        if it >= max_it_secant:
            print('-----> LP bifurcation detection failed')
        else:
            print('-----> LP detected.')
            self.idx.append(it_cont+1)
            self.nbif += 1
            self.isbif = True

print('\n-------------------------------------------')
print('|  Continuation of the periodic solution  |')
print('-------------------------------------------\n')

it_cont = 1
z_cont = z
omega_cont = omega
point_prev = 0
point_pprev = 0
branch_switch = False
idx_BP = [0]
idx_LP2 = [0]
idx_NS = [0]

if stability:
    detect = {
        'fold':False,#True
        'NS':False,
        'BP':False
    }
    fold = Fold(self, max_it_secant=10, tol_sec=1e-5)
    q0_BP = np.ones(nz+1)
    p0_BP = np.ones(nz+1)
    inl_tmp = np.array([0])
    inl_ext = np.kron(inl_tmp, np.ones(2*NH+1)) + \
        np.kron(np.ones((len(inl_tmp),1)), np.arange(2*NH+1)*n)
    ind_f = 0


relpath = 'data/'
path = abspath + relpath
#mat =  io.loadmat(path + 'hb.mat')


omega2 = omega/nu
# samme A som fra periodic calculation
A = assembleA(self, omega2)
while( it_cont <= it_cont_max  and
       omega / scale_t <= omega_cont_max and
       omega / scale_t >= omega_cont_min ):

    print('[Iteration]:    {}'.format(it_cont))

    ## Predictor step
    J_z = hjac(self, z, A)
    J_w = hjac_omega(self, omega, z)

    # Assemble A from eq. 31
    if it_cont == 1:
        A_pred = np.vstack((
            np.hstack((J_z, J_w[:,None])),
            np.ones(nz+1) ))
    else:
        A_pred = np.vstack((
            np.hstack((J_z, J_w[:,None])),
            tangent))

    # Tangent vector at iteration point, eq. 31 (search direction)
    # tangent = [z, omega].T
    tangent = linalg.solve(A_pred, np.append(np.zeros(nz),1) )
    tangent = tangent/linalg.norm(tangent)
    tangent_pred = cont_dir * step * tangent

    z = z_cont + tangent_pred[:nz]
    omega = omega_cont + tangent_pred[nz]

    point = np.append(z,omega)
    if ( it_cont >= 4 and True and
         ( (point  - point_prev) @ (point_prev - point_pprev) /
           (norm(point - point_prev) * norm(point_prev-point_pprev)) <
           np.cos(angle_max_pred)) ):# and
         # (it_cont >= index_LP+1) and
         # (it_cont >= index_BP+2) ):

        tangent_pred = -tangent_pred
        z = z_cont + tangent_pred[:nz]
        omega = omega_cont + tangent_pred[nz]
        point = np.append(z,omega)
        print('| Max angle reached. Predictor tangent is reversed. |')

    if branch_switch:
        pass

    ## Corrector step. NR iterations
    # Follow eqs. 31-34
    omega2 = omega/nu
    t = assemblet(self, omega2)
    A = assembleA(self, omega2)
    force = sineForce(f_amp, omega, t, n, fdofs)

    it_NR = 1
    V = tangent
    H = state_sys(self, z, A, force)
    Obj = linalg.norm(H) / linalg.norm(z)
    print('It. {} - Convergence test: {:e} ({:e})'.format(it_NR, Obj, tol_NR))

    # break

    # TODO: consider modified NR for increased performance
    # https://stackoverflow.com/a/44710451
    while (Obj > tol_NR ) and (it_NR <= max_it_NR):
        H = state_sys(self, z, A, force)
        H = np.append(H,0)
        J_z = hjac(self, z, A)
        J_w = hjac_omega(self, omega, z)
        Hx = np.vstack((
            np.hstack((J_z, J_w[:,None])),
            V))
        R = np.append(np.hstack((J_z, J_w[:,None])) @ V, 0)

        dNR = linalg.solve(Hx, H)
        dV = linalg.solve(Hx,R)
        #lu, piv = linalg.lu_factor(Hx)
        #dNR = linalg.lu_solve((lu, piv), H)
        #dV = linalg.lu_solve((lu, piv), R)

        dz = dNR[:nz]
        domega = dNR[nz]
        z = z - dz
        omega = omega - domega
        omega2 = omega/nu
        V = (V-dV) / norm(V-dV)
        t = assemblet(self, omega2)
        force = sineForce(f_amp, omega, t, n, fdofs)


        A = assembleA(self, omega2)
        H = state_sys(self, z, A, force)
        Obj = linalg.norm(H) / linalg.norm(z)
        print('It. {} - Convergence test: {:e} ({:e})'.format(it_NR, Obj, tol_NR))
        it_NR = it_NR + 1

    if it_NR > max_it_NR:
        z = z_cont
        omega = omega_cont
        step = step / 10
        print('The maximum number of iterations is reached without convergence.'
              'Step size decreased by factor 10: {:0.2f}'.format(step))
        continue

    if stability:
        if it_NR == 1:
            J_z = hjac(self, z, A)
        B, stab = hills_stability(self, omega, J_z)
        lamb_vec.append(B)
        stab_vec.append(stab)

    if stability and it_cont > 1 and it_cont > idx_BP[-1] + 1:
        G = np.vstack((
            np.hstack((J_z, J_w[:,None])),
            tangent))
        tt_LP = tangent
        t_BP, p0_BP, q0_BP = test_func(G, p0_BP, q0_BP,'BP')
        tt_BP = np.real(t_BP)

        if detect['fold'] and it_cont > fold.idx[-1] + 1:
            fold.detect(omega, z, A, J_z, B, ind_f, force, it_cont)
        # if detect['NS'] and it_cont > idx_NS[-1] + 1:
        #     detect_NS()
        # if detect['BP'] and it_cont > idx_BP[-1] + 1:
        #     detect_bp()


    z_cont = z
    omega_cont = omega
    point_pprev = point_prev
    point_prev = point
    c, phi, _ = hb_components(scale_x*z, n, NH)
    x = hb_signal(omega, t, c, phi)
    xamp = np.max(x, axis=1)
    xamp_vec.append(xamp)
    omega_vec.append(omega)
    step_vec.append(step)
    z_vec.append(z)

    dof = 0
    print(' NR: {}\tFreq: {:f}\tAmp: {:0.3e}\tStep: {:0.2f}\tStable: {}'.
          format(it_NR-1, omega/2/np.pi / scale_t, xamp[dof], step, stab))

    if adaptive_stepsize:
        step = step * opt_it_NR/it_NR
        step = min(step_max, step)
        step = max(step_min, step)


    anim.update(omega_vec,xamp_vec)
    it_cont += 1



# def detect_branchpoint(self, J_z, J_w, tangent):
#     """
#     Branch point bifurcations occur when two branches of periodic solution meet.

#     At a branch bifurcation these four conditions are true:
#     * h(z,ω) = 0
#     * Rank h_z(z,ω) = nz-1
#     * h_ω(z,ω) ∈ range h_z(z,ω). Ie:  Rank [h_z,h_ω] = nz-1
#     * Exactly two branches of periodic solutions intersect with two distinct
#       tangents.

#     """

#     G = np.vstack((
#         np.hstack((J_z, J_w[:,None])),
#         tangent))

#     Gq0 = null_approx(G,'BP')
#     Gp0 = null_approx(G.T,'BP')
#     print('-----> Detecting BP bifurcation...')
#     H = state_sys(self, z, A, force)
#     F_it_NR, M_it_NR, w_it_NR = test_func(G, Gp0, Gq0, 'LP2', H)
#     print('Test function = {}'.format(norm(F_it_NR) / norm(z)))

#     it = 1
#     while (it < max_it_secant and norm(F_it_NR)/norm(z) > tol_sec):
#         J_ext = 1
#         dNR = pinv(J_ext[:,:nz+1]) @ -F_it_NR
#         z = z + dNR[:nz]
#         omega = omega + dNR[nz+1]
#         omega2 = omega / nu
#         t = assemblet(self, omega2)
#         force = sineForce(f_amp, omega, t, n, fdofs)

#         J_w = jac_w(self, omega, z)
#         G = np.vstack((
#             np.hstack((A0, J_w[:,None])),
#             tangent))
#         Gq0 = null_approx(G,'BP')
#         Gp0 = null_approx(G.T,'BP')
#         H = state_sys(self, z, A, force)
#         F_it_NR, M_it_NR, w_it_NR = test_func(G, Gp0, Gq0, 'LP2', H)
#         print('Test function = {}'.format(norm(F_it_NR) / norm(z)))

#         it += 1
#     if it >= max_it_secant:
#         print('-----> BP bifurcation detection failed')
#         z = z_save
#         omega = omega_save
#         A = A_save
#     else:
#         print('-----> BP detected. Computing switching direction...')
#         idx_BP.append(it_cont+1)
#         it_BP += 1
#         is_BP = True

#         J_z = jac_sys(self, z_pert, A)
#         J_w = jac_w(self, omega, z)
#         J_BP = np.vstack((
#             np.hstack((J_z, J_w[:,None])),
#             tangent))
#         phi1 = pinv(J_BP) @ np.append(np.zeros(nz),1)
#         phi1 = phi / norm(phi)
#         psi = null_approx(J_z.T,'BP')
#         phi2 = null_approx(J_BP,'BP')
#         beta2 = 1

#         # perturbation
#         # First pertubate z, then omega
#         eps = 1e-8
#         Hess = np.empty((nz,nz+1,nz+1))
#         # TODO make parallel for loop
#         for i in range(nz):
#             z_pert = z
#             zpert[i] = zpert[i] + eps
#             J_z_pert = jac_sys(self, z_pert, A)
#             J_w_pert = jac_w(self, omega, z_pert)
#             Hess[...,i] = np.hstack(((J_z_pert - J_z)/eps,
#                                      J_w_pert - J_w))
#         omega_pert = omega + eps
#         # TODO should be omega2_pert
#         A = assembleA(self, omega_pert/nu)
#         J_z_pert = jac_sys(self, z, A)
#         J_w_pert = jac_w(self, omega_pert, z)
#         Hess[...,nz] = np.hstack(((J_z_pert - J_z)/eps,
#                                  J_w_pert - J_w))

#         multi_prod12 = np.zeros(nz)
#         multi_prod22 = np.zeros(nz)
#         for i in range(nz):
#             for j in range(nz+1):
#                 for k in range(nz+1):
#                     multi_prod12[i] = multi_prod12[i] + \
#                         Hess[i,j,k] * phi[j] * phi2[k]
#                     multi_prod12[i] = multi_prod12[i] + \
#                         Hess[i,j,k] * phi2[j] * phi2[k]

#         c12 = psi @ multi_prod12
#         c22 = psi @ multi_prod22
#         alpha2 = -1/2 * c22/c12 * beta2
#         Y0_dot = alpha2 * phi1 + beta2 * phi2
#         V0_switch = Y0_dot / norm(Y0_dot)
#         branch_switch = True

# def detect_NS():
#     """Detect Neimark-Sacker(NS) bifurcation.

#     At a NS bifurcation, also called torus bifurcation, quasi periodic
#     oscillation emanates. Quasi periodic oscillations contains the forcing
#     frequency ω and at least one other frequency ω₂(the envelope)
#     These two frequencies are incommensurate, ie ω/ω₂ is irrational.

#     NS bifurcations are detected when the bialternate matrix product of B is
#     singular, B_⊙=B⊙I. The bialternate matrix is singular when two of its
#     eigenvalues μ₁+μ₂=0. For example: μ₁,₂=±jγ

#     """
#     z_save = z
#     omega_save = omega
#     A_save = A

#     mat_B = b2_inv @ b1
#     if rmc_permute:
#         pass

#     Atest, right_vec, B_eig_sort = B_reduc(mat_B, n)
#     right_vec_inv = np.inv(right_vec)
#     G = bialtprod( Atest );
#     Gq0 = null_approx(G, 'NS')
#     Gp0 = null_approx(G.T, 'NS')

#     print('-----> Detecting NS bifurcation...')
#     H = state_sys(self, z, A, force)
#     F_it_NR, M_it_NR, w_it_NR = test_func(G, Gp0, Gq0, 'NS', H)
#     print('Test function = {}'.format(norm(F_it_NR) / norm(z)))

#     it = 1
#     while (it < max_it_secant and norm(F_it_NR)/norm(z) > tol_sec):
#         J_ext = 1
#         dNR = pinv(J_ext[:,:nz+1]) @ -F_it_NR
#         z = z + dNR[:nz]
#         omega = omega + dNR[nz+1]
#         omega2 = omega / nu
#         t = assemblet(self, omega2)
#         force = sineForce(f_amp, omega, t, n, fdofs)
#         A = assembleA(self, omega2)
#         A0 = jac_sys(self, z, A) / scale_x
#         # TODO recalculate Delta1
#         A1 = Delta1
#         #b1 = [ A1, A0; - eye( size( A0 ) ), zeros( size( A0 ) ) ];
#         mat_B = b2_inv @ b1
#         if rmc_permute:
#             pass
#         Atest, right_vec, B_eig_sort = B_reduc(mat_B, n)
#         right_vec_inv = np.inv(right_vec)
#         G = bialtprod(Atest)
#         Gq0 = null_approx(G, 'NS')
#         Gp0 = null_approx(G.T, 'NS')
#         H = state_sys(self, z, A, force)
#         F_it_NR, M_it_NR, w_it_NR = test_func(G, Gp0, Gq0, 'NS', H)
#         print('Test function = {}'.format(norm(F_it_NR) / norm(z)))

#         it += 1
#     if it >= max_it_secant:
#         print( '-----> NS bifurcation detection failed')
#         z = z_save
#         omega = omega_save
#         A = A_save
#     else:
#         print('-----> NS detected.')
#         idx_NS.append(it_cont+1)
#         it_NS += 1
#         is_NS = True

