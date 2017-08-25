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
            Delta1, Delta2, b2_inv = hills_stability_init(self, omega2)

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
             B, stab = hills_stability(self, H_z, Delta1, Delta2, b2_inv)

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

def hills_stability_init(self, omega2):
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

    # eq. 38
    Delta1 = C0
    for i in range(1,NH+1):
        blk = np.vstack((
            np.hstack((C0, - 2*i * omega2/scale_t * M0)),
            np.hstack((2*i * omega2/scale_t * M0, C0)) ))
        Delta1 = block_diag(Delta1, blk)

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

    return Delta1, Delta2, b2_inv

def hills_stability(self, H_z, Delta1, Delta2, b2_inv):
    scale_x = self.scale_x
    n = self.n
    rcm_permute = self.rcm_permute

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




abspath='/home/paw/ownCloud/speciale/code/python/vib/'
relpath = 'data/T05_Data/'
path = abspath + relpath
mat =  io.loadmat(path + 'NLBeam.mat')
M0 = mat['M']
C0 = mat['C']
K0 = mat['K']

f_amp = 3
# Excitation frequency. lowest sine freq in Hz
f0 = 25

par_hb ={
    'NH': 5,
    'npow2': 9,
    'nu': 1,
    'stability': True,
    'rcm_permute': False,
    'tol_NR': 1e-6,
    'max_it_NR': 15,
    'scale_x': 5e-6, # == 5e-12
    'scale_t': 3000
}

par_cont = {
    'omega_min': 0.001,
    'omega_max': 40,
    'cont_dir': 1,
    'opt_it_NR': 3,
    'step': 0.1,
    'step_min': 0.1,
    'step_max': 20,
    'angle_max_pred': 90,
    'it_cont_max': 1e4,
    'adaptive_stepsize': True
}


inl = np.array([[27,-1], [27,-1]])
# inl = np.array([])
enl = np.array([3,2])
knl = np.array([8e9,-1.05e7])
nonlin = (inl, knl, enl)
# machine precision for float
eps = np.finfo(float).eps

## Force parameters
# location of harmonic force
fdofs = 7
f0 = 25
f_amp = 3

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
from hb_plots import anim_init, anim_update

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

omega_cont_min = 0.001*2*np.pi #par['omega_cont_min']
omega_cont_max = 40*2*np.pi #par['omega_cont_max']
cont_dir = 1 #par['cont_direction']
opt_it_NR = 3 #par['opt_it_NR']
step = 0.1 #scale_t * 0.1 #par['stepsize']
step_min = scale_t * 0.1 #par['stepsize_min']
step_max = scale_t * 20 #par['stepsize_max']
angle_max_pred = 90*np.pi/180 #par['angle_max_pred']
it_cont_max = 1e4 #par['it_cont_max']
adaptative_stepsize = True

xamp_vec =[]
omega_vec = []
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

anim = anim_init(omega_vec, xamp_vec, scale_t, omega_cont_max)


print('\n-------------------------------------------')
print('|  Continuation of the periodic solution  |')
print('-------------------------------------------\n')

it_cont = 1
z_cont = z
omega_cont = omega
point_prev = 0
point_pprev = 0
branch_switch = False

relpath = 'data/'
path = abspath + relpath
#mat =  io.loadmat(path + 'hb.mat')

omega2 = omega/nu
A = assembleA(self, omega2)
while( it_cont <= it_cont_max  and
       omega / scale_t < omega_cont_max and
       omega / scale_t > omega_cont_min ):

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
        pass

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

    dof = 0
    print(' NR: {}\tFreq: {:f}\tAmp: {:0.3e}\tStep: {:0.2f}'.
          format(it_NR-1, omega/2/np.pi / scale_t, xamp[dof], step))

    if adaptative_stepsize:
        step = step * opt_it_NR/it_NR
        step = min(step_max, step)
        step = max(step_min, step)


    anim_update(omega_vec,xamp_vec, scale_t, *anim)
    it_cont += 1
