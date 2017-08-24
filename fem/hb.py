#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from scipy.linalg import block_diag, kron, solve, lstsq
from scipy.linalg import norm
from scipy.fftpack import fft, ifft
from scipy import linalg
from scipy import sparse

import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from scipy import io

def anim_init(omegas, amps):

    dof = 0
    xx = np.asarray(omegas)/scale_t
    yy = np.asarray(amps).T[dof]

    fig = plt.figure(5)
    fig.clf()
    ax = fig.add_subplot(111)
    #ax.clear()
    #fig, ax = plt.subplots(1, 1)
    #ax.hold(True)
    ax.set_title('Nonlinear FRF')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Amplitude (m)')
    ax.set_xlim((25,omega_cont_max/2/np.pi))
    ax.set_ylim((0,np.max(yy)*1.5))
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
    fig.canvas.draw()

    # cache the background
    background = fig.canvas.copy_from_bbox(ax.bbox)

    points = ax.plot(xx/2/np.pi, yy, '-')[0]
    cur_point = ax.plot(xx[-1]/2/np.pi, yy[-1], 'o')[0]
    return (fig, ax, background, points, cur_point)

def anim_update(x,y,  fig, ax, background, points, cur_point): #*kwargs):

    dof = 0
    xx = np.asarray(x)/scale_t
    yy = np.asarray(y).T[dof]
    # print(xx/2/np.pi)
    # fig, ax, background, points = kwargs
    points.set_data(xx/2/np.pi, yy)
    cur_point.set_data(xx[-1]/2/np.pi, yy[-1])

    # recompute the ax.dataLim
    #ax.relim()
    # update ax.viewLim using the new dataLim
    #ax.autoscale_view(scalex=False)
    #ylim = ax.get_ylim()

    ax.set_ylim((0,max(yy)*1.1))

    # restore background
    fig.canvas.restore_region(background)
    # redraw just the points
    ax.draw_artist(points)
    ax.draw_artist(cur_point)
    # fill in the axes rectangle
    fig.canvas.blit(ax.bbox)


def plots():

    dof = 0
    gtype = 'displ'
    # gtype = 'acc'
    if gtype == 'displ':
        y = hb_signal(c, phi, omegap, tp)
        ystr = 'Displacement (m)'
    elif gtype == 'vel':
        y = hb_signal(cd, phid, omegap, tp)
        ystr = 'Velocity (m/s)'
    else:
        y = hb_signal(cdd, phidd, omegap, tp)
        ystr = 'Acceleration (m/s²)'

    fig = plt.figure(1)
    fig_steady = fig
    fig.clf()
    ax = fig.add_subplot(111)
    #fig, ax = plt.subplots()

    ax.plot(tp,y[dof],'-')
    ax.axhline(y=0, ls='--', lw='0.5',color='k')
    ax.set_title('Displacement vs time, dof: {}'.format(dof))
    ax.set_xlabel('Time (t)')
    ax.set_ylabel(ystr)
    # use sci format on y axis when figures are out of the [0.01, 99] bounds
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))

    nh = cnorm.shape[1] -1

    fig = plt.figure(2)
    fig_har = fig
    fig.clf()
    ax = fig.add_subplot(111)
    ax.clear()
    ax.bar(np.arange(nh+1), cnorm[dof])
    ax.set_title('Displacement harmonic component, dof: {}'.format(dof))
    ax.set_xlabel('Order')
    # use double curly braces to "escape" literal curly braces...
    ax.set_ylabel(r'$C_{{{dof}-h}}$'.format(dof=dof))
    ax.set_xlim([-0.5, nh+0.5])
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))



    x = hb_signal(c, phi, omegap, tp)
    xd = hb_signal(cd, phid, omegap, tp)

    fig  =plt.figure(3)
    fig_phase = fig
    fig.clf()
    ax = fig.add_subplot(111)
    ax.plot(x[dof],xd[dof])
    ax.set_title('Phase space, dof: {}'.format(dof))
    ax.set_xlabel('Displacement (m)')
    ax.set_ylabel('Velocity (m/s)')
    ax.ticklabel_format(axis='both', style='sci', scilimits=(-2,2))
    # ax.axis('equal')


    fig  =plt.figure(4)
    fig_stab = fig
    fig.clf()
    ax = fig.add_subplot(111)

    lamb = B
    gtype = 'multipliers'
    gtype = 'exp'
    if gtype == 'multipliers':
        str1 = 'Floquet multipliers'
        str2 = '$\sigma$'
        T = 1/f0
        sigma = np.exp(lamb*T)
        xx = np.real(sigma)
        yy = np.imag(sigma)
        idx_s = np.where(np.abs(sigma) <= 1)
        idx_u = np.where(np.abs(sigma) > 1)

        circ = plt.Circle((0, 0), radius=1, ec='k', fc='None', ls='-')
        ax.add_patch(circ)

    else:
        str1 = 'Floquet exponent'
        str2 = '$\lambda$'
        xx = np.real(lamb)
        yy = np.imag(lamb)
        idx_s = np.where(xx <= 0)
        idx_u = np.where(xx > 0)

        ax.axhline(y=0, color='k')
        ax.axvline(x=0, color='k')

    if len(idx_u[0]) != 0:
        ax.plot(xx[idx_u],yy[idx_u],'ro', label='unstable')
    ax.plot(xx[idx_s],yy[idx_s],'bx', label='stable')
    ax.set_title('Stability ({}), dof: {}'.format(str1, dof))
    ax.set_xlabel(r'Real({})'.format(str2))
    ax.set_ylabel(r'Imag({})'.format(str2))
    ax.legend()

    xmax = max(np.max(np.abs(xx))*1.1, 1.1)
    ymax = max(np.max(np.abs(yy))*1.1, 1.1)
    ax.set_xlim(xmax * np.array([-1,1]))
    ax.set_ylim(ymax * np.array([-1,1]))
    ax.grid(True, which='both')

    relpath = 'plots/hb/'
    path = abspath + relpath
    if inl.size != 0:
        str1 = 'nonlin' + str(len(inl))
    else:
        str1 = 'lin'
    if savefig:
        fig_steady.savefig(path + 'steady_' +str1 + '.png')
        fig_har.savefig(path + 'har_' +str1 + '.png')
        fig_phase.savefig(path + 'phase_' +str1 + '.png')




    # fig = plt.figure(3)
    # ax = fig.add_subplot(111)
    # ax.clear()
    # ax.plot(x,xd)
    # ax.set_title('Configuration space, dof: {}'.format(dof))
    # ax.set_xlabel('Displacement x₁ (m)')
    # ax.set_ylabel('Displacement x₂ (m)')


    plt.show()


def state_sys(z, force):
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

    # n = self.n
    # Nt = self.Nt
    # NH = self.NH
    # scale_x = self.scale_x

    x = ifft_coeff(z, n, Nt, NH)
    fnl = force_nl(x*scale_x)

    b = fft_coeff(force - fnl, NH)
    # eq. 21
    H = A @ z - b
    return H

def hjac(z, force=None):
    """ Computes the jacobian matrix of h wrt Fourier coefficients z, denoted
    h_z.

    From eq. (30)
    h_z = ∂h/∂z = A - ∂b/∂z = A - Γ⁺ ∂f/∂x Γ
    ∂f/∂x is available analytical in time domain

    It is only necessary to calculate the jacobian ∂b/∂z for nonlinear forces
    (ie. it is zero for linear forces).

    """
    if inl.size == 0:
        return A
    else:
        x = ifft_coeff(z, n, Nt, NH)
        dFnl_dx_tot_mat = der_force_nl(x * scale_x) * scale_x
        # the derivative ∂b/∂z
        bjac = np.empty((nZ, nZ))
        full_rhs = dFnl_dx_tot_mat @ mat_func_form_sparse

        for j in range(nZ):
            rhs = np.squeeze(np.asarray(full_rhs[:,j].todense()))
            return_values = sparse.linalg.lsmr(mat_func_form_sparse, rhs)
            x = - return_values[0]
            bjac[:,j] = x

        hjac = A - bjac
        return hjac


def hjac_omega(omega, K, M, C, NH, z):
    """ Calculate the derivative of h wrt ω, h_ω.

    From eq. 32,
    h_ω = ∂A/∂ω*z
    """
    A = np.zeros(K.shape)
    for i in range(1,NH+1):
        blk = np.vstack((
            np.hstack((-2*i**2 * omega * M, -i * C)),
            np.hstack((i * C, -2*i**2 * omega * M)) ))
        A = block_diag( A, blk )

    return A @ z

def der_force_nl(x):
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


def force_nl(x):
    if inl.size == 0:
        return np.array([0])

    # ns = Nt
    ndof, Nt = x.shape
    nbln = inl.shape[0]
    idof = np.arange(ndof)

    fnl = np.zeros((ndof+1, Nt))
    x = np.vstack((x, np.zeros((1,Nt)) ))
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

def fft_coeff(x, NH):
    """ Extract FFT-coefficients from X=fft(x)
    """
    # n: dofs
    n, Nt = x.shape
    # Format of X after transpose: (Nt, n)
    X = fft(x).T / Nt

    re_fft_im_fft = np.hstack([-2*np.imag(X[1:NH+1]),
                               2* np.real(X[1:NH+1])])

    # X[0] only contains real numbers (it is the dc/0-frequency-part), but we
    # still need to extract the real part. Otherwise z is casted to complex128
    z = np.hstack([np.real(X[0]), re_fft_im_fft.ravel()])

    return z

def ifft_coeff(z, n, Nt, NH):
    """ extract iFFT-coefficients from x=ifft(X)
    """

    X = np.zeros((n, Nt), dtype='complex')
    X[:,0] = Nt*z[:n]

    for i in range(NH):
        X[:,i+1] = 2 * (Nt/2   * z[(n*(2*i+1)+n): (n*(2*i+1)+2*n)] - \
                        Nt/2*1j* z[(n*(2*i+1))  : (n*(2*i+1)+n)])

    x = np.real(np.fft.ifft(X))

    return x

def sineforce(A, omega, t, n, Nt):

    phi_f = 0
    phi = phi_f /180 * np.pi

    f = np.zeros((n, Nt))
    # add force to dofs
    for dof in fdofs:
        f[dof] = f[dof] + A * np.sin(omega*t + phi)

    return f


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

    cnorm = np.abs(c) / (eps + np.max(np.abs(c)))

    return c, cnorm, phi

def hb_signal(c, phi, omega, t):
    n = c.shape[0]
    NH = c.shape[1]-1
    Nt = len(t)
    tt = np.arange(1,NH+1)[:,None] * omega * t

    x = np.zeros((n, Nt))
    for i in range(n):

        tmp = tt + np.outer(phi[i,1:],np.ones(Nt))
        tmp = c[i,0]*np.ones(Nt) + c[i,1:] @ np.sin(tmp)
        x[i] = tmp #np.sum(tmp, axis=0)

    return x

def B_reduc(B_tilde, n):

    w = linalg.eigvals(B_tilde)
    idx = np.argsort(np.abs(np.imag(w)))[:2*n]

    return w[idx]

def assembleA(omega2,M,C,K,NH):
    """Assemble A, describing the linear dynamics. eq (20)
    """

    A = K
    for i in range(1,NH+1):
        blk = np.vstack((
            np.hstack((K - ( i * omega2 )**2 * M, -i * omega2 * C)),
            np.hstack((i * omega2 * C, K - (i * omega2)**2 * M)) ))
        A = block_diag( A, blk )
    return A

def assemblet(omega2):
    return np.arange(2**npow2) / 2**npow2 * 2*np.pi / omega2

# parameters
#def __init__(scale_x, scale_t):
"""
Parameters
----------
because frequency(in rad/s) and amplitude have different orders of magnitude,
time and displacement have to be rescaled to avoid ill-conditioning.
"""

abspath='/home/paw/ownCloud/speciale/code/python/vib/'
relpath = 'data/T05_Data/'
path = abspath + relpath

savefig = True
savefig = False

mat =  io.loadmat(path + 'NLBeam.mat')

inl = np.array([[27,-1], [27,-1]])
# inl = np.array([])
enl = np.array([3,2])
knl = np.array([8e9,-1.05e7])

scale_x = 5e-6
scale_t = 3000

## Force parameters
# location of harmonic force
fdofs = np.array([7])
f_amp = 3

## HB parameters
# number of harmonic retained in the Fourier series
NH = 5
# number of time samples in the Fourier transform, ie. 2^8=256 samples
npow2 = 9
# Excitation frequency. lowest sine freq in Hz
f0 = 25
# nu accounts for subharmonics of excitation freq w0
nu = 1
# amplitude of first guess
amp0 = 1e-3
stability = True
rcm_permute = False

tol_NR = 1e-6 * scale_x  # == 5e-12
max_it_NR = 15

M0 = mat['M']
C0 = mat['C']
K0 = mat['K']
M = M0 * scale_x / scale_t**2
C = C0 * scale_x / scale_t
K = K0 * scale_x

n = M0.shape[0]
w0 = f0 *2*np.pi
omega = w0*scale_t
omega2 = omega / nu

t = np.arange(2**npow2) / 2**npow2 * 2*np.pi / omega2
Nt = len(t)
# number of unknowns in Z-vector, eq (4)
nZ = n * (2 * NH + 1)
force = sineforce(f_amp, omega, t, n, Nt)

# Assemble A, describing the linear dynamics. eq (20)
A = assembleA(omega2,M,C,K,NH)

# Form Q(t), eq (8). Q is orthogonal trigonometric basis(harmonic terms)
# Then use Q to from the kron product Q(t) ⊗ Iₙ, eq (6)
mat_func_form = np.empty((n*Nt, nZ))
Q = np.empty((NH*2+1,1))
for i in range(Nt):
    Q[0] = 1
    for ii in range(1,NH+1):
        Q[ii*2-1] = np.sin(omega * t[i]*ii)
        Q[ii*2] = np.cos(omega * t[i]*ii)

    # Stack the kron prod, so each block row is for time(i)
    mat_func_form[i*n:(i+1)*n,:] = kron(Q.T, np.eye(n))
mat_func_form_sparse = sparse.csr_matrix(mat_func_form)


# Assemble Δs of the linear eigenvalue problem, eq. 44.
# B2 is assembled now. B1 requires hz, thus assembled after sol is found
if stability:
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


# Initial guess for x. Here calculated by broadcasting. np.outer could be used
# instead to calculate the outer product
amp = np.ones(n)*amp0
x_guess = amp[:,None] * np.sin(omega * t)/scale_x


# Initial guess for z, z = Γ⁺x as given by eq (26). Γ is found as Q(t)⊗Iₙ, eq 6

# Both lstsq and pinv computes a least-square solution. The difference seems to
# be the way z is  normalized. Both z is computed such that the 2-norm |x - Γz|
# is minimized.
# A word of warning: The pseudoinverse of a sparse matrix will typically be
# fully dense. We only want the pseudo-inverse for finding a minimum norm of a
# least square solution. Thus we should never calculate pinv explicitly, but
# use lstsq to find a iterative solution.
# For lstsq, z is computed such that it haves the fewest possible nonzero'es.
# For pinv, z is computed such that the norm |z| is minimized (in the case of
# under-determined systems, ie not full rank) The pseudo-inverse exists for any
# matrix A. When A has full rank, the inverse can be expressed as a simple
# algebraic formula. Here for linear independent rows (as given by the article)
# (It is different for linear independent columns):
# A⁺ = Aᵀ(AAᵀ)⁻¹
z_guess, *_ = lstsq(mat_func_form, x_guess.ravel())
# Ztronc =linalg.pinv(mat_func_form) @ x_guess.ravel()


# Solve h(z,ω)=A(ω)-b(z)=0 (ie find z that is root of this eq), eq. (21)
# NR solution: (h_z is the derivative of h wrt z)
# zⁱ⁺¹ = zⁱ - h(zⁱ,ω)/h_z(zⁱ,ω)
# h_z = A(ω) - b_z(z) = A - Γ⁺ ∂f/∂x Γ
# where the last exp. is in time domain. df/dx is thus available analytical.
# See eq (30)
print('Newton-Raphson iterative solution')

relpath = 'data/'
path = abspath + relpath
mat =  io.loadmat(path + 'jac_sys.mat')

Obj = 1
it_NR = 1
# machine precision for float
eps = np.finfo(float).eps
z = z_guess

while (Obj > tol_NR ) and (it_NR <= max_it_NR):
    H = state_sys(z, force)
    jac_sys = hjac(z)

    zsol, *_ = lstsq(jac_sys,H)
    z = z - zsol

    # print('H', norm(H))
    # print('jac_sys', norm(jac_sys))
    # print('zsol',norm(zsol))

    Obj = linalg.norm(H) / (eps + linalg.norm(z))
    print('It. {} - Convergence test: {:e} ({:e})'.format(it_NR, Obj, tol_NR))
    it_NR = it_NR + 1

if it_NR > max_it_NR:
    print('Number of iterations exceeded {}. Change Harmonic Balance'
          'parameters.'.format(max_it_NR))
    #raise ValueError("""Number of iterations exceeded {}. Change Harmonic Balance
    #parameters.""".format(max_it_NR))


if stability:
    # eq. 45
    A0 = jac_sys/scale_x
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


### ONly needed for single hb solution
c, cnorm, phi = hb_components(scale_x*z, n, NH)

iw = np.arange(NH+1) * omega2
cd = c*iw
cd[:,0] = 0
phid = phi + np.pi/2

cdd = -c * iw**2
cdd[:,0] = 0
phidd = phi

tp = t*scale_t
# make t one full period + dt, ie include first point of next period
tp = np.append(tp, tp[-1] + tp[1] - tp[0])
omegap = omega/scale_t
omega2p = omega2/scale_t
zp = z*scale_x

plt.ion()
plots()
### END

### Continuation
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

# amplitude of hb components
c, cnorm, phi = hb_components(scale_x*z, n, NH)
x = hb_signal(c, phi, omega, t)
xamp = np.max(x, axis=1)

xamp_vec.append(xamp)
omega_vec.append(omega)
stab_vec.append(stab)
lamb_vec.append(B)

anim = anim_init(omega_vec, xamp_vec)
fig, ax, background, points, cur_point = anim

print('\n-------------------------------------------')
print('|  Continuation of the periodic solution  |')
print('-------------------------------------------\n')


it_last_turn = 0
it_cont = 1
z_cont = z
omega_cont = omega
point_prev = 0
point_pprev = 0
branch_switch = False

relpath = 'data/'
path = abspath + relpath
#mat =  io.loadmat(path + 'hb.mat')

while( it_cont <= it_cont_max  and
       omega / scale_t < omega_cont_max and
       omega / scale_t > omega_cont_min ):

    print('[Iteration]:    {}'.format(it_cont))

    ## Predictor step
    J_z = hjac(z)
    J_w = hjac_omega(omega, K, M, C, NH, z)

    # Assemble A from eq. 31
    if it_cont == 1:
        A_pred = np.vstack((
            np.hstack((J_z, J_w[:,None])),
            np.ones(nZ+1) ))
    else:
        A_pred = np.vstack((
            np.hstack((J_z, J_w[:,None])),
            tangent))

    # Tangent vector at iteration point, eq. 31 (search direction)
    # tangent = [z, omega].T
    tangent = linalg.solve(A_pred, np.append(np.zeros(nZ),1) )
    tangent = tangent/linalg.norm(tangent)
    tangent_pred = cont_dir * step * tangent

    z = z_cont + tangent_pred[:nZ]
    omega = omega_cont + tangent_pred[nZ]

    point = np.append(z,omega)
    if ( it_cont >= 4 and False and
         ( (point  - point_prev) @ (point_prev - point_pprev) /
           (norm(point - point_prev) * norm(point_prev-point_pprev)) <
           np.cos(angle_max_pred)) ):# and
         # (it_cont >= index_LP+1) and
         # (it_cont >= index_BP+2) ):

        it_last_turn = it_cont

        tangent_pred = -tangent_pred
        z = z_cont + tangent_pred[:nZ]
        omega = omega_cont + tangent_pred[nZ]
        print('| Max angle reached. Predictor tangent is reversed. |')

    if branch_switch:
        pass

    ## Corrector step. NR iterations
    # Follow eqs. 31-34
    omega2 = omega/nu
    t = assemblet(omega2)
    A = assembleA(omega2,M,C,K,NH)
    force = sineforce(f_amp, omega, t, n, Nt)

    it_NR = 1
    V = tangent
    H = state_sys(z, force)
    Obj = linalg.norm(H) / linalg.norm(z)
    #print('It. {} - Convergence test: {:e} ({:e})'.format(it_NR, Obj, tol_NR))

    # TODO: consider modified NR for increased performance
    # https://stackoverflow.com/a/44710451
    while (Obj > tol_NR ) and (it_NR <= max_it_NR):
        H = state_sys(z, force)
        H = np.append(H,0)
        J_z = hjac(z)
        J_w = hjac_omega(omega, K, M, C, NH, z)
        Hx = np.vstack((
            np.hstack((J_z, J_w[:,None])),
            V))
        R = np.append(np.hstack((J_z, J_w[:,None])) @ V, 0)

        dNR = linalg.solve(Hx, H)
        dV = linalg.solve(Hx,R)
        #lu, piv = linalg.lu_factor(Hx)
        #dNR = linalg.lu_solve((lu, piv), H)
        #dV = linalg.lu_solve((lu, piv), R)

        dz = dNR[:nZ]
        domega = dNR[nZ]
        z = z - dz
        omega = omega - domega
        omega2 = omega/nu
        V = (V-dV) / norm(V-dV)
        t = assemblet(omega2)
        force = sineforce(f_amp, omega, t, n, Nt)


        A = assembleA(omega2,M,C,K,NH)
        H = state_sys(z, force)
        Obj = linalg.norm(H) / linalg.norm(z)
        #print('It. {} - Convergence test: {:e} ({:e})'.format(it_NR, Obj, tol_NR))
        it_NR = it_NR + 1

    if it_NR > max_it_NR:
        print('The maximum number of iterations is reached without convergence.'
              'Step size decreased by factor 10')
        z = z_cont
        omega = omega_cont
        step = step / 10
        continue

    if stability:
        pass

    z_cont = z
    omega_cont = omega
    point_pprev = point_prev
    point_prev = point
    c, _, phi = hb_components(scale_x*z, n, NH)
    x = hb_signal(c, phi, omega, t)
    xamp = np.max(x, axis=1)
    xamp_vec.append(xamp)
    omega_vec.append(omega)
    step_vec.append(step)

    dof = 0
    print(' NR: {}\tFreq: {:f}\tAmp: {:0.3e}\tStep: {:0.2f}'.
          format(it_NR-1, omega/2/np.pi / scale_t, xamp[dof], step))

    anim_update(omega_vec,xamp_vec, fig, ax, background, points, cur_point)

    if adaptative_stepsize:
        step = step * opt_it_NR/it_NR
        step = min(step_max, step)
        step = max(step_min, step)

    it_cont += 1
