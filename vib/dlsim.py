#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
dnlsys -- a collection of classes and functions for modeling nonlinear
linear state space systems.
"""

from scipy.interpolate import interp1d
from numpy.fft import fft

class StateSpace(object):
    def __init__(self, A,B,C,D, **kwargs):
        """Initialize the state space lti/dlti system."""

        self.A, self.B, self.C, self.D = A, B, C, D

        self.n = self.A.shape[0]
        self.m = self.B.shape[1]
        self.p = self.C.shape[0]

        self.T1 = None
        self.T2 = None

        self.dt = 0.1

    def nlterms(self, eq, degree, structure):
        """Set active nonlinear terms/monomials to be optimized"""
        if eq in ('state', 'x'):
            self.xdegree = np.asarray(degree)
            self.xstructure = structure
            # all possible terms
            self.xpowers = combinations(self.n+self.m, degree)
            self.n_nx = self.xpowers.shape[0]
            self.xactive = \
                select_active(self.xstructure,self.n,self.m,self.n,self.xdegree)
            self.E = np.zeros((self.n, self.n_nx))
        elif eq in ('output', 'y'):
            self.ydegree = np.asarray(degree)
            self.ystructure = structure
            self.ypowers = combinations(self.n+self.m, degree)
            self.n_ny = self.ypowers.shape[0]
            self.yactive = \
                select_active(self.ystructure,self.n,self.m,self.p,self.ydegree)
            self.F = np.zeros((self.p, self.n_ny))

    # def transient(self, t1, t2=None):
    #     """Transient handling. t1: periodic, t2: aperiodic"""

    def simulate(self, u, t=None, x0=None, T1=None, T2=None):
        """Calculate the output and the states of a nonlinear state-space model with
        transient handling.

        """
        self.T1 = T1
        self.T2 = T2

        # Number of samples
        ns = u.shape[0]
        if self.T1 is not None:
            # Prepend transient samples to the input
            idx = transient_indices_periodic(self.T1, ns)
            u = u[idx]

        t, y, x = dnlsim(self, u, t, x0)

        if self.T1 is not None:
            # remove prepended transient samples
            idx = remove_transient_indices_periodic(self.T1, ns, p=1)
            x = x[idx]
            y = y[idx]

        return t, y, x


# https://github.com/scipy/scipy/blob/master/scipy/signal/ltisys.py
def dnlsim(system, u, t=None, x0=None):
    """Simulate output of a discrete-time nonlinear system.

	Calculate the output and the states of a nonlinear state-space model.
        x(t+1) = A x(t) + B u(t) + E zeta(x(t),u(t))
        y(t)   = C x(t) + D u(t) + F eta(x(t),u(t))
    where zeta and eta are polynomials whose exponents are given in xpowers and
    ypowers, respectively. The maximum degree in one variable (a state or an
    input) in zeta or eta is given in max_nx and max_ny, respectively. The
    initial state is given in x0.

    """

    u = np.asarray(u)

    if u.ndim == 1:
        u = np.atleast_2d(u).T

    if t is None:
        out_samples = len(u)
        stoptime = (out_samples - 1) * system.dt
    else:
        stoptime = t[-1]
        out_samples = int(np.floor(stoptime / system.dt)) + 1

    # Pre-build output arrays
    xout = np.empty((out_samples, system.A.shape[0]))
    yout = np.empty((out_samples, system.C.shape[0]))
    tout = np.linspace(0.0, stoptime, num=out_samples)

    # Check initial condition
    if x0 is None:
        xout[0, :] = np.zeros((system.A.shape[1],))
    else:
        xout[0, :] = np.asarray(x0)

    # Pre-interpolate inputs into the desired time steps
    if t is None:
        u_dt = u
    else:
        if len(u.shape) == 1:
            u = u[:, np.newaxis]

        u_dt_interp = interp1d(t, u.transpose(), copy=False, bounds_error=True)
        u_dt = u_dt_interp(tout).transpose()

    # prepare nonlinear part
    repmat_x = np.ones(system.xpowers.shape[0])
    repmat_y = np.ones(system.ypowers.shape[0])
    # Simulate the system
    for i in range(0, out_samples - 1):
        # State equation x(t+1) = A*x(t) + B*u(t) + E*zeta(x(t),u(t))
        zeta_t = np.prod(np.outer(repmat_x, np.hstack((xout[i], u_dt[i])))
                          **system.xpowers, axis=1)
        xout[i+1, :] = (np.dot(system.A, xout[i, :]) +
                        np.dot(system.B, u_dt[i, :]) +
                        np.dot(system.E, zeta_t))
        # Output equation y(t) = C*x(t) + D*u(t) + F*eta(x(t),u(t))
        eta_t = np.prod(np.outer(repmat_x, np.hstack((xout[i], u_dt[i])))
                        **system.ypowers, axis=1)
        yout[i, :] = (np.dot(system.C, xout[i, :]) +
                      np.dot(system.D, u_dt[i, :]) +
                      np.dot(system.F, eta_t))

    # Last point
    eta_t = np.prod(np.outer(repmat_x, np.hstack((xout[-1], u_dt[-1])))
                     **system.ypowers, axis=1)
    yout[-1, :] = (np.dot(system.C, xout[-1, :]) +
                   np.dot(system.D, u_dt[-1, :]) +
                   np.dot(system.F, eta_t))

    return tout, yout, xout


def poly_deriv(powers):
    """Calculate derivative of a multivariate polynomial

    """
    # Polynomial coefficients of the derivative
    d_coeff = powers[:,None]
    n = powers.shape[1]
    #  Terms of the derivative
    d_powers = np.repeat(powers[...,None],n, axis=2)
    for i in range(n):
        # Derivative w.r.t. variable i has one degree less in variable i than
        # original polynomial If original polynomial is constant w.r.t.
        # variable i, then the derivative is zero, but take abs to avoid a
        # power -1 (zero coefficient anyway)
        d_powers[:,i,i] = np.abs(powers[:,i]-1)

        # TODO
        # This would be more correct, but is slower
        # d_powers(:,i,i) = powers(:,i) - 1;
        # d_powers(powers(:,i) == 0,:,i) = 0;

    return d_powers, d_coeff


def multEdwdx(contrib, power, coeff, E, n):
    """Multiply a matrix E with the derivative of a polynomial w(x,u) wrt. x

    Multiplies a matrix E with the derivative of a polynomial w(x,u) wrt the n
    elements in x. The samples of x and u are in a vector contrib. The
    derivative of w(x,u) w.r.t. x is given by the exponents in x and u (given
    in power) and the corresponding coefficients (given in coeff). The maximum
    degree of a variable (an x or a u) in w(x,u) is given in nx.

	Returns
    -------
	out : ndarray(n_out,n,N)
        Product of E and the derivative of the polynomial w(x,u) w.r.t. the
        elements in x at all samples.

	Parameters
    ----------
	contrib : ndarray(n+m,N)
        N samples of the signals x and u
    power : ndarray(n_nx,n+m,n+m)
        The exponents of the derivatives of w(x,u) w.r.t. x and u, i.e.
        power(i,j,k) contains the exponent of contrib j in the derivative of
        the ith monomial w.r.t. contrib k.
    coeff : ndarray(n_nx,n+m)
        The corresponding coefficients, i.e. coeff(i,k) contains the
        coefficient of the derivative of the ith monomial in w(x,u) w.r.t.
        contrib k.
    E : ndarray(n_out,n_nx)
    nx : int
        maximum degree of a variable (an x or a u) in w(x,u)
    n : int
        number of x signals w.r.t. which derivatives are taken

	Example:
       % Consider w(x1,x2,u) = [x1^2    and E = [1 3 5
       %                        x1*x2;           2 4 6]
       %                        x2*u^2]
       % then the derivatives of E*w w.r.t. x1 and x2 are given by
       % E*[2*x1 0
       %    1*x2 1*x1
       %    0    1*u^2]
       % and the derivative of w w.r.t. u is given by [0
       %                                               0
       %                                               2*x2*u]
    E = [1 3 5; 2 4 6];
       pow = zeros(3,3,3);
       pow(:,:,1) = [1 0 0;
                     0 1 0;
                     0 0 0]; % Derivative w.r.t. x1 has terms 2*x1, 1*x2, and 0
       pow(:,:,2) = [0 0 0;
                     1 0 0;
                     0 0 2]; % Derivative w.r.t. x2 has terms 0, 1*x1, and 1*u^2
       pow(:,:,3) = [0 0 0;
                     0 0 0;
                     0 1 1]; % Derivative w.r.t. u has terms 0, 0, and 2*x2*u
       coeff = [2 0 0;
                1 1 0;
                0 1 2];
       nx = 2; % Maximum second degree factor in monomials of w (x1^2 in first monomial, u^2 in third monomial)
       n = 2; % Two signals x
       contrib = randn(3,10); % Ten random samples of signals x1, x2, and u
       out = fEdwdx(contrib,pow,coeff,E,nx,n);
       % => out(:,:,t) = E*[2*contrib(1,t) 0
       %                    1*contrib(2,t) 1*contrib(1,t)
       %                    0              1*contrib(3,t)^2]

    """

    # n_all = number of signals x and u; N = number of samples
    n_all, N = contrib.shape
    # n_out = number of rows in E; n_nx = number of monomials in w
    n_out, n_nx =E.shape
    out = np.zeros((n_out,n,N))
    # Loop over all signals x w.r.t. which derivatives are taken
    for k in range(n):
        # Repeat coefficients of derivative of w w.r.t. x_k
        A = np.outer(coeff[:,k], np.ones(N))
        for j in range(n_all):     # Loop over all signals x and u
            for i in range(n_nx):  # Loop over all monomials
                # Derivative of monomial i wrt x_k
                A[i,:] *= contrib[j,:]**power[i,j,k]
        # E times derivative of w wrt x_k
        out[:,k,:] = np.matmul(E,A)

    return out

def nl_terms(contrib,power):
    """Construct polynomial terms.

    Computes polynomial terms, where contrib contains the input signals to the
    polynomial and pow contains the exponents of each term in each of the
    inputs. The maximum degree of an individual input is given in max_degree.

	Parameters
    ----------
	contrib : ndarray(n+m,N)
        matrix with N samples of the input signals to the polynomial.
        Typically, these are the n states and the m inputs of the nonlinear
        state-space model.

    power : ndarray(nterms,n+m
        matrix with the exponents of each term in each of the inputs to the polynomial
    max_degree : int
        maximum degree in an individual input of the polynomial

    Returns
    -------
	out : ndarray(nterms,N)
        matrix with N samples of each term

	Example:
		n = 2; % Number of states
       m = 1; % Number of inputs
       N = 1000; % Number of samples
       x = randn(n,N); % States
       u = randn(m,N); % Input
       contrib = [x; u]; % States and input combined
       pow = [2 0 0;
              1 1 0;
              1 0 1;
              0 2 0;
              0 1 1;
              0 0 2]; % All possible quadratic terms in states and input: x1^2, x1*x2, x1*u, x2^2, x2*u, u^2
       max_degree = max(max(pow)); % Maximum degree in an individual state or input
       out = fTermNL(contrib,pow,max_degree);
       % => out = [x(1,:).^2;
       %           x(1,:).*x(2,:);
       %           x(1,:).*u;
       %           x(2,:).^2;
       %           x(2,:).*u;
       %           u.^2];
    """

    # Number of samples
    N = contrib.shape[1]
    # Number of terms
    nterms = power.shape[0]
    out = np.empty((nterms,N))
    for i in range(nterms):
        # All samples of term i
	    out[i] = np.prod(contrib**power.T[:,None,i], axis=0)

    return out


def element_jacobian(samples, Edwdx, C, Fdwdx, active):
    """Compute Jacobian of the output y wrt. A, B, and E

    The Jacobian is calculated by filtering an alternative state-space model

    See fJNL

    """
    p, n = C.shape  # Number of outputs and number of states
    # Number of samples and number of inputs in alternative state-space model
    N, npar = samples.shape
    nactive = len(active) # Number of active parameters in A, B, or E

     # Add C to F partial(eta(t))/partial(x(t)) for all samples at once
    Fdwdx += C[...,None]

    out = np.zeros((p,N,nactive))
    for k in range(nactive):
        # Which column in A, B, or E matrix
        j = np.mod(active[k], npar)
        # Which row in A, B, or E matrix
        i = (active[k]-j)//npar
        # partial derivative of x(0) wrt. A(i,j), B(i,j), or E(i,j)
        Jprev = np.zeros(n)
        for t in range(1,N):
            # Calculate state update alternative state-space model at time t
            # Terms in alternative states at time t-1
            J = Edwdx[:,:,t-1] @ Jprev
            # Term in alternative input at time t-1
            J[i] += samples[t-1,j]
            # Calculate output alternative state-space model at time t
            out[:,t,k] = Fdwdx[:,:,t] @ J
            # Update previous state alternative state-space model
            Jprev = J

    return out

def analytical_jacobian(system):

    """Compute the Jacobians in a steady state nonlinear state-space model

    Jacobians of a nonlinear state-space model

        x(t+1) = A x(t) + B u(t) + E zeta(x(t),u(t))
        y(t)   = C x(t) + D u(t) + F eta(x(t),u(t))

    i.e. the partial derivatives of the modeled output w.r.t. the active
    elements in the A, B, E, F, D, and C matrices

    """

    # Collect states and outputs with prepended transient sample
    contrib = np.hstack((x_trans, u_trans)).T

    # E partial(zeta(t))/partial(x(t)) + A(n,n,NT)
    A_EdwxIdx = multEdwdx(contrib,system.xd_powers,np.squeeze(system.xd_coeff),
                          system.E,system.n) + system.A[...,None]
    zeta = nl_terms(contrib, system.xpowers).T  # (NT,n_nx)

    # F partial(eta(t))/partial(x(t)) (p,n,NT)
    FdwyIdx = multEdwdx(contrib,system.yd_powers,np.squeeze(system.yd_coeff),
                        system.F,system.n)
    eta = nl_terms(contrib, system.ypowers).T  # (NT,n_ny)

    # calculate jacobians wrt state space matrices
    JF = np.kron(np.eye(system.p), eta)  # Jacobian wrt all elements in F
    JF = JF[:,system.yactive]  # all active elements in F. (p*NT,nactiveF)
    JF = JF[idx_utrans]  # (p*N,nactiveF)

    JD = np.kron(np.eye(system.p), um)  # (p*N, p*m)
    JC = np.kron(np.eye(p), x)  # (p*N,p*n)

    # calculate Jacobian by filtering an alternative state-space model
    JE = element_jacobian(zeta, A_EdwxIdx, system.C, FdwyIdx, system.xactive)
    JE = JE.transpose((1,0,2)).reshape((system.p*n_trans, len(system.xactive)))
    JE = JE[idx_utrans]  # (p*N,nactiveE)

    JA = element_jacobian(x_trans, A_EdwxIdx, system.C, FdwyIdx,
                          np.arange(system.n**2))
    JA = JA.transpose((1,0,2)).reshape((system.p*n_trans, len(system.xactive)))
    JA = JA[idx_utrans]  # (p*N,n**2)

    JB = element_jacobian(u_trans, A_EdwxIdx, system.C, FdwyIdx,
                          np.arange(system.n+system.m))
    JB = JB.transpose((1,0,2)).reshape((system.p*n_trans, system.n*system.m))
    JB = JB[idx_utrans]  # (p*N,n*m)

    J = np.hstack((JA, JB, JC, JD, JE, JF))[without_T2]


model = StateSpace(A_data, B_data, C_data, D_data)
system = model
model.nlterms('x', [2,3], 'full')
model.nlterms('y', [2,3], 'full')

u = data['u']
y = data['y']

# samples per period
npp = u.shape[0]
# average time signal over periods and flatten
um = u.mean(axis=-1)  # (N,m,R)
ym = y.mean(axis=-1)
um = um.swapaxes(1,2).reshape(-1,m, order='F')  # (N*R,m)
ym = ym.swapaxes(1,2).reshape(-1,p, order='F')  # (N*R,p)

# transient settings
nt = ns
T1 = np.r_[nt, np.r_[0:(R-1)*npp+1:npp]]

# Compute the (transient-free) modeled output and the corresponding states
t_mod, y_mod, x_mod = model.simulate(um, T1=T1)

covYinvsq = np.empty_like(covY)
for f in range(F):
    covYinvsq[f] = _matrix_square_inv(covY[f])

weight = covYinvsq


N = um.shape[0]

# Number of frequency bins where weighting is specified (ie nfd = floor(npp/2),
# where npp is the number of samples in one period and one phase realization
# for a multisine excitation
nfd = weight.shape[0]

# TODO this only works for T2 scalar.
# Determine if weighting is in frequency or time domain: only implemented for
# periodic signals.
if weight is None:
    freq_weight = False
    weight = np.ones((N,p))
elif nfd > 1:
    freq_weight = True
    if system.T2 is not None:
        R = round((N-T2)/nfd/2)
        if np.mod(N-T2,R) != 0:
            raise ValueError('Transient handling and weighting matrix are incompatible. T2 {}'.
                             format(T2))
else:
    # time domain
    freq_weight = False

# TODO fix transient
# without_T2 = remove_transient_indices_nonperiodic(system.T2,N,system.p)
without_T2 = np.s_[:N]
# Compute the (weighted) error signal without transient
err_old = ylin[without_T2] - ym[without_T2]
if freq_weight:
    err_old = err_old.reshape(((N-T2)//R,R,p)).swapaxes(1,2)
    err_old = fft(err_old, axis=0)
    # Select only the positive half of the spectrum
    err_old = err_old[:nfd]
    err_old = np.matmul(weight, err_old)
    cost = np.vdot(err_old, err_old).real
    err_old = err_old.swapaxes(1,2).ravel(order='F')
    err_old = np.hstack((err_old.real, err_old.imag))
else:
    err_old = err_old * weight[without_T2]
    cost = np.dot(err_old,err_old)

# Compute the derivatives of the polynomials zeta and e
system.xd_powers, system.xd_coeff = poly_deriv(system.xpowers)
system.yd_powers, system.yd_coeff = poly_deriv(system.ypowers)

# Extract the transient part of the input and states
idx_trans = transient_indices_periodic(system.T1, N)
idx_utrans = remove_transient_indices_periodic(system.T1, N, system.p)
u_trans = um[idx_trans]
n_trans = u_trans.shape[0]
x_trans = x[idx_trans]

J = analytical_jacobian(system)



