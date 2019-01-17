#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pyvib.statespace import StateSpace as linss
from pyvib.statespace import Signal
from pyvib.pnlss import PNLSS
from pyvib.common import db
from pyvib.forcing import multisine
from scipy.linalg import norm
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

# generate model to estimate
A = np.array([[0.73915535, -0.62433133],[ 0.6247377,  0.7364469]])
B = np.array([[0.79287245], [-0.34515159]])
C = np.array([[0.71165154, 0.34917771]])
D = np.array([[0.04498052]])
E = np.array([[ 1.88130305e-01, -2.70291900e-01,  9.12423046e-03,
                -5.78088500e-01,  9.54588221e-03,  5.08576019e-04,
                -1.33890850e+00, -2.02171960e+00, -4.05918956e-01,
                -1.37744223e+00,  1.21206232e-01, -9.26349423e-02,
                -5.38072197e-01,  2.34134460e-03,  4.94334690e-02,
                -1.88329572e-02],
              [-5.35196110e-01, -3.66250013e-01,  2.34622651e-02,
               1.43228677e-01, -1.35959331e-02,  1.32052696e-02,
               7.98717915e-01,  1.35344901e+00, -5.29440815e-02,
               4.88513652e-01,  7.81285093e-01, -3.41019453e-01,
               2.27692972e-01,  7.70150211e-02, -1.25046731e-02,
               -1.62456154e-02]])
F = np.array([[-0.00867042, -0.00636662,  0.00197873, -0.00090865, -0.00088879,
               -0.02759694, -0.01817546, -0.10299409,  0.00648549,  0.08990175,
               0.21129849,  0.00030216,  0.03299013,  0.02058325, -0.09202439,
               -0.0380775]])

true_model = PNLSS(A, B, C, D)
true_model.nlterms('x', [2,3], 'full')
true_model.nlterms('y', [2,3], 'full')
true_model.E = E
true_model.F = F

# excitation signal
RMSu = 0.05  # Root mean square value for the input signal
npp = 1024   # Number of samples
R = 4        # Number of phase realizations (one for validation and one for testing)
P = 3;       # Number of periods
kind = 'Odd' # 'Full','Odd','SpecialOdd', or 'RandomOdd': kind of multisine
f1 = 0       # first excited line
f2 = round(0.9*npp/2) # Last excited line
fs = npp
m = 1        # inputs
p = 1        # outputs

# get predictable random numbers. https://dilbert.com/strip/2001-10-25
np.random.seed(10)
# u:(R,P*npp)
u, t, lines, freq = multisine(f1,f2, fs, npp, P, R, lines=kind, rms=RMSu)
# if multiple input is required, this will copy u m times
# u = np.repeat(u.ravel()[:,None], m, axis=1)  # (R*P*npp,m)

# Transient: Add one period before the start of each realization. To generate
# steady state data.
T1 = np.r_[npp, np.r_[0:(R-1)*P*npp+1:P*npp]]
_, y, _ = true_model.simulate(u.ravel(), T1=T1)

u = u.reshape((R,P,npp)).transpose((2,0,1))  # (npp,R,P)
u = np.repeat(u[:,None],m,axis=1)  # (npp,m,R,P)
y = y.reshape((R,P,npp)).transpose((2,0,1))
y = np.repeat(y[:,None],m,axis=1)

# Add colored noise to the output. randn generate white noise
np.random.seed(10)
noise = 1e-3*np.std(y[:,-1,-1]) * np.random.randn(*y.shape)
# Do some filtering to get colored noise
noise[1:-2] += noise[2:-1]
y += noise

# visualize periodicity
# TODO

# partitioning the data
# test for performance testing and val for model selection
utest = u[:,:,-1,-1]
ytest = y[:,:,-1,-1]
uval = u[:,:,-2,-1]
yval = y[:,:,-2,-1]
# all other realizations are used for estimation
u = u[...,:-2,:]
y = y[...,:-2,:]

# model orders and Subspace dimensioning parameter
nvec = [2,3]
maxr = 5
# store figure handle for later saving the figures
figs = {}

# create signal object
sig = Signal(u,y)
sig.lines(lines)
# average signal over periods. Used for training
um, ym = sig.average()
npp, F = sig.npp, sig.F
R, P = sig.R, sig.P

linmodel = linss()
# estimate bla, total distortion, and noise distortion
linmodel.bla(sig)
# get best model on validation data
models, infodict = linmodel.scan(nvec, maxr)
l_errvec = linmodel.extract_model(yval, uval)
figs['info'] = linmodel.plot_info()
figs['fmodels'] = linmodel.plot_models()

# estimate PNLSS
# transient: Add one period before the start of each realization. Note that
# this for the signal averaged over periods
T1 = np.r_[npp, np.r_[0:(R-1)*npp+1:npp]]

model = PNLSS(linmodel.A, linmodel.B, linmodel.C, linmodel.D)
model.signal = linmodel.signal
model.nlterms('x', [2,3], 'full')
model.nlterms('y', [2,3], 'full')
model.transient(T1)
model.optimize(lamb=100, weight=True, nmax=60)

# compute linear and nonlinear model output on training data
tlin, ylin, xlin = linmodel.simulate(um, T1)
_, ynlin, _ = model.simulate(um)

# get best model on validation data. Change Transient settings, as there is
# only one realization
nl_errvec = model.extract_model(yval, uval, T1=sig.npp)

# compute model output on test data(unseen data)
_, yltest, _ = linmodel.simulate(utest, T1=sig.npp)
_, ynltest, _ = model.simulate(utest, T1=sig.npp)

# linear and nonlinear model error
plt.figure()
plt.plot(ym)
plt.plot(ym-ylin)
plt.plot(ym-ynlin)
plt.xlabel('Time index')
plt.ylabel('Output (errors)')
plt.legend(('output','linear error','PNLSS error'))
plt.title('Estimation results')

# optimization path for PNLSS
plt.figure()
plt.plot(db(nl_errvec))
imin = np.argmin(nl_errvec)
plt.scatter(imin, db(nl_errvec[imin]))
plt.xlabel('Successful iteration number')
plt.ylabel('Validation error [dB]')
plt.title('Selection of the best model on a separate data set')

# result on test data
plt.figure()
#  Normalized frequency vector
freq = np.arange(sig.npp)/sig.npp*sig.fs
plottime = np.hstack((ytest, ytest-yltest, ytest-ynltest))
plotfreq = np.fft.fft(plottime, axis=0)
nfd = plotfreq.shape[0]
plt.plot(freq[:nfd//2], db(plotfreq[:nfd//2]), '.')
plt.xlabel('Frequency (normalized)')
plt.ylabel('Output (errors) (dB)')
plt.legend(('Output','Linear error','PNLSS error'))
plt.title('Test results')


# data = sio.loadmat('data.mat')

# Y_data = data['Y'].transpose((1,2,3,0))
# U_data = data['U'].transpose((1,2,3,0))

# G_data = data['G']
# covGML_data = data['covGML']
# covGn_data = data['covGn']
# covY_data = data['covY'].transpose((2,0,1))

# lines = data['lines'].squeeze() - 1 # 0-based!
# non_exc_even = data['non_exc_even'].squeeze() - 1
# non_exc_odd = data['non_exc_odd'].squeeze() - 1
# A_data = data['A']
# B_data = data['B']
# C_data = data['C']
# D_data = data['D']
# W_data = data['W']

# y = data['y_orig']
# u = data['u_orig']


#def fLMnlssWeighted(u,y,model,MaxCount,W,lambda,LambdaJump):
def bla():
    """
    Optimize PNLSS model using weighted Levenberg-Marquardt algorithm.

        fLMnlssWeighted performs a Levenberg-Marquardt optimization on the
        parameters of a PNLSS model (i.e. the elements of the matrices A, B, C, D,
        E, and F). The difference between the modeled and the measured output is
        minimized in a weighted least squares sense, either in the time or the
        frequency domain. A simple stabilization method can be applied by
        simulating a validation data set during estimation and checking whether or
        not the modeled output stays within prespecified bounds. If not, the
        Levenberg-Marquardt iteration acts as if the cost function increased.

        Parameters
        ----------
            u : N x m input signal
           y : N x p output signal
           model : initial model (see fCreateNLSSmodel). Additionally, two
                   optional fields can be added to the model to perform a
                   simple stabilization method (both fields are needed to
                   perform this method):
                   u_val : m x N_val matrix with N_val samples of the
                           validation input (optional, no default value)
                   max_out : bound on the maximum absolute value of the
                             simulated validation output (optional, no default
                             value)
                   After each Levenberg-Marquardt iteration, the validation
                   data is simulated (without taking into account the
                   transient settings). If the simulated validation output
                   does not respect the max_out bound, then the iteration is
                   considered unsuccessful.
           MaxCount : (maximum) number of iterations (there is not yet an
                      early stopping criterion)
           W : p x p x NFD weighting matrix if frequency-domain weighting
               (e.g. square root of covariance matrix of output noise
               spectrum), where NFD is the number of frequency bins in the
               positive half of the spectrum (of one period and one phase
               realization) of the input (e.g. NFD = floor(Npp/2), where Npp
               is the number of samples in one period and one phase
               realization for a multisine excitation).
               N x p weighting sequences if time-domain weighting.
               [] if no weighting.
               (optional, default is no weighting)
           lambda : initial Levenberg-Marquardt parameter
                    (optional, default = 0, which corresponds to a
                    Gauss-Newton algorithm). After a successful iteration,
                    lambda is halved. After an unsuccessful iteration, lambda
                    is multiplied with a factor sqrt(10), unless lambda was
                    zero, in which case lambda is put equal to the dominant
                    singular value of the Jacobian.
           LambdaJump : each LambdaJump iterations, the Levenberg-Marquardt
                        parameter is made smaller by a factor 10, so that the
                        algorithm leans more towards a Gauss-Newton algorithm,
                        which converges faster than a gradient-descent
                        algorithm (optional, default = 1001)
        Returns
        -------
            model : optimized model (= best on estimation data)
           y_mod : output of the optimized model
           models : collection of models (initial model + model after a
                    successful iteration)
           Cost : collection of the unweighted rms error at each iteration
                  (NaN if iteration was not successful (i.e. when the weighted
                  rms error increased))

        Example:
           % Model input/output data of a Hammerstein system
            N = 2e3; % Number of samples
           u = randn(N,1); % Input signal
           f_NL = @(x) x + 0.2*x.^2 + 0.1*x.^3; % Nonlinear function
           [b,a] = cheby1(2,5,2*0.3); % Filter coefficients
           x = f_NL(u); % Intermediate signal
           y = filter(b,a,x); % Output signal
           scale = u \ x; % Scale factor
           sys = ss(tf(scale*b,a,[])); % Initial linear model = scale factor times underlying dynamics
           nx = [2 3]; % Quadratic and cubic terms in state equation
           ny = [2 3]; % Quadratic and cubic terms in output equation
           T1 = 0; % No periodic signal transient handling
           T2 = 200; % Number of transient samples to discard
           model = fCreateNLSSmodel(sys.a,sys.b,sys.c,sys.d,nx,ny,T1,T2); % Initial linear model
           model.xactive = fSelectActive('inputsonly',2,1,2,nx); % A Hammerstein system only has nonlinear terms in the input
           model.yactive = fSelectActive('inputsonly',2,1,1,nx); % A Hammerstein system only has nonlinear terms in the input
           MaxCount = 50; % Maximum number of iterations
           W = []; % No weighting
           [modelOpt,yOpt] = fLMnlssWeighted(u,y,model,MaxCount,W); % Optimized model and modeled output
           t = 0:N-1;
           figure
               plot(t,y,'b')
               hold on
               plot(t,yOpt,'r')
               xlabel('Time')
               ylabel('Output')
               legend('True','Modeled')

        Reference:
            Paduart, J., Lauwers, L., Swevers, J., Smolders, K., Schoukens, J.,
            and Pintelon, R. (2010). Identification of nonlinear systems using
            Polynomial Nonlinear State Space models. Automatica, 46:647-656.
    """
    pass



# Note that using inv(A) implicitly calls solve and creates an identity
    # matrix. Thus it is faster to allocate In once and then call solve.

# def costfnc(A,B,C,D,G,freq,weight=None):
#     """Compute the cost function as the sum of squares of the weighted error

#     cost = ∑ₖ e[k]ᴴ*σ_G⁻¹*e[k], where the weight is the inverse of the
#     covariance matrix of `G`
#     """

#     # frf of the state space model
#     Gss = fss2frf(A,B,C,D,freq/fs)
#     err = Gss - G
#     # cost = ∑ₖ e[k]ᴴ*σ_G⁻¹*e[k]
#     cost = np.einsum('ki,kij,kj',err.conj().reshape(err.shape[0],-1),
#                      weight,err.reshape(err.shape[0],-1)).real

#     # Normalize with the number of excited frequencies
#     # cost /= F
#     #err_w = np.matmul(weight, err)
#     return cost, err


# # Number of frequency bins where weighting is specified (ie nfd = floor(npp/2),
# # where npp is the number of samples in one period and one phase realization
# # for a multisine excitation
# nfd = weight.shape[0]
# freq_weight = True

# # TODO this only works for T2 scalar.
# # Determine if weighting is in frequency or time domain: only implemented for
# # periodic signals.
# if weight is None:
#     freq_weight = False
#     weight = np.ones((N,p))
# elif nfd > 1:
#     freq_weight = True
#     if system.T2 is not None:
#         R = round((N-T2)/nfd/2)
#         if np.mod(N-T2,R) != 0:
#             raise ValueError('Transient handling and weighting matrix are incompatible. T2 {}'.
#                              format(T2))
# else:
#     # time domain
#     freq_weight = False


class PNLSS2(object):
    """Create polynomial nonlinear state-space model from initial linear state-space model.

    Create a polynomial nonlinear state-space model from a linear
    initialization with state-space matrices A, B, C, and D. The state equation
    is extended with a multivariate polynomial in the states and the inputs.
    The nonlinear degree(s) of this polynomial is/are specified in nx.
    Similarly, the output equation is extended with a multivariate polynomial,
    where the degrees are specified in ny. The transient handling is reflected
    in T1 (for periodic data) and T2 (for aperiodic data). A saturation
    nonlinearity instead of a polynomial one is obsoleted; the optional
    parameter sat should be zero if specified.

        Output parameters:
            model : structure containing the parameters and relevant data of
                    the polynomial nonlinear state-space model. This structure
                    has the following fields:
                   A : n x n state matrix
                   B : n x m input matrix
                   C : p x n output matrix
                   D : p x m feed-through matrix
                   lin.A : n x n state matrix of the linear initialization
                   lin.B : n x m input matrix of the linear initialization
                   lin.C : p x n output matrix of the linear initialization
                   lin.D : p x m feed-through matrix of the linear initialization
                   nx : vector with nonlinear degrees in state update
                   ny : vector with nonlinear degrees in output equation
                   n : number of states
                   m : number of inputs
                   p : number of outputs
                   xpowers : n_nx x (n+m) matrix containing the exponents of
                             each of the n_nx monomials in the state update
                             (see also fCombinations)
                   n_nx : number of monomials in the state update
                   E : n x n_nx matrix with polynomial coefficients in the
                       state update
                   xactive : linear indices of the active elements in the
                             transpose of the E matrix (active elements =
                             elements on which optimization will be done). By
                             default, all elements in the E matrix are set as
                             active. See fSelectActive to change this
                             property.
                   ypowers : n_ny x (n+m) matrix containing the exponents of
                             each of the n_ny monomials in the output equation
                             (see also fCombinations)
                   n_ny : number of monomials in the output equation
                   F : p x n_ny matrix with polynomial coefficients in the
                       output equation
                   yactive : linear indices of the active elements in the
                             transpose of the F matrix (active elements =
                             elements on which optimization will be done). By
                             default, all elements in the F matrix are set as
                             active. See fSelectActive to change this
                             property.
                   T1 : vector that indicates how the transient is handled for
                        periodic signals (see also the Input parameters)
                   T2 : scalar indicating how many samples from the start are
                        removed or vector indicating which samples are removed
                        (see also fComputeIndicesTransientRemovalArb)
                   sat : obsolete, zero flag
                   satCoeff : obsolete, n x 1 vector of ones

    Examples
    --------
       n = 3; % Number of states
       m = 1; % Number of inputs
       p = 1; % Number of outputs
       sys = drss(n,p,m); % Random linear state-space model
       nx = [2 3]; % Quadratic and cubic terms in state equation
       ny = [2 3]; % Quadratic and cubic terms in output equation
       T1 = 0; % No transient handling
       T2 = []; % No transient handling
       sat = 0; % Obsolete parameter sat = 0
       model = fCreateNLSSmodel(sys.a,sys.b,sys.c,sys.d,nx,ny,T1,T2,sat); % Linear state-space model
       N = 1e3; % Number of samples
       u = randn(N,1); % Input signal
       y = fFilterNLSS(model,u); % Modeled output signal
       t = 0:N-1; % Time vector
       y_lsim = lsim(sys,u,t); % Alternative way to calculate output of linear state-space model
       figure
           plot(t,y_lsim,'b')
           hold on
           plot(t,y,'r')
           xlabel('Time')
           ylabel('Output')
           legend('lsim','PNLSS')

    """

# from scipy import signal

# class StateSpace(signal.LinearTimeInvariant):
#     def __new__(cls, *system, **kwargs):
#         """Create new StateSpace object and settle inheritance."""
#         # Handle object conversion if input is an instance of `lti`
#         if len(system) == 1 and isinstance(system[0], LinearTimeInvariant):
#             return system[0].to_ss()

#         # Choose whether to inherit from `lti` or from `dlti`
#         if cls is StateSpace:
#             if kwargs.get('dt') is None:
#                 return StateSpaceContinuous.__new__(StateSpaceContinuous,
#                                                     *system, **kwargs)
#             else:
#                 return StateSpaceDiscrete.__new__(StateSpaceDiscrete,
#                                                   *system, **kwargs)

#         # No special conversion needed
#         return super(StateSpace, cls).__new__(cls)

#     def __init__(self, *system, **kwargs):
#         """Initialize the state space lti/dlti system."""
#         # Conversion of lti instances is handled in __new__
#         if isinstance(system[0], LinearTimeInvariant):
#             return

#         # Remove system arguments, not needed by parents anymore
#         super(StateSpace, self).__init__(**kwargs)

#         self._A = None
#         self._B = None
#         self._C = None
#         self._D = None

#         self.A, self.B, self.C, self.D = abcd_normalize(*system)
