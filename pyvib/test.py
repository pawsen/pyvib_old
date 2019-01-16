#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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


from pyvib.statespace import StateSpace as linss
from pyvib.statespace import Signal
from pyvib.pnlss import PNLSS
from scipy.linalg import norm
import scipy.io as sio
import numpy as np

data = sio.loadmat('data.mat')

Y_data = data['Y'].transpose((1,2,3,0))
U_data = data['U'].transpose((1,2,3,0))

G_data = data['G']
covGML_data = data['covGML']
covGn_data = data['covGn']
covY_data = data['covY'].transpose((2,0,1))

lines = data['lines'].squeeze() - 1 # 0-based!
non_exc_even = data['non_exc_even'].squeeze() - 1
non_exc_odd = data['non_exc_odd'].squeeze() - 1
A_data = data['A']
B_data = data['B']
C_data = data['C']
D_data = data['D']
W_data = data['W']

y = data['y_orig']
u = data['u_orig']

n = 2
r = 3
fs = 1

nvec = [2,3]
maxr = 5


sig = Signal(u,y)
sig.lines(lines)
linmodel = linss()
linmodel.bla(sig)
models, infodict = linmodel.scan(nvec, maxr)

linmodel.plot_info()
linmodel.plot_models()
sig.average()

# model = PNLSS(A_data, B_data, C_data, D_data)
# model.signal = linmodel.signal
# model.nlterms('x', [2,3], 'full')
# model.nlterms('y', [2,3], 'full')

# # samples per period
# npp, F = model.signal.npp, model.signal.F
# R, P = model.signal.R, model.signal.P

# # transient settings
# # Add one period before the start of each realization
# nt = npp
# T1 = np.r_[nt, np.r_[0:(R-1)*npp+1:npp]]
# T2 = 0
# model.transient(T1,T2)
# model.optimize(lamb=100, weight=W_data.T)



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
