#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .common import  mmul_weight
import numpy as np
from numpy.fft import fft
from scipy.special import comb
from scipy.interpolate import interp1d

#only for class PNLSS
from .common import (matrix_square_inv, lm)
from copy import deepcopy

"""
PNLSS -- a collection of classes and functions for modeling nonlinear
linear state space systems.
"""
class PNLSS(object):
    def __init__(self, A,B,C,D, **kwargs):
        """Initialize the state space lti/dlti system."""

        self.A, self.B, self.C, self.D = A, B, C, D

        self.n = self.A.shape[0]
        self.m = self.B.shape[1]
        self.p = self.C.shape[0]

        self.T1 = None
        self.T2 = None

        self.dt = 0.1

    def setss(self, *system):
        if len(system) == 6:
            self.A, self.B, self.C, self.D, self.E, self.F = system

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
            # Compute the derivatives of the polynomials zeta and e
            self.xd_powers, self.xd_coeff = poly_deriv(self.xpowers)
        elif eq in ('output', 'y'):
            self.ydegree = np.asarray(degree)
            self.ystructure = structure
            self.ypowers = combinations(self.n+self.m, degree)
            self.n_ny = self.ypowers.shape[0]
            self.yactive = \
                select_active(self.ystructure,self.n,self.m,self.p,self.ydegree)
            self.F = np.zeros((self.p, self.n_ny))
            self.yd_powers, self.yd_coeff = poly_deriv(self.ypowers)

    def transient(self, T1=None, T2=None):
        """Transient handling. t1: periodic, t2: aperiodic
        Get transient index. Only needed to run once
        """

        self.T1 = T1
        self.T2 = T2
        sig = self.signal
        ns = sig.R * sig.npp
        # Extract the transient part of the input
        self.idx_trans = transient_indices_periodic(T1, ns)
        self.idx_remtrans = remove_transient_indices_periodic(T1, ns, sig.p)
        self.umt = sig.um[self.idx_trans]
        self.n_trans = self.umt.shape[0]

        # without_T2 = remove_transient_indices_nonperiodic(system.T2,N,system.p)
        self.without_T2 = np.s_[:ns]

    def simulate(self, u, t=None, x0=None, T1=None, T2=None):
        """Calculate the output and the states of a nonlinear state-space model with
        transient handling.

        """

        if T1 is None:
            T1 = self.T1
            T2 = self.T2

        # Number of samples
        ns = u.shape[0]
        if T1 is not None:
            # Prepend transient samples to the input
            idx = self.idx_trans
            u = u[idx]

        t, y, x = dnlsim(self, u, t, x0)

        if T1 is not None:
            # remove transient samples. p=1 is correct. TODO why?
            idx = remove_transient_indices_periodic(self.T1, ns, p=1)
            x = x[idx]
            y = y[idx]

        self.x_mod = x
        self.y_mod = y
        return t, y, x

    def optimize(self, method=None, weight=True, info=True, copy=False, lamb=None):
        """Optimize the estimated the nonlinear state space matrices"""

        self.freq_weight = True
        if weight is True:
            covYinvsq = np.empty_like(self.covY)
            for f in range(self.signal.F):
                covYinvsq[f] = matrix_square_inv(self.covG[f])
            self.weight = covYinvsq
        else:
            self.weight = weight

        x0 = self.flatten_ss()
        if method is None:
            res = lm(costfnc, x0, jacobian, system=self, weight=self.weight,
                     info=info, lamb=lamb)
        else:
            res = least_squares(costfnc,x0,jacobian, method='lm',
                                x_scale='jac',
                                kwargs={'system':self,'weight':self.weight})

        if copy:
            # restore state space matrices as they are
            self.A, self.B, self.C, self.D, self.E, self.F = extract_ss(x0, self)

            nmodel = deepcopy(self)
            nmodel.A, nmodel.B, nmodel.C, nmodel.D, model.E, model.F = \
                extract_ss(res['x'], nmodel)
            nmodel.res = res
            return nmodel

        self.A, self.B, self.C, self.D, self.E, self.F = extract_ss(res['x'], self)
        self.res = res

    def cost(self, weight=None):

        if weight is True:
            weight = self.weight

        x0 = self.flatten_ss()
        err = costfnc(x0, self, weight=weight)
        # TODO maybe divide by 2 to match scipy's implementation of minpack
        self.cost = np.dot(err, err)
        return self.cost

    def flatten_ss(self):
        """Returns the state space as flattened array"""

        # samples per period
        n, m, p = self.n, self.m, self.p
        n_nx, n_ny = self.n_nx, self.n_ny
        self.npar = n**2 + n*m + p*n + p*m + n*n_nx + p*n_ny

        # initial guess
        x0 = np.empty(self.npar)
        x0[:n**2] = self.A.ravel()
        x0[n**2 + np.r_[:n*m]] = self.B.ravel()
        x0[n**2 + n*m + np.r_[:n*p]] = self.C.ravel()
        x0[n*(p+m+n) + np.r_[:p*m]] = self.D.ravel()
        x0[n*(p+m+n)+p*m + np.r_[:n*n_nx]] = self.E.ravel()
        x0[n*(p+m+n+n_nx)+p*m + np.r_[:p*n_ny]] = self.F.ravel()
        return x0


def combinations(n, degrees):
    """Lists all nonlinear terms in a multivariate polynomial.

    Lists the exponents of all possible monomials in a multivariate polynomial
    with n inputs. Only the nonlinear degrees in ``degrees`` are considered.

    Parameters
    ----------
    n: int
        number of inputs
    degrees: ndarray
        array with the degrees of nonlinearity

    Returns
    -------
    monomials : ndarray(ncomb,n)
        matrix of exponents

    Examples
    --------
    A polynomial with all possible quadratic and cubic terms in the variables x
    and y contains the monomials x*x, x*y, y*y, x*x*x, x*x*y, x*y*y, and y*y*y.
    >>> out = combinations(2,[2, 3])
    array([[2, 0],  #  -> x^2 * y^0 = x*x
           [1, 1],  #  -> x^1 * y^1 = x*y
           [0, 2],  #  -> x^0 * y^2 = y*y
           [3, 0],  #  -> x^3 * y^0 = x*x*x
           [2, 1],  #  -> x^2 * y^1 = x*x*y
           [1, 2],  #  -> x^1 * y^2 = x*y*y
           [0, 3]]) #  -> x^0 * y^3 = y*y*y

    Element (i,j) of ``out`` indicates the power to which variable j is raised
    in monomial i. For example, out[4] = [2,1], which means that the fifth
    monomial is equal to x^2*y^1 = x*x*y.

    """

    degrees = np.asarray(degrees)
    if not np.issubdtype(degrees.dtype, np.integer):
        raise ValueError('wrong type in degrees. Should only be int. Is {}'.
                         format(degrees.dtype))
    # Consider only nonlinear degrees
    degrees = degrees[degrees > 1]

    # Determine total number of combinations/monomials
    ncomb = 0
    for degree in degrees:
	    ncomb += comb(n+degree-1, degree, exact=True)

    # List the exponents of each input in all monomials
    monomials = np.zeros((ncomb,n),dtype=int)
    idx = 0  # Running index indicating the last used row in out
    for degree in degrees:
        # All combinations in homogeneous polynomial of degree
        comb_i = hom_combinations(n,degree)
        ncomb_i = comb_i.shape[0]
        monomials[idx: idx+ncomb_i] = comb_i
        idx += ncomb_i

    return monomials


def hom_combinations(n,degree):
    """Lists the exponents of all possible terms in a homogeneous polynomial
    monomial representation, e.g. [1 2] represents x1*x2**2

    Examples
    --------
    >>> hom_combinations(2,2)
    array([[2, 0],
           [1, 1],
           [0, 2]])
    """

    # Number of combinations in homogeneous polynomial
    ncomb = comb(n+degree-1,degree, exact=True)
    # Preallocating and start from all ones => x1*x1*x1
    monomials = np.ones((ncomb,degree), dtype=int)

    for i in range(1,ncomb):
        monomials[i] = monomials[i-1].copy()
        j = degree-1  # Index indicating which factor to change
        while monomials[i,j] == n:
            # Try to increase the last factor, but if this is not possible,
            # look the previous one that can be increased
            j -= 1
         # Increase factor j wrt previous monomial, e.g. x1*x1*x1 -> x1*x1*x2
        monomials[i,j] += 1
        # Monomial after x1*x1*xmax is x1*x2*x2, and not x1*x2*xmax
        monomials[i,j+1:degree] = monomials[i,j]

    # Exponents representation, e.g. [2, 1] represents x1^2*x2 = x1*x1*x2
    combinations = np.zeros((ncomb,n), dtype=int)
    for i in range(ncomb):  # # Loop over all terms
        for j in range(n):  # # Loop over all inputs
            # Count the number of appearances of input j in monomial i. +1 for
            # zero index
            combinations[i,j] = np.sum(monomials[i] == j+1)

    return combinations

def select_active(structure,n,m,q,nx):
    """Select active elements in E or F matrix.

    Select the active elements (i.e. those on which optimization will be done)
    in the E or F matrix. In particular, the linear indices (see also sub2ind
    and ind2sub) of the active elements in the transpose of the E or F matrix
    are calculated.

    Parameters:
    -----------
    structure: str
        string indicating which elements in the E or F matrix are active. The
        possibilities are 'diagonal', 'inputsonly', 'statesonly',
        'nocrossprod', 'affine', 'affinefull', 'full', 'empty', 'nolastinput',
        or num2str(row_E).
        'diagonal': active elements in row j of the E matrix are those
                    corresponding to pure nonlinear terms in state j (only for
                    state equation)
        'inputsonly' : only terms in inputs
        'statesonly' : only terms in states
        'nocrossprod' : no cross-terms
        'affine' : only terms that are linear in one state
        'affinefull' : only terms that are linear in one state or constant in
                     the states
        'full' : all terms
        'empty' : no terms
        'nolastinput' : no terms in last input
        num2str(row_E) : only row row_E in E matrix is active (only for state equation)
    n : int
        number of states
    m : int
        number of inputs
    q : int
        number of rows in corresponding E/F matrix
           q = n if E matrix is considered,
           q = p if F matrix is considered
    nx : int | list
        degrees of nonlinearity in E/F matrix


    Returns
    -------
    active: linear indices of the active elements in the transpose of the E or F matrix

    Examples
    --------

    		n = 2; % Number of states
           m = 1; % Number of inputs
           p = 1; % Number of outputs
           nx = 2; % Degree(s) of nonlinearity
           terms = combinations(n+m,nx)  # Powers of all possible terms in n+m inputs of degree(s) nx
           % => terms = [2 0 0;
           %             1 1 0;
           %             1 0 1;
           %             0 2 0;
           %             0 1 1;
           %             0 0 2];
           % There are six quadratic terms in the two states x1 and x2, and
           % the input u, namely x1^2, x1*x2, x1*u, x2^2, x2*u, and u^2.
           % The matrix E is a 2 x 6 matrix that contains the polynomial
           % coefficients in each of these 6 terms for both state updates.
           % The active elements will be calculated as linear indices in the
           % transpose of E, hence E can be represented as
           % E = [e1 e2 e3 e4  e5  e6;
           %      e7 e8 e9 e10 e11 e12];
           % The matrix F is a 1 x 6 matrix that contains the polynomial
           % coefficients in each of the 6 terms for the output equation.
           % The matrix F can be represented as
           % F = [f1 f2 f3 f4 f5 f6];

           % Diagonal structure
           activeE = fSelectActive('diagonal',n,m,n,nx);
           % => activeE = [1 10].';
           % Only e1 and e10 are active. This corresponds to a term x1^2 in
           % the first state equation and a term x2^2 in the second state
           % equation.

           % Inputs only structure
           activeE = fSelectActive('inputsonly',n,m,n,nx);
           % => activeE = [6 12].';
           % Only e6 and e12 are active. This corresponds to a term u^2 in
           % both state equations. In all other terms, at least one of the
           % states (possibly raised to a certain power) is a factor.
           activeF = fSelectActive('inputsonly',n,m,p,nx);
           % => activeF = 6;
           % Only f6 is active. This corresponds to a term u^2 in the output
           % equation.

           % States only structure
           activeE = fSelectActive('statesonly',n,m,n,nx);
           % => activeE = [1 2 4 7 8 10].';
           % Only e1, e2, e4, e7, e8, and e10 are active. This corresponds to
           % terms x1^2, x1*x2, and x2^2 in both state equations. In all other
           % terms, the input (possibly raised to a certain power) is a
           % factor.

           % No cross products structure
           activeE = fSelectActive('nocrossprod',n,m,n,nx);
           % => activeE = [1 4 6 7 10 12].';
           % Only e1, e4, e6, e7, e10, and e12 are active. This corresponds to
           % terms x1^2, x2^2, and u^2 in both state equations. All other
           % terms are crossterms where more than one variable is present as a
           % factor.

           % State affine structure
           activeE = fSelectActive('affine',n,m,n,nx);
           % => activeE = [3 5 9 11].';
           % Only e3, e5, e9, and e11 are active. This corresponds to
           % terms x1*u and x2*u in both state equations, since in these terms
           % only one state appears, and it appears linearly.

           % Full state affine structure
           activeE = fSelectActive('affinefull',n,m,n,nx);
           % => activeE = [3 5 6 9 11 12].';
           % Only e3, e5, e6, e9, e11, and e12 are active. This corresponds to
           % terms x1*u, x2*u and u^2 in both state equations, since in these
           % terms at most one state appears, and if it appears, it appears
           % linearly.

           % Full structure
           activeE = fSelectActive('full',n,m,n,nx);
           % => activeE = (1:12).';
           % All elements in the E matrix are active.

           % Empty structure
           activeE = fSelectActive('empty',n,m,n,nx);
           % => activeE = [];
           % None of the elements in the E matrix are active.

           % One row in E matrix structure
           row_E = 2; % Select which row in E is active
           activeE = fSelectActive('row_E',n,m,n,nx);
           % => activeE = [7 8 9 10 11 12].';
           % Only the elements in the second row of E are active

           % No terms in last input structure
           % This is useful in a polynomial nonlinear state-space (PNLSS)
           % model when considering the initial state as a parameter. The
           % state at time one can be estimated by adding an extra input
           % u_art(t) that is equal to one at time zero and zero elsewhere.
           % Like this, an extended PNLSS model is estimated, where the last
           % column in its B matrix corresponds to the state at time one in
           % the original PNLSS model. To ensure that the optimization is only
           % carried out on the parameters of the original PNLSS model, only
           % the corresponding coefficients in the E/F matrix should be
           % selected as active.
           terms_extended = fCombinations(n+m+1,nx); % Powers of all possible terms with one extra input
           % => terms_extended = [2 0 0 0;
           %                      1 1 0 0;
           %                      1 0 1 0;
           %                      1 0 0 1;
           %                      0 2 0 0;
           %                      0 1 1 0;
           %                      0 1 0 1;
           %                      0 0 2 0;
           %                      0 0 1 1;
           %                      0 0 0 2];
           % The nonlinear terms in the extra input should not be considered
           % for optimization.
           activeE_extended = fSelectActive('nolastinput',n,m+1,n,nx);
           % => activeE_extended = [1 2 3 5 6 8 11 12 13 15 16 18].';
           % Only the terms where the last input is raised to a power zero are
           % active. This corresponds to the elif structure == where all terms in the
           % original PNLSS model are active.
           % The example below illustrates how to combine a certain structure
           % in the original model (e.g. 'nocrossprod') with the estimation of
           % the initial state.
           activeE_extended = fSelectActive('nolastinput',n,m+1,n,nx);
           activeE_extended = activeE_extended(fSelectActive('nocrossprod',n,m,n,nx));
           % => activeE_extended = [1 5 8 11 15 18].';
           % This corresponds to the terms x1^2, x2^2, and u1^2 in both rows
           % of the E_extended matrix, and thus to all terms in the original
           % model, except for the crossterms.
           % Note that an alternative approach is to include the initial state
           % in the parameter vector (see also fLMnlssWeighted_x0u0).

    """
    # All possible nonlinear terms of degrees nx in n+m inputs
    combis = combinations(n+m,nx)
    n_nl = combis.shape[0]  # Number of terms

    stype = {'diagonal', 'inputsonly', 'statesonly', 'nocrossprod', 'affine',
             'affinefull', 'full', 'empty', 'nolastinput'}
    if structure == 'diagonal':
        # Diagonal structure requires as many rows in E (or F) matrix as the number of states
        if n != q:
            raise ValueError('Diagonal structure can only be used in state equation, not in output equation')
        # Find terms that consist of one state, say x_j, raised to a nonzero power
        active = np.where((np.sum(combis[:,:n] != 0,1) == 1) &
                          (np.sum(combis[:,n:] != 0,1) == 0))[0]
        # Select these terms only for row j in the E matrix
        for i, item in enumerate(active):
            # Which state variable is raised to a nonzero power
            tmp = np.where(combis[item] != 0)[0]
            # Linear index of active term in transpose of E
            active[i] += (tmp)*n_nl
    elif structure == 'inputsonly':
        # Find terms where all states are raised to a zero power
        active = np.where(np.sum(combis[:,:n] != 0,1) == 0)[0]
    elif structure == 'statesonly':
        # Find terms where all inputs are raised to a zero power
        active = np.where(np.sum(combis[:,n:] != 0,1) == 0)[0]
    elif structure == 'nocrossprod':
        # Find terms where only one variable (state or input) is raised to a
        # nonzero power
        active = np.where(np.sum(combis != 0,1) == 1)[0]
    elif structure == 'affine':
        # Find terms where only one state is raised to power one, and all
        # others to power zero. There are no conditions on the powers in the
        # input variables
        active = np.where((np.sum(combis[:,:n] != 0,1) == 1) &
                          (np.sum(combis[:,:n], 1) == 1))[0]

    elif structure == 'affinefull':
        # Find terms where at most one state is raised to power one, and
        # all others to power zero. There are no conditions on the powers
        # in the input variables
        active = np.where((np.sum(combis[:,:n] != 0,1) <= 1) &
                          (np.sum(combis[:,:n], 1) <= 1))[0]
    elif structure == 'full':
        # Select all terms in E/F matrix
        active = np.arange(q*n_nl)
    elif structure == 'empty':
        # Select no terms
        active = []
    elif structure == 'nolastinput':
        if m > 0:
            # Find terms where last input is raised to power zero
            active = np.where(combis[:,-1] == 0)[0]
        else:
            raise ValueError('There is no input for {}'.format(structure))
    else:
        # Check if one row in E is selected. Remember we use 0-based rows
        if (isinstance(structure, (int, np.integer)) and
            structure in np.arange(n)):
            row_E = int(structure)
            active = row_E*n_nl + np.arange(n_nl)
        else:
           raise ValueError('Wrong structure {}. Should be: {}'.
                            format(structure, stype))

    if structure in \
       ('inputsonly','statesonly','nocrossprod','affine','affinefull'):
        # Select terms for all rows in E/F matrix
        active = (np.tile(active[:,None], q) +
                  np.tile(np.linspace(0,n_nl,q, dtype=int) ,(len(active),1))).ravel()

    # Sort the active elements
    return np.sort(active)

def remove_transient_indices_nonperiodic(T2,N,p):
    """Remove transients from arbitrary data.

    Computes the indices to be used with a (N,p) matrix containing p output
    signals of length N, such that y[indices] contains the transient-free
    output(s) of length NT stacked on top of each other (if more than one
    output). The transient samples to be removed are specified in T2 (T2 =
    np.arange(T2) if T2 is scalar).

    Parameters
    ----------
    T2 : int
        scalar indicating how many samples from the start are removed or vector
        indicating which samples are removed
    N : int
        length of the total signal
    p : int
        number of outputs

    Returns
    -------
    indices : ndarray(int)
        vector of indices, such that y(indices) contains the output(s) without
        transients. If more than one output (p > 1), then y(indices) stacks the
        transient-free outputs on top of each other.
    nt : int
        length of the signal without transients


    Examples
    --------
    One output, T2 scalar
    N = 1000 # Total number of samples
    T2 = 200; % First 200 samples should be removed after filtering
    p = 1; % One output
    [indices, NT] = fComputeIndicesTransientRemovalArb(T2,N,p);
    % => indices = (201:1000).'; % Indices of the transient-free output (in uint32 format in version 1.0)
    % => NT = 800; % Number of samples in the transient-free output

    Two outputs, T2 scalar
    N = 1000; % Total number of samples
    T2 = 200; % First 200 samples should be removed after filtering
    p = 2; % Two outputs
    [indices, NT] = fComputeIndicesTransientRemovalArb(T2,N,p);
    % => indices = ([201:1000 1201:2000]).'; % Indices of the transient-free outputs (in uint32 format in version 1.0)
    % => NT = 800; % Number of samples in each transient-free output
    % If y = [y1 y2] is a 1000 x 2 matrix with the two outputs y1 and y2,
    % then y(indices) = [y1(201:1000);
    %                    y2(201:1000)]
    % is a vector with the transient-free outputs stacked on top of
    % each other

   % One output, T2 is a vector
    N1 = 1000; % Number of samples in a first data set
    N2 = 500; % Number of samples in a second data set
    N = N1 + N2; % Total number of samples
    T2_1 = 1:200; % Transient samples in first data set
    T2_2 = 1:100; % Transient samples in second data set
    T2 = [T2_1 (N1+T2_2)]; % Transient samples
    p = 1; % One output
    [indices, NT] = fComputeIndicesTransientRemovalArb(T2,N,p);
    % => indices = ([201:1000 1101:1500])
    % => NT = 1200;
    """

    if T2 is None:
        T2 = [0]

    # TODO make it possible to give T2 as list, T2 = [200]
    if isinstance(T2, (int, np.integer)):  #np.isscalar(T2):
        # Remove all samples up to T2
        T2 = np.arange(T2)

    T2 = np.atleast_1d(np.asarray(T2, dtype=int))
    # Remove transient samples from the total
    # TODO wrong: now we remove idx 0, which is wrong
    without_T2 = np.delete(np.arange(N), T2)

    # Length of the transient-free signal(s)
    nt = len(without_T2)
    if p > 1:  # for multiple outputs
        indices = np.zeros(p*NT, dtype=int)
        for i in range(p):
            # Stack indices for each output on top of each other
            indices[i*NT:(i+1)*NT] = without_T2 + i*N
    else:
        indices = without_T2

    return indices, nt

def transient_indices_periodic(T1,N):
    """Computes indices for transient handling of periodic signals.

	Computes the indices to be used with a vector u of length N that contains
	(several realizations of) a periodic signal, such that u[indices] has T1[0]
	transient samples prepended to each realization. The starting samples of
	each realization can be specified in T1[1:]. Like this, steady-state
	data can be obtained from a PNLSS model by using u[indices] as an input
	signal to a PNLSS model (see fFilterNLSS) and removing the transient
	samples afterwards (see fComputeIndicesTransientRemoval).

	Parameters
    ----------
	T1 : int | ndarray(int)
        array that indicates how the transient is handled. The first element
        T1[0] is the number of transient samples that should be prepended to
        each realization. The other elements T1[1:] indicate the starting
        sample of each realization in the signal. If T1 has only one element,
        T1[1] is put to zero, ie. first element.
    N : int
        length of the signal containing all realizations

    Returns
    -------
	indices : ndarray(int)
        indices of a vector u that contains (several realizations of) a
        periodic signal, such that u[indices] has a number of transient samples
        added before each realization

	Examples
    --------
    Npp = 1000; % Number of points per period
    R = 2; % Number of phase realizations
    T = 100; % Number of transient samples
    T1 = [T 1:Npp:(R-1)*Npp+1]; % Transient handling vector
    N = R*Npp; % Total number of samples
    indices = fComputeIndicesTransient(T1,N);
    % => indices = [901:1000 1:1000 1901:2000 1001:2000]
    %            = [transient samples realization 1, ...
    %               realization 1, ...
    %               transient samples realization 2, ...
    %               realization 2]

    # np.r_[T, np.r_[0:(R-1)*npp+1:npp]]
    """

    T1 = np.atleast_1d(np.asarray(T1, dtype=int))
    ntrans = T1[0]

    if ntrans != 0:

        if len(T1) == 1:
            # If starting samples of realizations not specified, then we assume
            # the realization start at the first sample
            T1 = np.append(T1, 0)
        # starting index of each realization and length of signal
        T1 = np.append(T1[1:], N)

        indices = np.array([], dtype=int)
        for i in range(len(T1)-1):
            trans = T1[i+1] -1 - np.mod(np.arange(ntrans)[::-1], T1[i+1]-T1[i])
            normal = np.arange(T1[i],T1[i+1])
            indices = np.hstack((indices, trans, normal))
    else:
        # No transient points => output = all indices of the signal
        indices = np.arange(N)

    return indices

def remove_transient_indices_periodic(T1,N,p):
    """Computes indices for transient handling for periodic signals after
    filtering

    Let u be a vector of length N containing (several realizations of) a
    periodic signal. Let uTot be a vector containing the signal(s) in u with
    T1[0] transient points prepended to each realization (see
    fComputeIndicesTransient). The starting samples of each realization can be
    specified in T1[1:]. Let yTot be a vector/matrix containing the p outputs
    of a PNLSS model after applying the input uTot. Then
    fComputeIndicesTransientRemoval computes the indices to be used with the
    vectorized form of yTot such that the transient samples are removed from
    yTot, i.e. y = yTot[indices] contains the steady-state output(s) stacked on
    top of each other.

    Parameters
    ----------
	T1 : ndarray(int)
        vector that indicates how the transient is handled. The first element
        T1[0] is the number of transient samples that were prepended to each
        realization. The other elements T1[1:] indicate the starting sample
        of each realization in the input signal. If T1 has only one element,
        T1[1] is put to zero.
    N : int
        length of the input signal containing all realizations
    p : int
        number of outputs

    Returns
    -------
	indices : ndarray(int)
        If uTot is a vector containing (several realizations of) a periodic
        signal to which T1[0] transient points were added before each
        realization, and if yTot is the corresponding output vector (or matrix
        if more than one output), then indices is such that the transient
        points are removed from y = yTot(indices). If p > 1, then indices is a
        column vector and y = yTot(indices) is a column vector with the steady
        state outputs stacked on top of each other.

	Examples
    --------
    Npp = 1000; % Number of points per period
    R = 2; % Number of phase realizations
    T = 100; % Number of transient samples
    T1 = [T 1:Npp:(R-1)*Npp+1]; % Transient handling vector
    N = R*Npp; % Total number of samples
    indices_tot = fComputeIndicesTransient(T1,N);
    % => indices_tot = [901:1000 1:1000 1901:2000 1001:2000]
    %                = [transient samples realization 1, ...
    %                   realization 1, ...
    %                   transient samples realization 2, ...
    %                   realization 2]
    p = 1; % One output
    indices_removal = fComputeIndicesTransientRemoval(T1,N,p);
    % => indices_removal = [101:1100 1201:2200].'
    % => indices_tot(indices_removal) = 1:2000
    %                                 = [realization 1, realization 2]
    p = 2; % More than one output
    indices_removal = fComputeIndicesTransientRemoval(T1,N,p);
    % => indices_removal = [101:1100 1201:2200 2301:3300 3401:4400].'
    % Let u be a vector containing [input realization 1;
    %                               input realization 2],
    % then uTot = u(indices_tot) is a vector containing
    %             [transient samples realization 1;
    %              input realization 1;
    %              transient samples realization 2;
    %              input realization 2]
    % Let y1 be a vector containing the first output and y2 be a vector
    % containing the second output when applying uTot as an input to a
    % PNLSS model, and let yTot = [y1 y2] be a 2200 x 2 matrix with y1
    % and y2 in its first and second column, respectively.
    % Note that y1 = yTot(1:2200).' and y2 = yTot(2201:4400).' (see
    % also ind2sub and sub2ind)
    % Then yTot(indices_removal) = [y1(101:1100);
    %                               y1(1201:2200);
    %                               y2(101:1100);
    %                               y2(1201:2200)]
    %                            = [output 1 corresponding to input realization 1;
    %                               output 1 corresponding to input realization 2;
    %                               output 2 corresponding to input realization 1;
    %                               output 2 corresponding to input realization 2]
    """

    T1 = np.atleast_1d(np.asarray(T1, dtype=int))
    ntrans = T1[0]

    if ntrans == 0:
        return np.arange(N)


    if len(T1) == 1:
        # If starting samples of realizations not specified, then we assume
        # the realization start at the first sample
        T1 = np.append(T1, 0)

    # starting index of each realization and length of signal
    T1 = np.append(T1[1:], N)

    indices = np.array([], dtype=int)
    for i in range(len(T1)-1):
        # Concatenate indices without transient samples
        indices = np.hstack((indices,
                             np.r_[T1[i]:T1[i+1]] + (i+1)*ntrans))

    # TODO This is not correct for p>1. We still store y.shape -> (N,p)
    if p > 1:
        # Total number of samples per output = number of samples without + with
        # transients
        nt = N + ntrans*(len(T1)-1)

        tmp = np.empty(p*N, dtype=int)
        for i in range(p):
            #Stack indices without transient samples on top of each other
            tmp[i*N:(i+1)*N] = indices + i*nt
        indices = tmp

    return indices

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

    out = np.zeros((p,N,nactive))
    for k, activ in enumerate(active):
        # Which column in A, B, or E matrix
        j = np.mod(activ, npar)
        # Which row in A, B, or E matrix
        i = (activ-j)//npar
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

def jacobian(x0, system, weight=None):

    """Compute the Jacobians in a steady state nonlinear state-space model

    Jacobians of a nonlinear state-space model

        x(t+1) = A x(t) + B u(t) + E zeta(x(t),u(t))
        y(t)   = C x(t) + D u(t) + F eta(x(t),u(t))

    i.e. the partial derivatives of the modeled output w.r.t. the active
    elements in the A, B, E, F, D, and C matrices

    x0 : ndarray
        flattened array of state space matrices

    """

    n, m, p = system.n, system.m, system.p
    R, p, npp = system.signal.R, system.signal.p, system.signal.npp
    nfd = npp//2
    n_trans = system.n_trans
    without_T2 = system.without_T2

    A, B, C, D, E, F = extract_ss(x0, system)

    # Collect states and outputs with prepended transient sample
    x_trans = system.x_mod[system.idx_trans]
    u_trans = system.umt
    contrib = np.hstack((x_trans, u_trans)).T

    # E∂ₓζ + A(n,n,NT)
    A_EdwxIdx = multEdwdx(contrib,system.xd_powers,np.squeeze(system.xd_coeff),
                          E,n) + A[...,None]
    zeta = nl_terms(contrib, system.xpowers).T  # (NT,n_nx)

    # F∂ₓη  (p,n,NT)
    FdwyIdx = multEdwdx(contrib,system.yd_powers,np.squeeze(system.yd_coeff),
                        F,n)
    eta = nl_terms(contrib, system.ypowers).T  # (NT,n_ny)

    # calculate jacobians wrt state space matrices
    JC = np.kron(np.eye(p), system.x_mod)  # (p*N,p*n)
    JD = np.kron(np.eye(p), system.signal.um)  # (p*N, p*m)
    JF = np.kron(np.eye(p), eta)  # Jacobian wrt all elements in F
    JF = JF[:,system.yactive]  # all active elements in F. (p*NT,nactiveF)
    JF = JF[system.idx_remtrans]  # (p*N,nactiveF)


    # Add C to F∂ₓη for all samples at once
    FdwyIdx += system.C[...,None]
    # calculate Jacobian by filtering an alternative state-space model
    JA = element_jacobian(x_trans, A_EdwxIdx, system.C, FdwyIdx,
                          np.arange(n**2))
    JA = JA.transpose((1,0,2)).reshape((p*n_trans, n**2))
    JA = JA[system.idx_remtrans]  # (p*N,n**2)

    JB = element_jacobian(u_trans, A_EdwxIdx, system.C, FdwyIdx,
                          np.arange(n*m))
    JB = JB.transpose((1,0,2)).reshape((p*n_trans, n*m))
    JB = JB[system.idx_remtrans]  # (p*N,n*m)

    JE = element_jacobian(zeta, A_EdwxIdx, system.C, FdwyIdx, system.xactive)
    JE = JE.transpose((1,0,2)).reshape((p*n_trans, len(system.xactive)))
    JE = JE[system.idx_remtrans]  # (p*N,nactiveE)

    jac = np.hstack((JA, JB, JC, JD, JE, JF))[without_T2]
    npar = jac.shape[1]

    # add frequency weighting
    # (p*ns, npar) -> (Npp,R,p,npar) -> (Npp,p,R,npar) -> (Npp,p,R*npar)
    jac = jac.reshape((npp,R,p,npar),
                      order='F').swapaxes(1,2).reshape((-1,p,R*npar),
                                                       order='F')
    # select only the positive half of the spectrum
    jac = fft(jac, axis=0)[:nfd]
    jac = mmul_weight(jac, weight)
    # (nfd,p,R*npar) -> (nfd,p,R,npar) -> (nfd,R,p,npar) -> (nfd*R*p,npar)
    jac = jac.reshape((-1,p,R,npar),
                      order='F').swapaxes(1,2).reshape((-1,npar), order='F')

    J = np.empty((2*nfd*R*p,npar))
    J[:nfd*R*p] = jac.real
    J[nfd*R*p:] = jac.imag

    return J


def extract_ss(x0, system):

    n, m, p = system.n, system.m, system.p
    n_nx, n_ny = system.n_nx, system.n_ny
    #ne = system.xactive # n*n_nx ?
    A = x0.flat[:n**2].reshape((n,n))
    B = x0.flat[n**2 + np.r_[:n*m]].reshape((n,m))
    C = x0.flat[n**2+n*m + np.r_[:p*n]].reshape((p,n))
    D = x0.flat[n*(p+m+n) + np.r_[:p*m]].reshape((p,m))
    E = x0.flat[n*(p+m+n)+p*m + np.r_[:n*n_nx]].reshape((n,n_nx))
    F = x0.flat[n*(p+m+n+n_nx)+p*m + np.r_[:p*n_ny]].reshape((p,n_ny))

    return A, B, C, D, E, F

def costfnc(x0, system, weight=None):
    # TODO fix transient
    T2 = system.T2
    R, p, npp = system.signal.R, system.signal.p, system.signal.npp
    nfd = npp//2
    without_T2 = system.without_T2

    # update the state space matrices from x0
    # TODO find a way to avoid explicitly updating the state space model.
    # It is not the expected behavior that calculating the cost should change
    # the model! Right now it is done because simulating is using the systems
    # ss matrices

    #A, B, C, D, E, F = extract_ss(x0, system)
    system.setss(*extract_ss(x0, system))
    # Compute the (transient-free) modeled output and the corresponding states
    t_mod, y_mod, x_mod = system.simulate(system.signal.um)

    # Compute the (weighted) error signal without transient
    err = system.y_mod[without_T2] - system.signal.ym[without_T2]
    if weight is not None and system.freq_weight:
        err = err.reshape((npp,R,p),order='F').swapaxes(1,2)
        # Select only the positive half of the spectrum
        err = fft(err, axis=0)[:nfd]
        err = mmul_weight(err, weight)
        #cost = np.vdot(err, err).real
        err = err.swapaxes(1,2).ravel(order='F')
        err_w = np.hstack((err.real.squeeze(), err.imag.squeeze()))
    elif weight is not None:
        err_w = err * weight[without_T2]
        #cost = np.dot(err,err)
    else:
        # no weighting
        return err

    return err_w
