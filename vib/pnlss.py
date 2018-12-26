import numpy as np
from scipy.special import comb
def combinations(n, degree):
    """Lists all nonlinear terms in a multivariate polynomial.
    #
    #	Usage:
    #		out = fCombinations(n,degrees)
    #
    #	Description:
    #		Lists the exponents of all possible monomials in a multivariate
    #		polynomial with n inputs. Only the nonlinear degrees in degrees are
    #		considered.
    #
    #	Output parameters:
    #		out : ncomb x n matrix of exponents
    #
    #	Input parameters:
    #		n : number of inputs
    #       degrees : vector with the degrees of nonlinearity
    #
    #	Example:
    #		% A polynomial with all possible quadratic and cubic terms in the
    #		% variables x and y contains the monomials x*x, x*y, y*y, x*x*x,
    #		% x*x*y, x*y*y, and y*y*y.
    #       out = fCombinations(2,[2 3])
    #       % => out = [2 0;        -> x^2 * y^0 = x*x
    #       %           1 1;        -> x^1 * y^1 = x*y
    #       %           0 2;        -> x^0 * y^2 = y*y
    #       %           3 0;        -> x^3 * y^0 = x*x*x
    #       %           2 1;        -> x^2 * y^1 = x*x*y
    #       %           1 2;        -> x^1 * y^2 = x*y*y
    #       %           0 3]        -> x^0 * y^3 = y*y*y
    #       % Element (i,j) of out indicates the power to which variable j is
    #       % raised in monomial i. For example, out(5,:) = [2 1], which means
    #       % that the fifth monomial is equal to x^2*y^1 = x*x*y.

    """

    # Consider only nonlinear degrees
    degrees = degrees[degrees > 1]

    # Determine total number of combinations/monomials
    ncomb = 0
    for degree in degrees:
	    ncomb += comb(n+degree-1, degree)


    # List the exponents of each input in all monomials
    monomials = np.zeros((ncomb,n))
    idx = 0  # Running index indicating the last used row in out
    for degree in degrees:
        # All combinations in homogeneous polynomial of degree
        comb_i = _combinations(n,degree)
        ncomb_i = comb_i.shape[0]
        monomials[idx: idx+ncomb_i] = comb_i
        idx += ncomb_i

    return monomials


def _combinations(n,degree):
    """Lists the exponents of all possible terms in a homogeneous polynomial
monomial representation, e.g. [1 1 2] represents x1*x1*x2

    """

    # Number of combinations in homogeneous polynomial
    ncomb = comb(n+degree-1,degree)
    # Preallocating, and start from all ones => x1*x1*x1
    monomials = np.ones((ncomb,degree))
    for i in range(1,ncomb):
        monomials[i] = monomials[i-1]  # Copy previous monomial
        j = degree  # Index indicating which factor to change
        while monomials[i,j] == n:
            # Try to increase the last factor, but if this is not possible,
            # look the previous one that can be increased
            j -= 1
         # Increase factor j wrt previous monomial, e.g. x1*x1*x1 -> x1*x1*x2
        monomials[i,j] += 1
        # Monomial after x1*x1*xmax is x1*x2*x2, and not x1*x2*xmax
        monomials[i,j+1:degree] = monomials[i,j]

    # Exponents representation, e.g. [2 1] represents x1^2*x2 = x1*x1*x2
	combinations = np.zeros((ncomb,n))
	for i in range(ncomb):  # # Loop over all terms
		for j in range(n):  # # Loop over all inputs
            # Count the number of appearances of input j in monomial i
			combinations[i,j] = np.sum(monomials[i] == j)

    return combinations


class PNLSS(object):
    """Create polynomial nonlinear state-space model from initial linear state-space model.
    #
    #	Usage:
    #		model = fCreateNLSSmodel(A,B,C,D,nx,ny,T1,T2,sat)
    #
    #	Description:
    #		Create a polynomial nonlinear state-space model from a linear
    #		initialization with state-space matrices A, B, C, and D. The state
    #		equation is extended with a multivariate polynomial in the states
    #		and the inputs. The nonlinear degree(s) of this polynomial is/are
    #		specified in nx. Similarly, the output equation is extended with a
    #		multivariate polynomial, where the degrees are specified in ny. The
    #		transient handling is reflected in T1 (for periodic data) and T2
    #		(for aperiodic data). A saturation nonlinearity instead of a
    #		polynomial one is obsoleted; the optional parameter sat should be
    #		zero if specified.
    #
    #	Output parameters:
    #		model : structure containing the parameters and relevant data of
    #		        the polynomial nonlinear state-space model. This structure
    #		        has the following fields:
    #               A : n x n state matrix
    #               B : n x m input matrix
    #               C : p x n output matrix
    #               D : p x m feed-through matrix
    #               lin.A : n x n state matrix of the linear initialization
    #               lin.B : n x m input matrix of the linear initialization
    #               lin.C : p x n output matrix of the linear initialization
    #               lin.D : p x m feed-through matrix of the linear initialization
    #               nx : vector with nonlinear degrees in state update
    #               ny : vector with nonlinear degrees in output equation
    #               n : number of states
    #               m : number of inputs
    #               p : number of outputs
    #               xpowers : n_nx x (n+m) matrix containing the exponents of
    #                         each of the n_nx monomials in the state update
    #                         (see also fCombinations)
    #               n_nx : number of monomials in the state update
    #               E : n x n_nx matrix with polynomial coefficients in the
    #                   state update
    #               xactive : linear indices of the active elements in the
    #                         transpose of the E matrix (active elements = 
    #                         elements on which optimization will be done). By
    #                         default, all elements in the E matrix are set as
    #                         active. See fSelectActive to change this
    #                         property.
    #               ypowers : n_ny x (n+m) matrix containing the exponents of
    #                         each of the n_ny monomials in the output equation
    #                         (see also fCombinations)
    #               n_ny : number of monomials in the output equation
    #               F : p x n_ny matrix with polynomial coefficients in the
    #                   output equation
    #               yactive : linear indices of the active elements in the
    #                         transpose of the F matrix (active elements = 
    #                         elements on which optimization will be done). By
    #                         default, all elements in the F matrix are set as
    #                         active. See fSelectActive to change this
    #                         property.
    #               T1 : vector that indicates how the transient is handled for
    #                    periodic signals (see also the Input parameters) 
    #               T2 : scalar indicating how many samples from the start are
    #                    removed or vector indicating which samples are removed
    #                    (see also fComputeIndicesTransientRemovalArb)
    #               sat : obsolete, zero flag
    #               satCoeff : obsolete, n x 1 vector of ones
    #
#	Example:
#       n = 3; % Number of states
#       m = 1; % Number of inputs
#       p = 1; % Number of outputs
#       sys = drss(n,p,m); % Random linear state-space model
#       nx = [2 3]; % Quadratic and cubic terms in state equation
#       ny = [2 3]; % Quadratic and cubic terms in output equation
#       T1 = 0; % No transient handling
#       T2 = []; % No transient handling
#       sat = 0; % Obsolete parameter sat = 0
#       model = fCreateNLSSmodel(sys.a,sys.b,sys.c,sys.d,nx,ny,T1,T2,sat); % Linear state-space model
#       N = 1e3; % Number of samples
#       u = randn(N,1); % Input signal
#       y = fFilterNLSS(model,u); % Modeled output signal
#       t = 0:N-1; % Time vector
#       y_lsim = lsim(sys,u,t); % Alternative way to calculate output of linear state-space model
#       figure
#           plot(t,y_lsim,'b')
#           hold on
#           plot(t,y,'r')
#           xlabel('Time')
#           ylabel('Output')
#           legend('lsim','PNLSS')

    """
    def __init__(self, A,B,C,D,nx,ny,T1,T2):

        # nonlinear terms in state equations
        self.xpowers = combinations(n+m, nx)  # all possible terms/monomials
        self.n_nx = xpowers.shape[0]  # number of terms
        self.E = np.zeros((n,n_nx))  # polynomial coefficients
        self.xactive = E.size  # active terms

        # nonlinear terms in output equation
        self.ypowers = combinations(n+m, ny)
        self.n_ny = ypowers.shape[0]
        self.F = np.zeros((p,n_ny))
        self.yactive = F.size

        # transient handling
        self.T1 = T1  # for periodic data
        self.T2 = T2  # for unperiodic data


# Nonlinear terms
nx = [2, 3]  # Nonlinear degrees in state update equation
ny = [2, 3]  # Nonlinear degrees in output equation
whichtermsx = 'full'  # Consider all monomials in the state update equation
whichtermsy = 'full'  # Consider all monomials in the output equation

# Transient settings
NTrans = N  # Add one period before the start of each realization
#T1 = [NTrans 1+(0:N:(R-1)*N)]  # Number of transient samples and starting indices of each realization
T2 = 0  # No non-periodic transient handling


def select_active(structure,n,m,q,nx):
    """
    #FSELECTACTIVE Select active elements in E or F matrix.
    #
    #	Usage:
    #		active = fSelectActive(structure,n,m,q,nx)
    #
    #	Description:
    #		Select the active elements (i.e. those on which optimization will
    #		be done) in the E or F matrix. In particular, the linear indices
    #		(see also sub2ind and ind2sub) of the active elements in the
    #		transpose of the E or F matrix are calculated.
    #
    #	Output parameters:
    #		active : linear indices of the active elements in the transpose of
    #		         the E or F matrix
    #
    #	Input parameters:
    #		structure : string indicating which elements in the E or F matrix
    #		            are active. The possibilities are 'diagonal',
    #		            'inputsonly', 'statesonly', 'nocrossprod', 'affine',
    #		            'affinefull', 'full', 'empty', 'nolastinput', or
    #		            num2str(row_E). Below is an explanation of each of
    #		            these structures:
    #                   'diagonal' : active elements in row j of the E matrix
    #                                are those corresponding to pure nonlinear
    #                                terms in state j (only for state equation)
    #                   'inputsonly' : only terms in inputs
    #                   'statesonly' : only terms in states
    #                   'nocrossprod' : no cross-terms
    #                   'affine' : only terms that are linear in one state
    #                   'affinefull' : only terms that are linear in one state
    #                                  or constant in the states
    #                   'full' : all terms
    #                   'empty' : no terms
    #                   'nolastinput' : no terms in last input (implemented
    #                                   since version 1.1)
    #                   num2str(row_E) : only row row_E in E matrix is active
    #                                    (only for state equation)
    #       n : number of states
    #       m : number of inputs
    #       q : number of rows in corresponding E/F matrix
    #           q = n if E matrix is considered,
    #           q = p if F matrix is considered
    #       nx : degrees of nonlinearity in E/F matrix
    #
    #	Example:
    #		n = 2; % Number of states
    #       m = 1; % Number of inputs
    #       p = 1; % Number of outputs
    #       nx = 2; % Degree(s) of nonlinearity
    #       terms = fCombinations(n+m,nx); % Powers of all possible terms in n+m inputs of degree(s) nx
    #       % => terms = [2 0 0;
    #       %             1 1 0;
    #       %             1 0 1;
    #       %             0 2 0;
    #       %             0 1 1;
    #       %             0 0 2];
    #       % There are six quadratic terms in the two states x1 and x2, and
    #       % the input u, namely x1^2, x1*x2, x1*u, x2^2, x2*u, and u^2.
    #       % The matrix E is a 2 x 6 matrix that contains the polynomial
    #       % coefficients in each of these 6 terms for both state updates.
    #       % The active elements will be calculated as linear indices in the
    #       % transpose of E, hence E can be represented as
    #       % E = [e1 e2 e3 e4  e5  e6;
    #       %      e7 e8 e9 e10 e11 e12];
    #       % The matrix F is a 1 x 6 matrix that contains the polynomial
    #       % coefficients in each of the 6 terms for the output equation.
    #       % The matrix F can be represented as
    #       % F = [f1 f2 f3 f4 f5 f6];
    #
    #       % Diagonal structure
    #       activeE = fSelectActive('diagonal',n,m,n,nx);
    #       % => activeE = [1 10].';
    #       % Only e1 and e10 are active. This corresponds to a term x1^2 in
    #       % the first state equation and a term x2^2 in the second state
    #       % equation.
    #
    #       % Inputs only structure
    #       activeE = fSelectActive('inputsonly',n,m,n,nx);
    #       % => activeE = [6 12].';
    #       % Only e6 and e12 are active. This corresponds to a term u^2 in
    #       % both state equations. In all other terms, at least one of the
    #       % states (possibly raised to a certain power) is a factor.
    #       activeF = fSelectActive('inputsonly',n,m,p,nx);
    #       % => activeF = 6;
    #       % Only f6 is active. This corresponds to a term u^2 in the output
    #       % equation.
    #
    #       % States only structure
    #       activeE = fSelectActive('statesonly',n,m,n,nx);
    #       % => activeE = [1 2 4 7 8 10].';
    #       % Only e1, e2, e4, e7, e8, and e10 are active. This corresponds to
    #       % terms x1^2, x1*x2, and x2^2 in both state equations. In all other
    #       % terms, the input (possibly raised to a certain power) is a
    #       % factor.
    #
    #       % No cross products structure
    #       activeE = fSelectActive('nocrossprod',n,m,n,nx);
    #       % => activeE = [1 4 6 7 10 12].';
    #       % Only e1, e4, e6, e7, e10, and e12 are active. This corresponds to
    #       % terms x1^2, x2^2, and u^2 in both state equations. All other
    #       % terms are crossterms where more than one variable is present as a
    #       % factor.
    #
    #       % State affine structure
    #       activeE = fSelectActive('affine',n,m,n,nx);
    #       % => activeE = [3 5 9 11].';
    #       % Only e3, e5, e9, and e11 are active. This corresponds to
    #       % terms x1*u and x2*u in both state equations, since in these terms
    #       % only one state appears, and it appears linearly. 
    #
    #       % Full state affine structure
    #       activeE = fSelectActive('affinefull',n,m,n,nx);
    #       % => activeE = [3 5 6 9 11 12].';
    #       % Only e3, e5, e6, e9, e11, and e12 are active. This corresponds to
    #       % terms x1*u, x2*u and u^2 in both state equations, since in these
    #       % terms at most one state appears, and if it appears, it appears
    #       % linearly.
    #
    #       % Full structure
    #       activeE = fSelectActive('full',n,m,n,nx);
    #       % => activeE = (1:12).';
    #       % All elements in the E matrix are active.
    #
    #       % Empty structure
    #       activeE = fSelectActive('empty',n,m,n,nx);
    #       % => activeE = [];
    #       % None of the elements in the E matrix are active.
    #
    #       % One row in E matrix structure
    #       row_E = 2; % Select which row in E is active
    #       activeE = fSelectActive('row_E',n,m,n,nx);
    #       % => activeE = [7 8 9 10 11 12].';
    #       % Only the elements in the second row of E are active
    #
    #       % No terms in last input structure
    #       % This is useful in a polynomial nonlinear state-space (PNLSS)
    #       % model when considering the initial state as a parameter. The
    #       % state at time one can be estimated by adding an extra input
    #       % u_art(t) that is equal to one at time zero and zero elsewhere.
    #       % Like this, an extended PNLSS model is estimated, where the last
    #       % column in its B matrix corresponds to the state at time one in  
    #       % the original PNLSS model. To ensure that the optimization is only
    #       % carried out on the parameters of the original PNLSS model, only
    #       % the corresponding coefficients in the E/F matrix should be
    #       % selected as active.
    #       terms_extended = fCombinations(n+m+1,nx); % Powers of all possible terms with one extra input
    #       % => terms_extended = [2 0 0 0;
    #       %                      1 1 0 0;
    #       %                      1 0 1 0;
    #       %                      1 0 0 1;
    #       %                      0 2 0 0;
    #       %                      0 1 1 0;
    #       %                      0 1 0 1;
    #       %                      0 0 2 0;
    #       %                      0 0 1 1;
    #       %                      0 0 0 2];
    #       % The nonlinear terms in the extra input should not be considered
    #       % for optimization.
    #       activeE_extended = fSelectActive('nolastinput',n,m+1,n,nx);
    #       % => activeE_extended = [1 2 3 5 6 8 11 12 13 15 16 18].';
    #       % Only the terms where the last input is raised to a power zero are
    #       % active. This corresponds to the elif structure == where all terms in the
    #       % original PNLSS model are active.
    #       % The example below illustrates how to combine a certain structure
    #       % in the original model (e.g. 'nocrossprod') with the estimation of
    #       % the initial state.
    #       activeE_extended = fSelectActive('nolastinput',n,m+1,n,nx);
    #       activeE_extended = activeE_extended(fSelectActive('nocrossprod',n,m,n,nx));
    #       % => activeE_extended = [1 5 8 11 15 18].';
    #       % This corresponds to the terms x1^2, x2^2, and u1^2 in both rows
    #       % of the E_extended matrix, and thus to all terms in the original
    #       % model, except for the crossterms.
    #       % Note that an alternative approach is to include the initial state
    #       % in the parameter vector (see also fLMnlssWeighted_x0u0).
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
        active = np.where(np.sum(combis(:,:n) != 0,1) == 1 & np.sum(combis(:,n:) != 0,1) == 0)
        # Select these terms only for row j in the E matrix
        for i = 1:len(active):  # Loop over these terms
            temp = np.where(combis[active[i,0],:] != 0) # Which state variable is raised to a nonzero power
            active[i] += (temp-1)*n_nl; # Linear index of active term in transpose of E
    elif structure == 'inputsonly':
        # Find terms where all states are raised to a zero power
        active = find(sum(combis(:,1:n) ~= 0,2) == 0);
        # Select these terms for all rows in E/F matrix
        active = repmat(active,[1,q]) + repmat(0:n_nl:(q-1)*n_nl,[length(active),1]);
        active = active(:);
    elif structure == 'statesonly':
        # Find terms where all inputs are raised to a zero power
        active = find(sum(combis(:,n+1:end) ~= 0,2) == 0);
        # Select these terms for all rows in E/F matrix
        active = repmat(active,[1,q]) + repmat(0:n_nl:(q-1)*n_nl,[length(active),1]);
        active = active(:);
    elif structure == 'nocrossprod':
        # Find terms where only one variable (state or input) is raised to a nonzero power
        active = find(sum(combis ~= 0,2) == 1);
        # Select these terms for all rows in E/F matrix
        active = repmat(active,[1,q]) + repmat(0:n_nl:(q-1)*n_nl,[length(active),1]);
        active = active(:);
    elif structure == 'affine':
        # Find terms where only one state is raised to power one, and all
        # others to power zero. There are no conditions on the powers in
        # the input variables
        active = find(sum(combis(:,1:n) ~= 0,2) == 1 & sum(combis(:,1:n),2) == 1);
        # Select these terms for all rows in E/F matrix
        active = repmat(active,[1,q]) + repmat(0:n_nl:(q-1)*n_nl,[length(active),1]);
        active = active(:);
    elif structure == 'affinefull':
        # Find terms where at most one state is raised to power one, and
        # all others to power zero. There are no conditions on the powers
        # in the input variables
        active = find(sum(combis(:,1:n) ~= 0,2) <= 1 & sum(combis(:,1:n),2) <= 1);
        # Select these terms for all rows in E/F matrix
        active = repmat(active,[1,q]) + repmat(0:n_nl:(q-1)*n_nl,[length(active),1]);
        active = active(:);
    elif structure == 'full':
        # Select all terms in E/F matrix
        active = (1:q*n_nl)
    elif structure == 'empty':
        # Select no terms
        active = []
    elif structure == 'nolastinput':
        if m > 0:
            # Find terms where last input is raised to power zero
            active = find(combis(:,end) == 0);
            # Select these terms for all rows in E/F matrix
            active = repmat(active,[1,q]) + repmat(0:n_nl:(q-1)*n_nl,[length(active),1]);
            active = active(:);
        else:
            # If there is no input, then select all terms in E/F matrix
            warning('There is no input')
            active = fSelectActive('full',n,m,q,nx);
    else:
        # Check if one row in E is selected
        row_E = str2double(structure);
        if isscalar(row_E) and ismember(row_E,1:n):
            # If so, select that row
            active = (row_E-1)*n_nl + (1:n_nl)
        else:
           raise ValueError('Wrong structure {}. Should be: {}'.
                            format(structure, stype))

    # Sort the active elements
    return np.sort(active)

