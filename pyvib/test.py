#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from scipy import signal

class StateSpace(signal.LinearTimeInvariant):
    def __new__(cls, *system, **kwargs):
        """Create new StateSpace object and settle inheritance."""
        # Handle object conversion if input is an instance of `lti`
        if len(system) == 1 and isinstance(system[0], LinearTimeInvariant):
            return system[0].to_ss()

        # Choose whether to inherit from `lti` or from `dlti`
        if cls is StateSpace:
            if kwargs.get('dt') is None:
                return StateSpaceContinuous.__new__(StateSpaceContinuous,
                                                    *system, **kwargs)
            else:
                return StateSpaceDiscrete.__new__(StateSpaceDiscrete,
                                                  *system, **kwargs)

        # No special conversion needed
        return super(StateSpace, cls).__new__(cls)

    def __init__(self, *system, **kwargs):
        """Initialize the state space lti/dlti system."""
        # Conversion of lti instances is handled in __new__
        if isinstance(system[0], LinearTimeInvariant):
            return

        # Remove system arguments, not needed by parents anymore
        super(StateSpace, self).__init__(**kwargs)

        self._A = None
        self._B = None
        self._C = None
        self._D = None

        self.A, self.B, self.C, self.D = abcd_normalize(*system)



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

sig = Signal(u,y)
sig.lines(lines)
linmodel = linss()
linmodel.bla(sig)
linmodel.estimate(n,r)
linmodel.optimize(copy=False)
sig.average()

model = StateSpace(A_data, B_data, C_data, D_data)
model.signal = linmodel.signal
system = model
self = model
model.nlterms('x', [2,3], 'full')
model.nlterms('y', [2,3], 'full')

# samples per period
npp, F = self.signal.npp, self.signal.F
R, P = self.signal.R, self.signal.P

# transient settings
# Add one period before the start of each realization
nt = npp
T1 = np.r_[nt, np.r_[0:(R-1)*npp+1:npp]]
T2 = 0
model.transient(T1,T2)
model.optimize(lamb=100, weight=W_data.T)



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


# Number of frequency bins where weighting is specified (ie nfd = floor(npp/2),
# where npp is the number of samples in one period and one phase realization
# for a multisine excitation
nfd = weight.shape[0]
freq_weight = True

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
