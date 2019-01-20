
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

# y = y[:,:,:2,:]
# u = u[:,:,:2,:]

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
