#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .frf import bla_periodic
from .subspace import (subspace, costfcn, jacobian, extract_ss, is_stable)
from .common import (matrix_square_inv, lm)
from .helper.modal_plotting import plot_subspace_info, plot_subspace_model
from copy import deepcopy
import numpy as np
from numpy.fft import fft
from scipy.optimize import least_squares

class Signal(object):
    def __init__(self, u, y, fs=1):
        self.u = u
        self.y = y
        self.fs = fs
        self.npp, self.m, self.R, self.P = u.shape
        self.npp, self.p, self.R, self.P = y.shape

    def lines(self, lines):
        self.lines = lines
        self.F = len(lines)
        self.freq = lines/self.npp  # Excited frequencies (normalized)

    def average(self):
        """Average over periods and flatten over realizations"""
        um = self.u.mean(axis=-1)  # (npp,m,R)
        ym = self.y.mean(axis=-1)
        self.um = um.swapaxes(1,2).reshape(-1,self.m, order='F')  # (npp*R,m)
        self.ym = ym.swapaxes(1,2).reshape(-1,self.p, order='F')  # (npp*R,p)

        # number of samples after average over periods
        self.ns = um.shape[0]  # ns = npp*R


class StateSpace(object):
    def __init__(self, *system, **kwargs):
        """Initialize the state space lti/dlti system."""

        self.A, self.B, self.C, self.D = [None]*4
        self.n, self.m, self.p = [None]*3

        N = len(system)
        if N == 4:
            self.A, self.B, self.C, self.D = system  #A, B, C, D
            self.n, self.m, self.p = self._get_shape()

        dt = kwargs.pop('dt', None)
        self.dt = dt

        self.G, self.covG, self.covGn = [None]*3

    def __repr__(self):
        """Return representation of the `StateSpace` system."""
        return '{0}(\n{1},\n{2},\n{3},\n{4},\ndt: {5}\n)'.format(
            self.__class__.__name__,
            repr(self.A),
            repr(self.B),
            repr(self.C),
            repr(self.D),
            repr(self.dt),
        )

    def _get_shape(self):
        # n, m, p
        return self.A.shape[0], self.B.shape[1], self.C.shape[0]

    def weightfcn(self):
        covGinvsq = np.empty_like(self.covG)
        for f in range(self.signal.F):
            covGinvsq[f] = matrix_square_inv(self.covG[f])
        self.weight = covGinvsq
        return self.weight

    def costfcn(self, weight=None):

        if weight is True:
            try:
                weight = self.weight
            except AttributeError:
                weight = self.weightfcn()

        x0 = self.flatten_ss()
        err = costfcn(x0, self, weight=weight)
        # TODO maybe divide by 2 to match scipy's implementation of minpack
        self.cost = np.dot(err, err)
        return self.cost

    def flatten_ss(self):
        """Returns the state space as flattened array"""

        n, m, p = self.n, self.m, self.p
        x0 = np.empty(self.npar)
        x0[:n**2] = self.A.ravel()
        x0[n**2 + np.r_[:n*m]] = self.B.ravel()
        x0[n**2 + n*m + np.r_[:n*p]] = self.C.ravel()
        x0[n**2 + n*m + n*p:] = self.D.ravel()
        return x0

    def bla(self, signal):
        """Get best linear approximation"""

        self.signal = signal
        # TODO bla expects  m, R, P, F = U.shape
        self.U = fft(signal.u, axis=0)[signal.lines].transpose((1,2,3,0))
        self.Y = fft(signal.y, axis=0)[signal.lines].transpose((1,2,3,0))
        G, covG, covGn = bla_periodic(self.U, self.Y)
        self.G = G.transpose((2,0,1))
        self.covG = covG.transpose((2,0,1))
        self.covGn = covGn.transpose((2,0,1))

        self.covY = 1

    def estimate(self, n, r, copy=False):
        """Subspace estimation"""

        self.n = n
        self.r = r

        A, B, C, D, z, stable = \
            subspace(self.G, self.covG, self.signal.freq/self.signal.fs, n, r)

        self.A, self.B, self.C, self.D, self.z, self.stable = \
            A, B, C, D, z, stable

        self.n, self.m, self.p = self._get_shape()
        # Number of parameters
        n, m, p = self.n, self.m, self.p
        self.npar = n**2 + n*m + p*n + p*m

        if copy:
            return A, B, C, D, z, stable

    def optimize(self, method=None, weight=True, info=True, copy=False, lamb=None):
        """Optimize the estimated state space matrices"""

        if weight is True:
            try:
                self.weight
            except AttributeError:
                self.weightfcn()
        else:
            self.weight = weight

        x0 = self.flatten_ss()
        if method is None:
            res = lm(costfcn, x0, jacobian, system=self, weight=self.weight,
                     info=info, lamb=lamb)
        else:
            res = least_squares(costfcn,x0,jacobian, method='lm',
                                x_scale='jac',
                                kwargs={'system':self,'weight':self.weight})
            # res2 = least_squares(costfcn,x0,jacobian, method='lm',
            #                      kwargs={'system':self,'weight':self.weight})


        if copy:
            nmodel = deepcopy(self)
            nmodel.A, nmodel.B, nmodel.C, nmodel.D = extract_ss(res['x'], nmodel)
            nmodel.res = res
            return nmodel

        self.A, self.B, self.C, self.D = extract_ss(res['x'], self)
        self.res = res

    def scan(self, nvec, maxr, optimize=True, method=None, weight=True,
             lamb=None, info=True):

        F = self.signal.F
        nvec = np.atleast_1d(nvec)
        maxr = maxr

        if weight is True:
            try:
                weight = self.weight
            except AttributeError:
                weight = self.weightfcn()

        infodict = {}
        models = {}
        if info:
            print(f"{'n':3} | {'r':3}")
        for n in nvec:
            minr = n + 1

            cost_old = np.inf
            if isinstance(maxr, (list, np.ndarray)):
                rvec = maxr[maxr >= minr]
                if len(rvec) == 0:
                    raise ValueError(f"maxr should be > {minr}. Is {maxr}")
            else:
                rvec = range(minr, maxr+1)

            infodict[n] = {}
            for r in rvec:
                if info:
                    print(f"{n:3d} | {r:3d}")

                self.estimate(n, r)
                # normalize with frequency lines to comply with matlab pnlss
                cost_sub = self.costfcn(weight=True)/F
                stable_sub = self.stable
                if optimize:
                    self.optimize(method=method, weight=weight,
                                    info=info, copy=False, lamb=lamb)

                cost = self.costfcn(weight=True)/F
                stable = is_stable(self.A, domain='z')
                infodict[n][r] = {'cost_sub': cost_sub, 'stable_sub': stable_sub,
                                  'cost': cost, 'stable': stable}
                if cost < cost_old:
                    # TODO instead of dict of dict, maybe use __slots__ method of
                    # class. Slots defines attributes names that are reserved for
                    # the use as attributes for the instances of the class.
                    cost_old = cost
                    models[n] = {'A': self.A, 'B': self.B, 'C': self.C, 'D':
                                 self.D, 'r':r, 'cost':cost, 'stable': stable}

            self.models = models
            self.infodict = infodict
        return models, infodict

    def plot_info(self):
        """Plot summary of subspace identification"""
        return plot_subspace_info(self.infodict)

    def plot_models(self):
        """Plot identified subspace models"""
        return plot_subspace_model(self.models, self.G, self.covG,
                                   self.signal.freq, self.signal.fs)

def extract(models, y, u, t=None, x0=None):
    """extract the best model using validation data"""

    dictget = lambda d, *k: [d[i] for i in k]
    err_old = np.inf
    for k, model in models.items():

        A, B, C, D = dictget(model, 'A', 'B', 'C', 'D')
        system = (A, B, C, D, dt)
        tout, yout, xout = dlsim(system, u, t, x0)
        err_rms = np.sqrt(np.mean((y - yout)**2))
        if err_rms < err_old:
            n = k
            err_old = err_rms

    return models[n]
