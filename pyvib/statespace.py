#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .frf import bla_periodic
from .subspace import (subspace, costfcn, jacobian, extract_ss, is_stable,
                       extract_model)
from .common import (lm, weightfcn)
from .helper.modal_plotting import plot_subspace_info, plot_subspace_model
from .pnlss import transient_indices_periodic, remove_transient_indices_periodic
from copy import deepcopy
import numpy as np
from numpy.fft import fft
from scipy.optimize import least_squares
from scipy.signal import dlsim


"""The file contains routines for LTI state space models.
"""

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

    def average(self, u=None, y=None):
        """Average over periods and flatten over realizations"""

        saveu = False
        savey = False
        if u is None:
            u = self.u
            saveu = True
        if y is None:
            y = self.y
            savey = True
        um = u.mean(axis=-1)  # (npp,m,R)
        ym = y.mean(axis=-1)
        um = um.swapaxes(1,2).reshape(-1,self.m, order='F')  # (npp*R,m)
        ym = ym.swapaxes(1,2).reshape(-1,self.p, order='F')  # (npp*R,p)

        if saveu:
            self.um = um
            # number of samples after average over periods
            self.mns = um.shape[0]  # mns = npp*R
        if savey:
            self.ym = ym

        return um, ym


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
        self.T1, self.T1 = [None]*2

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
        self.without_T2 = np.s_[:ns]

    def simulate(self, u, t=None, x0=None, T1=None, T2=None):
        """Calculate the output and the states of a nonlinear state-space model with
        transient handling.

        """
        # Number of samples
        ns = u.shape[0]
        if T1 is None:
            T1 = self.T1
            T2 = self.T2
            if T1 is not None:
                idx = self.idx_trans
        else:
            idx = transient_indices_periodic(T1, ns)

        if T1 is not None:
            # Prepend transient samples to the input
            u = u[idx]

        dt = 1/self.signal.fs
        system = (self.A, self.B, self.C, self.D, dt)
        t, y, x = dlsim(system, u, t, x0)

        if T1 is not None:
            # remove transient samples. p=1 is correct. TODO why?
            idx = remove_transient_indices_periodic(T1, ns, p=1)
            x = x[idx]
            y = y[idx]
            t = t[idx]

        return t, y, x

    def costfcn(self, weight=None):

        if weight is True:
            try:
                weight = self.weight
            except AttributeError:
                weight = weightfcn(self.covG)

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
        self.G, self.covG, self.covGn = bla_periodic(self.U, self.Y)
        self.G = self.G.transpose((2,0,1))
        if self.covG is not None:
            self.covG = self.covG.transpose((2,0,1))
        if self.covGn is not None:
            self.covGn = self.covGn.transpose((2,0,1))

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

    def optimize(self, method=None, weight=True, info=True, nmax=50, lamb=None,
                 ftol=1e-8, xtol=1e-8, gtol=1e-8, copy=False):
        """Optimize the estimated state space matrices"""

        if weight is True:
            try:
                self.weight
            except AttributeError:
                self.weight = weightfcn(self.covG)
        else:
            self.weight = weight

        x0 = self.flatten_ss()
        if method is None:
            res = lm(costfcn, x0, jacobian, system=self, weight=self.weight,
                     info=info, nmax=nmax, lamb=lamb, ftol=ftol, xtol=xtol,
                     gtol=gtol)
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
             info=True, nmax=50, lamb=None, ftol=1e-8, xtol=1e-8, gtol=1e-8):

        F = self.signal.F
        nvec = np.atleast_1d(nvec)
        maxr = maxr

        if weight is True:
            try:
                weight = self.weight
            except AttributeError:
                weight = weightfcn(self.covG)

        infodict = {}
        models = {}
        if info:
            print('Starting subspace scanning')
            print(f"n: {nvec.min()}-{nvec.max()}. r: {maxr}")
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
                cost_sub = self.costfcn(weight=weight)/F
                stable_sub = self.stable
                if optimize:
                    self.optimize(method=method, weight=weight, info=info,
                                  nmax=nmax, lamb=lamb, ftol=ftol, xtol=xtol,
                                  gtol=gtol, copy=False)

                cost = self.costfcn(weight=True)/F
                stable = is_stable(self.A, domain='z')
                infodict[n][r] = {'cost_sub': cost_sub, 'stable_sub': stable_sub,
                                  'cost': cost, 'stable': stable}
                if cost < cost_old and stable:
                    # TODO instead of dict of dict, maybe use __slots__ method of
                    # class. Slots defines attributes names that are reserved for
                    # the use as attributes for the instances of the class.
                    cost_old = cost
                    models[n] = {'A': self.A, 'B': self.B, 'C': self.C, 'D':
                                 self.D, 'r':r, 'cost':cost, 'stable': stable}

            self.models = models
            self.infodict = infodict
        return models, infodict

    def save(self, fname):
        """Save statespace representation to disk"""
        pass

    def plot_info(self, fig=None, ax=None):
        """Plot summary of subspace identification"""
        return plot_subspace_info(self.infodict, fig, ax)

    def plot_models(self):
        """Plot identified subspace models"""
        return plot_subspace_model(self.models, self.G, self.covG,
                                   self.signal.freq, self.signal.fs)

    def extract_model(self, y=None, u=None, models=None, n=None, t=None, x0=None):
        """extract the best model using validation data"""

        dt = 1/self.signal.fs
        if models is None:
            models = self.models

        if n is None:
            if y is None or u is None:
                raise ValueError('y and u cannot be None when several models'
                                 ' are given')
            model, err_vec = extract_model(models, y, u, dt, t, x0)
        elif {'A', 'B', 'C', 'D'} <= models.keys() and n is None:
            model = models
        else:
            model = models[n]
            err_vec = []

        dictget = lambda d, *k: [d[i] for i in k]
        self.A, self.B, self.C, self.D, self.r, self.stable = \
            dictget(model, 'A', 'B', 'C', 'D', 'r', 'stable')

        self.n, self.m, self.p = self._get_shape()
        # Number of parameters
        n, m, p = self.n, self.m, self.p
        self.npar = n**2 + n*m + p*n + p*m

        return err_vec
