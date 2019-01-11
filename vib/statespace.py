#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from vib.frf import bla_periodic
from vib.subspace import (subspace, costfnc, jacobian, extract_ss)
from vib.common import (matrix_square_inv, lm)
from numpy.fft import fft
from scipy.linalg import norm
from scipy.optimize import least_squares
import numpy as np
from copy import deepcopy

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
            self.n, self.m, self.p = _get_shape()

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
        self.A, self.B, self.C, self.D, self.z, self.stable =  \
            subspace(self.G, self.covG, self.signal.freq/self.signal.fs, n, r)

        self.n, self.m, self.p = self._get_shape()

    def optimize(self, method=None, weight=True, info=True, copy=False):
        """Optimize the estimated state space matrices"""

        # Number of parameters
        n, m, p = self.n, self.m, self.p
        self.npar = n**2 + n*m + p*n + p*m

        # initial guess
        x0 = np.empty(self.npar)
        x0[:n**2] = self.A.ravel()
        x0[n**2 + np.r_[:n*m]] = self.B.ravel()
        x0[n**2 + n*m + np.r_[:n*p]] = self.C.ravel()
        x0[n**2 + n*m + n*p:] = self.D.ravel()

        if weight is not None:
            covGinvsq = np.zeros_like(self.covG)
            for f in range(self.signal.F):
                covGinvsq[f] = matrix_square_inv(self.covG[f])
            self.weight = covGinvsq

        if method is None:
            res = lm(costfnc, x0, jacobian, system=self, weight=self.weight,
                     info=info)

        else:
            res = least_squares(costfnc,x0,jacobian, method='lm',
                                x_scale='jac',
                                kwargs={'system':self,'weight':self.weight})
            # res2 = least_squares(costfnc,x0,jacobian, method='lm',
            #                      kwargs={'system':self,'weight':self.weight})


        if copy:
            nmodel = deepcopy(self)
            nmodel.A, nmodel.B, nmodel.C, nmodel.D = extract_ss(res['x'], nmodel)
            nmodel.res = res
            return nmodel

        self.A, self.B, self.C, self.D = extract_ss(res['x'], self)
        self.res = res

    def cost(self):
        n, m, p = self.n, self.m, self.p
        self.npar = n**2 + n*m + p*n + p*m
        x0 = np.empty(self.npar)
        x0[:n**2] = self.A.ravel()
        x0[n**2 + np.r_[:n*m]] = self.B.ravel()
        x0[n**2 + n*m + np.r_[:n*p]] = self.C.ravel()
        x0[n**2 + n*m + n*p:] = self.D.ravel()
        err = costfnc(x0, self, weight=self.weight)
        # divide by 2 to match scipy's implementation of minpack
        self.cost = np.dot(err, err)/2

        return self.cost

