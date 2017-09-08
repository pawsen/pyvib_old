#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.sparse import coo_matrix

class NL_force(object):

    def __init__(self):
        self.nls = []
        self.dnls_force = []
        self.dnls_damp = []

    def add(self, _NL_compute):
        self.nls.append(_NL_compute)
        if _NL_compute.is_force:
            self.dnls_force.append(_NL_compute)
        else:
            self.dnls_damp.append(_NL_compute)

    def force(self, x, xd):

        if x.ndim == 1:
            ndof, = x.shape
            ns = 1
        else:
            ndof, ns = x.shape
        fnl = np.zeros((ndof+1, ns))

        for nl in self.nls:
            fnl = nl.compute(x, xd, fnl)
        # squeeze in case ns = 1
        fnl = fnl[:ndof,:].squeeze()
        return fnl

    def dforce(self, x, xd, is_force=True):
        """Derivative of nonlinear functional
        """
        if x.ndim == 1:
            ndof, = x.shape
            ns = 1
        else:
            ndof, ns = x.shape

        dfnl = np.zeros((ndof+1, ns*(ndof+1)))

        if is_force:
            for nl in self.dnls_force:
                dfnl = nl.dcompute(x, xd, dfnl)
        else:
            for nl in self.dnls_damp:
                dfnl = nl.dcompute(x, xd, dfnl)

        if ns == 1:
            return dfnl[:ndof,:ndof].squeeze()

        # create sparse structure from dfnl
        # TODO: dont create dfnl in the first place...:)
        ind = np.arange(ns*(ndof+1))
        ind = np.delete(ind, np.s_[ndof::ndof+1])
        dfnl = dfnl[:ndof, ind]
        dfnl = np.reshape(dfnl, (ndof**2,ns), order='F')
        # dont ask...
        ind = np.outer(np.ones(ndof), np.arange(ndof)) * ns*ndof + \
            np.outer(np.arange(ndof), np.ones(ndof))
        ind = np.outer(ind.T, np.ones(ns)) + \
            ns*ndof * np.outer(np.ones(ndof**2), np.arange(0,(ns-1)*ndof+1, ndof)) + \
            ndof * np.outer(np.ones(ndof**2), np.arange(ns))

        ind = ind.ravel(order='F').astype('int')

        #https://stackoverflow.com/questions/28995146/matlab-ind2sub-equivalent-in-python
        arr = ns*ndof*np.array([1,1])
        ii, jj = np.unravel_index(ind, tuple(arr), order='F')
        dfnl_s = coo_matrix((dfnl.ravel(order='F'), (ii, jj)),
                            shape=(ndof*ns, ndof*ns)).tocsr()

        return dfnl_s
        # return dfnl

    def energy(self, x, xd):
        if x.ndim == 1:
            ndof, = x.shape
            ns = 1
        else:
            ndof, ns = x.shape
        energy = 0
        for nl in self.nls:
            energy = nl.energy(x, xd, energy)
        return energy


class _NL_compute(object):
    is_force = True

    def compute(self, x, fnl):
        pass
    def dcompute(self, x, fnl):
        pass
    def energy(self, x, fnl):
        pass

class NL_polynomial(_NL_compute):
    def __init__(self, inl, enl, knl, is_force=True):
        self.inl = inl
        self.enl = enl
        self.knl = knl
        self.is_force = is_force

    def compute(self, x, xd, fnl):
        inl = self.inl
        ndof = fnl.shape[0] - 1
        idof = np.arange(ndof)
        nbln = inl.shape[0]
        if self.is_force is False:
            x = xd

        for j in range(nbln):
            # connected from
            i1 = inl[j,0]
            # conencted to
            i2 = inl[j,1]

            # Convert to the right index
            idx1 = np.where(i1 == idof)
            x1 = x[idx1]

            # if connected to ground
            if i2 == -1:
                idx2 = ([ndof],)
                x2 = 0
            else:
                idx2 = np.where(i2 == idof)
                x2 = x[idx2]
            x12 = x1 - x2
            # in case of even functional
            if (self.enl[j] % 2 == 0):
                x12 = np.abs(x12)

            f12 = self.knl[j] * x12**self.enl[j]
            fnl[idx1] += f12
            fnl[idx2] -= f12
        return fnl

    def dcompute(self, x, xd, dfnl):
        """Derivative of nonlinear functional
        """
        inl = self.inl
        ndof = dfnl.shape[0]-1
        idof = np.arange(ndof)
        nbln = inl.shape[0]
        if self.is_force is False:
            x = xd

        for j in range(nbln):
            i1 = inl[j,0]
            i2 = inl[j,1]
            idx1 = np.where(i1 == idof)[0][0]
            x1 = x[idx1]
            if i2 == -1:
                idx2 = ndof
                x2 = 0
            else:
                idx2 = np.where(i2 == idof)[0][0]
                x2 = x[idx2]
            x12 = x1 - x2
            df12 = self.knl[j] * self.enl[j] * np.abs(x12)**(self.enl[j]-1)
            # in case of even functional
            if (self.enl[j] % 2 == 0):
                idx = np.where(x12 < 0)
                df12[idx] = -df12[idx]

            #import pdb; pdb.set_trace()
            # add the nonlinear force to the right dofs
            dfnl[idx1, idx1::ndof+1] += df12
            dfnl[idx2, idx1::ndof+1] -= df12
            dfnl[idx1, idx2::ndof+1] -= df12
            dfnl[idx2, idx2::ndof+1] += df12
        return dfnl

    def energy(self, x, xd, energy):
        inl = self.inl
        if self.is_force is False:
            x = xd
        if x.ndim == 1:
            ndof, = x.shape
            ns = 1
        else:
            ndof, ns = x.shape
        idof = np.arange(ndof)
        nbln = inl.shape[0]
        for j in range(nbln):
            i1 = inl[j,0]
            i2 = inl[j,1]
            idx1 = np.where(i1 == idof)
            x1 = x[idx1]
            if i2 == -1:
                idx2 = ([ndof],)
                x2 = 0
            else:
                idx2 = np.where(i2 == idof)
                x2 = x[idx2]
            x12 = x1 - x2
            if (self.enl[j] % 2 == 0):
                x12 = np.abs(x12)
            e12 = self.knl[j] / (self.enl[j]+1) * abs(x12)**(self.enl[j]+1)
            # TODO there should be more to this if-statement
            if x12 < 0:
                pass
                #e12 = 0
            energy += e12
        return energy

class NL_tanh_damping(_NL_compute):
    def __init__(self, inl, enl, knl):
        self.inl = inl
        self.enl = enl
        self.knl = knl
        self.is_force = False

    def compute(self, x, xd, fnl):
        inl = self.inl
        ndof = fnl.shape[0] - 1
        idof = np.arange(ndof)
        nbln = inl.shape[0]
        for j in range(nbln):
            i1 = inl[j,0]
            i2 = inl[j,1]
            idx1 = np.where(i1 == idof)
            xd1 = xd[idx1]
            if i2 == -1:
                idx2 = ([ndof],)
                xd2 = 0
            else:
                idx2 = np.where(i2 == idof)
                xd2 = xd[idx2]
            xd12 = xd1 - xd2
            f12 = self.knl[j] * np.tanh(xd12 * self.enl[j])
            fnl[idx1] += f12
            fnl[idx2] -= f12
        return fnl

    def dcompute(self, x, xd, dfnl):
        inl = self.inl
        ndof = dfnl.shape[0]-1
        idof = np.arange(ndof)
        nbln = inl.shape[0]

        for j in range(nbln):
            i1 = inl[j,0]
            i2 = inl[j,1]
            idx1 = np.where(i1 == idof)[0][0]
            xd1 = xd[idx1]
            if i2 == -1:
                idx2 = ndof
                xd2 = 0
            else:
                idx2 = np.where(i2 == idof)[0][0]
                xd2 = xd[idx2]
            xd12 = xd1 - xd2
            df12 = self.enl[j] * self.knl[j] * (1 - np.tanh(xd12 * self.enl[j])**2)
            dfnl[idx1, idx1::ndof+1] += df12
            dfnl[idx2, idx1::ndof+1] -= df12
            dfnl[idx1, idx2::ndof+1] -= df12
            dfnl[idx2, idx2::ndof+1] += df12
        return dfnl

class NL_piecewise_linear(_NL_compute):
    def __init__(self, x, y, slopes, delta, symmetric, inl, is_force=True):
        self.x = x
        self.y = y
        self.slopes = slopes
        self.delta = delta
        self.symmetric = symmetric
        self.inl = inl
        self.is_force = is_force

    def compute(self, x, xd, fnl):
        inl = self.inl
        nbln = inl.shape[0]
        ndof = fnl.shape[0] - 1
        idof = np.arange(ndof)
        if self.is_force is False:
            x = xd

        for j in range(nbln):
            i1 = inl[j,0]
            i2 = inl[j,1]
            idx1 = np.where(i1 == idof)
            x1 = x[idx1]
            if i2 == -1:
                idx2 = ([ndof],)
                x2 = 0
            else:
                idx2 = np.where(i2 == idof)
                x2 = x[idx2]
            x12 = x1 - x2
            if self.symmetric[j] is True and x12 < 0:
                x12 = -x12
            f12, _ = piecewise_linear(self.x[j], self.y[j], self.slopes[j],
                                      self.delta[j], x12)
            fnl[idx1] += f12
            fnl[idx2] -= f12
        return fnl

    def dcompute(self, x, xd, dfnl):
        inl = self.inl
        ndof = dfnl.shape[0]-1
        idof = np.arange(ndof)
        nbln = inl.shape[0]
        if self.is_force is False:
            x = xd

        for j in range(nbln):
            i1 = inl[j,0]
            i2 = inl[j,1]
            idx1 = np.where(i1 == idof)[0][0]
            x1 = x[idx1]
            if i2 == -1:
                idx2 = ndof
                x2 = 0
            else:
                idx2 = np.where(i2 == idof)[0][0]
                x2 = x[idx2]
            x12 = x1 - x2
            if self.symmetric[j] is True and x12 < 0:
                x12 = -x12
            _, df12 = piecewise_linear(self.x[j], self.y[j], self.slopes[j],
                                       self.delta[j], x12)
            dfnl[idx1, idx1::ndof+1] += df12
            dfnl[idx2, idx1::ndof+1] -= df12
            dfnl[idx1, idx2::ndof+1] -= df12
            dfnl[idx2, idx2::ndof+1] += df12
        return dfnl

class NL_spline(_NL_compute):
    def __init__(self, x, coeff, symmetric, inl, is_force=True):
        self.x = x
        self.coeff = coeff
        self.symmetric = symmetric
        self.inl = inl
        self.is_force = is_force

    def compute(self, x, xd, fnl):
        inl = self.inl
        nbln = inl.shape[0]
        ndof = fnl.shape[0] - 1
        idof = np.arange(ndof)
        if self.is_force is False:
            x = xd

        for j in range(nbln):
            i1 = inl[j,0]
            i2 = inl[j,1]
            idx1 = np.where(i1 == idof)
            x1 = x[idx1]
            if i2 == -1:
                idx2 = ([ndof],)
                x2 = 0
            else:
                idx2 = np.where(i2 == idof)
                x2 = x[idx2]
            x12 = x1 - x2
            if self.symmetric[j] is True and x12 < 0:
                x12 = -x12
            f12, _ = spline_interp(self.x[j], self.coeff[j], x12)
            fnl[idx1] += f12
            fnl[idx2] -= f12
        return fnl

    def dcompute(self, x, xd, dfnl):
        inl = self.inl
        ndof = dfnl.shape[0]-1
        idof = np.arange(ndof)
        nbln = inl.shape[0]
        if self.is_force is False:
            x = xd

        for j in range(nbln):
            i1 = inl[j,0]
            i2 = inl[j,1]
            idx1 = np.where(i1 == idof)[0][0]
            x1 = x[idx1]
            if i2 == -1:
                idx2 = ndof
                x2 = 0
            else:
                idx2 = np.where(i2 == idof)[0][0]
                x2 = x[idx2]
            x12 = x1 - x2
            if self.symmetric[j] is True and x12 < 0:
                x12 = -x12
            _, df12 = spline_interp(self.x[j], self.coeff[j], x12)
            dfnl[idx1, idx1::ndof+1] += df12
            dfnl[idx2, idx1::ndof+1] -= df12
            dfnl[idx1, idx2::ndof+1] -= df12
            dfnl[idx2, idx2::ndof+1] += df12
        return dfnl


def spline_interp(x, a, xv):
    m = len(x)
    n = m - 2

    if xv < x[0]:
        iseg = True
        x0 = x[0]
        a0 = a[0]
        a1 = a[1]
        a2 = a[2]
        a3 = a[3]
        y0 = a0 + a1 * x0 + a2 * x0**2 + a3 * x0**3
        s = a1 + 2 * a2 * x0 + 3 * a3 * x0**2
        p = y0 - s * x0
        yv = s * xv + p
        yvp = s
        return yv, yvp

    if xv > x[-1]:
        x0 = x[-1]
        aa = a[4*(n-1):4*n]
        a0 = aa[0]
        a1 = aa[1]
        a2 = aa[2]
        a3 = aa[3]
        y0 = a0 + a1 * x0 + a2 * x0**2 + a3 * x0**3
        s = a1 + 2 * a2 * x0 + 3 * a3 * x0**2
        p = y0 - s * x0
        yv = s * xv + p
        yvp = s
        return yv, yvp

    iseg = np.argwhare(x >= xv)
    iseg = iseg[0] - 1
    if iseg == 0:
        iseg = 1

    aa = a[4*(iseg-1):4*iseg]
    a0 = aa
    a0 = aa[0]
    a1 = aa[1]
    a2 = aa[2]
    a3 = aa[3]

    yv = a0 + a1 * xv + a2 * xv**2 + a3 * xv**3
    yvp = a1 + 2 * a2 * xv + 3 * a3 * xv**2

    return yv, yvp


def piecewise_linear(x, y, s, d, xv):

    if any(d):
        yv, yvp = piecewise_linear_reg(x, y, s, d, xv)
        if len(yv) != 0:
            return yv, yvp

    if xv < x[0]:
        yv = s[0] * xv + y[0] - s[0] * x[1]
        yvp = s[0]
    else:
        idx = np.argwhere(x <= xv)
        ids = idx[-1]+1
        idp = idx[-1]
        yv = s[ids] * xv + y[idp] - s[ids] * x[idp]
        yvp = s[ids]

    return yv, yvp


def piecewise_linear_reg(x, y, s, d, xv):

    n = len(x)
    yv = []
    yvp = []
    for i in range(n):
        if abs(x[i] - xv) > d[i]:
            continue

        dd = d[i]
        xa = x[i] - dd
        sa = s[i]
        sb = s[i+1]
        ya = y[i] - sa * dd
        yb = y[i] + sb * dd

        t = (xv - xa) / 2 / dd
        h00 = 2 * t**3 - 3 * t**2 + 1
        h10 = t**3 - 2 * t**2 + t
        h01 = -2 * t**3 + 3 * t**2
        h11 = t**3 - t**2
        yv = h00 * ya + h10 * 2 * dd * sa + h01 * yb + h11 * 2 * dd * sb

        dh00 = 6 * t**2 - 6 * t
        dh10 = 3 * t**2 - 4 * t + 1
        dh01 = -6 * t**2 + 6 * t
        dh11 = 3 * t**2 - 2 * t
        yvp = (dh00*ya + dh10*2*dd*sa + dh01*yb + dh11*2*dd*sb) / 2 / dd
        break

    return yv, yvp
