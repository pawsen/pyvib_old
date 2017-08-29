#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.sparse import coo_matrix

def force_nl(x, inl, knl, enl):
    if inl.size == 0:
        return np.array([0])

    # ns = Nt
    ndof, ns = x.shape
    nbln = inl.shape[0]
    idof = np.arange(ndof)

    fnl = np.zeros((ndof+1, ns))
    x = np.vstack((x, np.zeros((1,ns)) ))
    nbln = inl.shape[0]

    for j in range(nbln):
        # connected from
        i1 = inl[j,0]
        # conencted to
        i2 = inl[j,1]

        # Convert to the right index
        idx1 = np.where(i1==idof)
        x1 = x[idx1]

        # if connected to ground
        if i2 == -1:
            idx2 = ([ndof],)
            x2 = 0
        else:
            idx2 = np.where(i2==idof)
            x2 = x[idx2]
        x12 = x1 - x2
        # in case of even functional
        if (enl[j] % 2 == 0):
            x12 = np.abs(x12)

        f12 = knl[j] * x12**enl[j]
        fnl[idx1] += f12
        fnl[idx2] -= f12
    fnl = fnl[:ndof,:]
    return fnl

def der_force_nl(x, inl, knl, enl):
    """Derivative of nonlinear functional
    """
    if inl.size == 0:
        return np.array([])

    ndof, ns = x.shape
    nbln = inl.shape[0]
    idof = np.arange(ndof)

    dfnl = np.zeros((ndof+1, ns*(ndof+1)))
    x = np.vstack((x, np.zeros((1,ns)) ))
    for j in range(nbln):
        # connected from
        i1 = inl[j,0]
        # conencted to
        i2 = inl[j,1]

        # Convert to the right index
        idx1 = np.where(i1==idof)[0][0]
        x1 = x[idx1]

        # if connected to ground
        if i2 == -1:
            idx2 = ndof
            x2 = 0
        else:
            idx2 = np.where(i2==idof)[0][0]
            x2 = x[idx2]
        x12 = x1 - x2

        df12 = knl[j] * enl[j] * np.abs(x12)**(enl[j]-1)

        # in case of even functional
        if (enl[j] % 2 == 0):
            idx = np.where(x12 < 0)
            df12[idx] = -df12[idx]
            #x12 = np.abs(x12)

        # add the nonlinear force to the right dofs
        dfnl[idx1, idx1::ndof+1] += df12
        dfnl[idx2, idx1::ndof+1] -= df12
        dfnl[idx1, idx2::ndof+1] -= df12
        dfnl[idx2, idx2::ndof+1] += df12

    if ns == 1:
        return dfnl[:dof,:dof]


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

