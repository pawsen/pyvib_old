#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from numpy.linalg import solve
from scipy.sparse import coo_matrix
from scipy.sparse.csr import csr_matrix
from scipy.sparse.linalg import spsolve
import femutil
# import sys
# sys.path.insert(0, '../gmsh')
from gmsh.gmsh import Mesh

def operator_assem(nn, ne, elmts, lines, boundary):
    # Find boundary DOFs
    bdof = np.array([],dtype=int).reshape(0,2)
    #% Boundary
    for key in boundary:
        idx, = np.where(lines[0] == key)
        bn_tmp = lines[1][idx]
        # DOFs to remove/exclude from assembly
        bn_tmp = np.unique(bn_tmp.flatten())*2 + int(boundary[key]['dir'])

        # TODO: fix np.full command: explicit int-casting
        bdof = np.vstack((bdof,
                          np.vstack((bn_tmp,
                                     np.full(bn_tmp.shape,
                                             int(boundary[key]['val']),dtype=int)
                          )).T))

    #% number of eqs. Same as number of DOFs to be solved for, ie. with prescribed
    # DOFs removed.
    neqs = nn*2 - bdof.shape[0]

    # mapping array. From DOF number to eq number. Ie. keep all eqs not in bdof
    IBC = np.zeros(nn*2,dtype=int)
    mask = np.ones(IBC.shape,dtype=bool)
    idx = bdof[:,0]
    mask[idx] = False
    IBC[~mask] = -1 # Only works as =-1. TODO
    IBC[mask] = np.arange(neqs,dtype=int)

    # Assembly operator
    DME = np.zeros((ne, 18),dtype=int)
    IELCON = np.zeros([ne, 9], dtype=int)

    # get current element type
    elem = -1
    for elem_t in elmts:
        ne_t = len(elmts[elem_t][0])
        for idx in range(ne_t):
            elem += 1
            ndof = femutil.elem[elem_t]['ndof']
            nnodes = femutil.elem[elem_t]['nnodes']
            for j in range(nnodes):
                # connectivities. Nodes in element
                IELCON[elem, j] = elmts[elem_t][1][elem,j]
                kk = IELCON[elem, j]
                for l in range(2):
                    DME[elem, 2*j+l] = IBC[kk*2 + l]

    return IBC, DME, neqs


def mesh_assem(mshfile):
    mesh = Mesh()
    mesh.read_msh(mshfile)

    # extract all elements we support and calculate
    elmts = {}
    ne = 0
    for key in mesh.Elmts:
        if key in femutil.elem:
            elmts[key] = mesh.Elmts[key]
            ne += len(elmts[key][0])
    # elmts = mesh.Elmts[2]
    # ne = elmts[1].shape[0]
    nodes = mesh.Verts
    nn = nodes.shape[0]

    # extract all lines/nodal loads/forces
    lines = mesh.Elmts[1]
    return nn, ne, elmts, lines, nodes

def load_assem(neqs, lines, IBC, load):
    """Assembles the global Right Hand Side Vector RHS

    Parameters
    ----------
    neq : int
      Number of equations in the system after removing the nodes
      with imposed displacements.
    IBC : ndarray (int)
      Array that maps the nodes with number of equations.
    load : dict
      Dict with loads physical id as key, returning the load size/val

    Returns
    -------
    RHSG : ndarray
      Array with the right hand side vector.

    """
    rhs = np.zeros((neqs))
    for load_t in load:
        idx, = np.where(lines[0] == load_t)
        if idx.size == 0: # no loads of the given type
            continue
        bn_tmp = lines[1][idx]
        bn_tmp = np.unique(bn_tmp.flatten())*2 + int(load[load_t]['dir'])
        #nloads = len(bn_tmp) if len(bn_tmp) > 0 else 1
        nloads = len(bn_tmp)
        rhs[IBC[bn_tmp]] += float(load[load_t]['val']) / nloads

    return rhs


def dense_assem(neqs, elmts, nodes, DME, material, uel=None):
    """Assembles the global stiffness matrix K using a dense storing scheme
    """
    K = np.zeros((neqs, neqs))
    elem = -1
    for elem_t in elmts:
        # number of elements in the current type of elements
        ne_t = len(elmts[elem_t][0])
        for idx in range(ne_t):
            elem += 1
            ndof = femutil.elem[elem_t]['ndof']
            nnodes = femutil.elem[elem_t]['nnodes']

            # physical properties from physical id.
            nu = float(material[elmts[elem_t][0][elem]]['nu'])
            E = float(material[elmts[elem_t][0][elem]]['E'])
            thk = float(material[elmts[elem_t][0][elem]]['thk'])

            # nodes in element
            IELCON = elmts[elem_t][1][elem]
            # node coord
            elcoor = np.zeros([nnodes, 2])
            for j in range(nnodes):
                elcoor[j, 0] = nodes[IELCON[j], 0]
                elcoor[j, 1] = nodes[IELCON[j], 1]

            ke = femutil.ke(elem_t, elcoor, thk, nu, E)
            dme = DME[elem, :ndof]

            for row in range(ndof):
                glob_row = dme[row]
                if glob_row != -1:
                    for col in range(ndof):
                        glob_col = dme[col]
                        if glob_col != -1:
                            K[glob_row, glob_col] = K[glob_row, glob_col] +\
                                                     ke[row, col]

    return K


def sparse_assem(neqs, elmts, nodes, DME, material, uel=None):
    """Assembles the global stiffness matrix K using a sparse storing scheme

    The scheme used to assemble is COOrdinate list (COO), and it converted to
    Compressed Sparse Row (CSR) afterward for the solution phase [1].

    Parameters
    ----------
    neqs     : int
      Number of active equations in the system.
    elmts    : dict
      Array with the number for the nodes in each element.
    nodes    : ndarray (float)
      Array with the nodal numbers and coordinates.
    DME      : ndarray (int)
      Assembly operator.
    material : dict
      Dict with the material profiles. Key is the physical id of the surfaces
    uel      : callable function (optional)
      Python function that returns the local stiffness matrix.

    Returns
    -------
    K : ndarray (float)
      Array with the global stiffness matrix in a sparse
      Compressed Sparse Row (CSR) format.

    References
    ----------
    .. [1] Sparse matrix. (2017, March 8). In Wikipedia, The Free Encyclopedia.
        https://en.wikipedia.org/wiki/Sparse_matrix

    """
    rows = []
    cols = []
    vals = []
    elem = -1
    for elem_t in elmts:
        ne_t = len(elmts[elem_t][0])
        for idx in range(ne_t):
            elem += 1
            ndof = femutil.elem[elem_t]['ndof']
            nnodes = femutil.elem[elem_t]['nnodes']

            # physical properties from physical id.
            nu = float(material[elmts[elem_t][0][elem]]['nu'])
            E = float(material[elmts[elem_t][0][elem]]['E'])
            thk = float(material[elmts[elem_t][0][elem]]['thk'])

            # nodes in element
            IELCON = elmts[elem_t][1][elem]
            # node coord
            elcoor = np.zeros([nnodes, 2])
            for j in range(nnodes):
                elcoor[j, 0] = nodes[IELCON[j], 0]
                elcoor[j, 1] = nodes[IELCON[j], 1]

            ke = femutil.ke(elem_t, elcoor, thk, nu, E)
            dme = DME[elem, :ndof]

            for row in range(ndof):
                glob_row = dme[row]
                if glob_row != -1:
                    for col in range(ndof):
                        glob_col = dme[col]
                        if glob_col != -1:
                            rows.append(glob_row)
                            cols.append(glob_col)
                            vals.append(ke[row, col])

    return coo_matrix((vals, (rows, cols)), shape=(neqs, neqs)).tocsr()


def dense_mass_assem(neqs, elmts, nodes, DME, material, uel=None):
    M = np.zeros((neqs, neqs))
    elem = -1
    for elem_t in elmts:
        # number of elements in the current type of elements
        ne_t = len(elmts[elem_t][0])
        for idx in range(ne_t):
            elem += 1
            ndof = femutil.elem[elem_t]['ndof']
            nnodes = femutil.elem[elem_t]['nnodes']

            # physical properties from physical id.
            rho = float(material[elmts[elem_t][0][elem]]['rho'])
            thk = float(material[elmts[elem_t][0][elem]]['thk'])

            # nodes in element
            IELCON = elmts[elem_t][1][elem]
            # node coord
            elcoor = np.zeros([nnodes, 2])
            for j in range(nnodes):
                elcoor[j, 0] = nodes[IELCON[j], 0]
                elcoor[j, 1] = nodes[IELCON[j], 1]

            me = femutil.me(elem_t, elcoor, thk, rho)
            dme = DME[elem, :ndof]

            for row in range(ndof):
                glob_row = dme[row]
                if glob_row != -1:
                    for col in range(ndof):
                        glob_col = dme[col]
                        if glob_col != -1:
                            M[glob_row, glob_col] = M[glob_row, glob_col] +\
                                                     me[row, col]

    return M


def mass_lump(neqs, elmts, nodes, DME, material, uel=None):
    """HRZ - mass lumping, in COOK, s. 380

    """
    M = np.zeros((neqs))
    elem = -1
    for elem_t in elmts:
        # number of elements in the current type of elements
        ne_t = len(elmts[elem_t][0])
        for idx in range(ne_t):
            elem += 1
            ndof = femutil.elem[elem_t]['ndof']
            nnodes = femutil.elem[elem_t]['nnodes']
            me_lumped = np.zeros((ndof))
            # physical properties from physical id.
            rho = float(material[elmts[elem_t][0][elem]]['rho'])
            thk = float(material[elmts[elem_t][0][elem]]['thk'])

            # nodes in element
            IELCON = elmts[elem_t][1][elem]
            # node coord
            elcoor = np.zeros([nnodes, 2])
            for j in range(nnodes):
                elcoor[j, 0] = nodes[IELCON[j], 0]
                elcoor[j, 1] = nodes[IELCON[j], 1]

            me = femutil.me(elem_t, elcoor, thk, rho)
            dme = DME[elem, :ndof]

            # do the lumping
            for i in range(ndof):
                me_lumped[i] = me[i,i]
            me_tot = np.sum(me)  # m in cook
            me_diag_sum = np.sum(me_lumped)  # S in COOK, s. 380
            me_lumped = (me_tot / me_diag_sum) * me_lumped

            for row in range(ndof):
                glob_row = dme[row]
                if glob_row != -1:
                    M[glob_row] = M[glob_row] + me_lumped[glob_row]

    return M


def static_solve(mat, rhs):
    """Solve a static problem [mat]{u_sol} = {rhs}
    """
    if type(mat) is csr_matrix:
        u_sol = spsolve(mat, rhs)
    elif type(mat) is np.ndarray:
        u_sol = solve(mat, rhs)
    else:
        raise ValueError("Not supported matrix storage scheme! {}".format(type(mat)))

    return u_sol
