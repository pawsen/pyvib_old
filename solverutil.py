#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from numpy.linalg import solve
from scipy.sparse import coo_matrix
from scipy.sparse import identity
from scipy.sparse.csr import csr_matrix
from scipy.sparse.linalg import spsolve
import femutil
from gmsh.gmsh import Mesh


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


class Fem(object):
    def __init__(self, mshfile):
        self.nn, self.ne, self.elmts, self.lines, self.nodes = mesh_assem(mshfile)
        self.neqs = self.nn*2

    def boundary_assem(self, boundary):
        # Find boundary DOFs
        bn = []
        for key in boundary:
            idx, = np.where(self.lines[0] == key)
            bn_tmp = self.lines[1][idx]
            # DOFs to remove/exclude from assembly
            bn_tmp = np.unique(bn_tmp.flatten())*2 + int(boundary[key]['dir'])
            bn.extend(bn_tmp)
        self.N = identity(self.neqs, format='csr')
        # set all entries corresponding to dofs on the boundary to zero
        self.N[bn, bn] = 0

    def load_assem(self, load):
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
        self.rhs = np.zeros((self.neqs))
        for load_t in load:
            idx, = np.where(self.lines[0] == load_t)
            if idx.size == 0: # no loads of the given type
                continue
            bn_tmp = self.lines[1][idx]
            bn_tmp = np.unique(bn_tmp.flatten())*2 + int(load[load_t]['dir'])
            #nloads = len(bn_tmp) if len(bn_tmp) > 0 else 1
            nloads = len(bn_tmp)
            #rhs[IBC[bn_tmp]] += float(load[load_t]['val']) / nloads
            self.rhs[bn_tmp] += float(load[load_t]['val']) / nloads

        #return rhs

    def enforce_boundary(self, A, harmonic=False):
        """Enforce boundary condition according to B.12 in [1].
        The operation boils down to 2(nnz + n) scalar multiplications, additions
        or subtractions

        For the eigenvalue problem, the method depends on which range of the
        frequency spectra we're interested in. Here implemented for lowest
        frequencies. See B.16 & B.17 in [1]
        According to [2], M should be positive definite, ie. full Rank. Thus:
        K should be enforced with harmonic=True and M with harmonic=False

        [1] Jakob S. Jensen & Niels Aage: "Lecture notes: FEMVIB"
        [2] https://docs.scipy.org/doc/scipy-0.19.0/reference/generated/scipy.sparse.linalg.eigs.html
        """
        dim = A.shape
        if harmonic is True:
            return self.N.dot(A).dot(self.N)
        else:
            return self.N.dot(A).dot(self.N) + identity(dim[0]) - self.N

    def sparse_assem(self, material, systype=True, uel=None):
        """Assembles the global stiffness matrix K using a sparse storing scheme

        The scheme used to assemble is COOrdinate list (COO), and it converted to
        Compressed Sparse Row (CSR) afterward for the solution phase [1].

        Parameters
        ----------
        material : dict
          Dict with the material profiles. Key is the physical id of the surfaces
        uel      : callable function (optional)
          Python function that returns the local stiffness matrix.

        Returns
        -------
        K : ndarray (float)
          Array with the global stiffness matrix in a sparse Compressed Sparse
        Row (CSR) format.

        References
        ----------
        .. [1] Sparse matrix. In Wikipedia, The Free Encyclopedia.
            https://en.wikipedia.org/wiki/Sparse_matrix

        """
        rows = []
        cols = []
        vals = []
        elem = -1
        for elem_t in self.elmts:
            ne_t = len(self.elmts[elem_t][0])
            for idx in range(ne_t):
                elem += 1
                ndof = femutil.elem[elem_t]['ndof']
                nnodes = femutil.elem[elem_t]['nnodes']

                # nodes in element
                IELCON = self.elmts[elem_t][1][elem]
                # node coord
                elcoor = np.zeros([nnodes, 2])
                for j in range(nnodes):
                    elcoor[j, 0] = self.nodes[IELCON[j], 0]
                    elcoor[j, 1] = self.nodes[IELCON[j], 1]

                # physical properties from physical id.
                thk = float(material[self.elmts[elem_t][0][elem]]['thk'])
                if systype is True:
                    nu = float(material[self.elmts[elem_t][0][elem]]['nu'])
                    E = float(material[self.elmts[elem_t][0][elem]]['E'])
                    ke = femutil.ke(elem_t, elcoor, thk, nu, E)

                else:
                    dens = float(material[self.elmts[elem_t][0][elem]]['rho'])
                    ke = femutil.me(elem_t, elcoor, thk, dens)

                # connectivities. Nodes in element
                dofs = np.zeros((ndof))
                for i in range(len(IELCON)):
                    node = IELCON[i]
                    for l in range(2):
                        dofs[i*2 + l] = node*2 + l

                for row in range(ndof):
                    for col in range(ndof):
                        rows.append(dofs[row])
                        cols.append(dofs[col])
                        vals.append(ke[row, col])

        return coo_matrix((vals, (rows, cols)),
                          shape=(self.neqs, self.neqs)).tocsr()

    def K_assem(self, material):
        self.K = self.sparse_assem(material, systype=True)

    def M_assem(self, material):
        self.M = self.sparse_assem(material, systype=False)


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
