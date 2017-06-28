#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from numpy import ndarray
from numpy.linalg import solve
import numpy as np
from scipy.sparse.csr import csr_matrix
from scipy.sparse.linalg import spsolve
import femutil
import sys
sys.path.insert(0, '../gmsh')
from gmsh import Mesh

sparse = True
sparse = False

# read mesh
mshfile = "template.msh"
mshfile = "angle_beam.msh"
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


# Materials for different physical types(idx)
# material = {
#     100 : { # interior
#         'nu' : 0.3,
#         'E' : 1.0
#     },
#     200 : { # exterior
#         'nu' : 0.3,
#         'E' : 5.0
#     }
# }
material = {
    100 : { # interior
        'nu' : 0.3,
        'E' : 1e3
    }
}

load = {
    500 : {
        'val' : -1, # -2 for template.msh
        'dir' : 1
    }
}

boundary = {
    300 : { # x-dir
        'val' : -1,
        'dir' : 0
    },
    400 : { # y-dir
        'val' : -2,
        'dir' : 1
    }
}


bdof = np.array([],dtype=int).reshape(0,2)
#% Boundary
for key in boundary:
    idx, = np.where(lines[0] == key)
    bn_tmp = lines[1][idx]
    # DOFs to remove/exclude from assembly
    bn_tmp = np.unique(bn_tmp.flatten())*2 + boundary[key]['dir']

    bdof = np.vstack((bdof,
                      np.vstack((bn_tmp,
                                 np.full(bn_tmp.shape, boundary[key]['val']))).T
    ))

#% number of eqs. Same as number of DOFs to be solved for, ie. with prescribed
# DOFs removed.
neqs = nn*2 - bdof.shape[0]

# mapping array. From DOF number to eq number. Ie. keep all eqs not in bdof
IBC = np.zeros(nn*2,dtype=int)
mask = np.ones(IBC.shape,dtype=bool)
idx = bdof[:,0]
mask[idx] = False
IBC[~mask] = -1 # not needed
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


#% loads
rhs = np.zeros((neqs))
for load_t in load:
    idx, = np.where(lines[0] == load_t)
    bn_tmp = lines[1][idx]
    bn_tmp = np.unique(bn_tmp.flatten())*2 + load[load_t]['dir']
    nloads = len(bn_tmp)
    rhs[IBC[bn_tmp]] += load[load_t]['val'] / nloads



#def dense_assem():
    """
    Assembles the global stiffness matrix K using a dense storing scheme

    """
K = np.zeros((neqs, neqs))
elem = -1
for elem_t in elmts:
    ne_t = len(elmts[elem_t][0])
    for idx in range(ne_t):
        elem += 1
        ndof = femutil.elem[elem_t]['ndof']
        nnodes = femutil.elem[elem_t]['nnodes']

        # physical properties from physical id.
        nu = material[elmts[elem_t][0][elem]]['nu']
        E = material[elmts[elem_t][0][elem]]['E']

        # nodes in element
        IELCON = elmts[elem_t][1][elem]
        # node coord
        elcoor = np.zeros([nnodes, 2])
        for j in range(nnodes):
            elcoor[j, 0] = nodes[IELCON[j], 0]
            elcoor[j, 1] = nodes[IELCON[j], 1]

        ke = femutil.ke(elem_t, elcoor, nu, E)
        dme = DME[elem, :ndof]

        for row in range(ndof):
            glob_row = dme[row]
            if glob_row != -1:
                for col in range(ndof):
                    glob_col = dme[col]
                    if glob_col != -1:
                        K[glob_row, glob_col] = K[glob_row, glob_col] +\
                                                 ke[row, col]

    #return K



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


# assemble stiffness
# if sparse:
#     K = sparse_assem
# else:
#     K = dense_assem()
U = static_solve(K,rhs)
print(U)
print(np.linalg.norm(U))


if not(np.allclose(K.dot(U)/K.max(), rhs/K.max())):
    print("The system is not in equilibrium!")
