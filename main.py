#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import solverutil as solver
from collections import defaultdict
import time

sparse = True
sparse = False

# read mesh
mshfile = "template.msh"
#mshfile = "angle_beam.msh"

t = time.time()

# Materials for different physical types(idx)
material = defaultdict(
    lambda : { # default value
            'nu' : 0.3,
            'E' : 1e3
        },
        {
            100 : { # interior
                'nu' : 0.3,
                'E' : 1.0
            },
            200 : { # exterior
                'nu' : 0.3,
                'E' : 5.0
            }
        }
)

load = {
    500 : {
        'val' : -2, # -2 for template.msh
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

nn, ne, elmts, lines, nodes = solver.mesh_assem(mshfile)
IBC, DME, neqs = solver.operator_assem(nn, ne, elmts, lines, boundary)

print("Number of nodes: {}".format(nn))
print("Number of elements: {}".format(ne))
print("Number of equations: {}".format(neqs))

# setup loads
rhs = solver.load_assem(neqs, lines, IBC, load)

# assemble
if sparse:
    K = solver.sparse_assem(neqs, elmts, nodes, DME, material)
else:
    K = solver.dense_assem(neqs, elmts, nodes, DME, material)
U = solver.static_solve(K,rhs)

elapsed = time.time() - t
print('Elapsed time: {}'.format(elapsed))

print(np.linalg.norm(U))
if not(np.allclose(K.dot(U)/K.max(), rhs/K.max())):
    print("The system is not in equilibrium!")
