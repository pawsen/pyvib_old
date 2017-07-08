#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import solverutil as solver
# https://wiki.python.org/moin/ConfigParserShootout
from configobj import ConfigObj
import gmsh.vtk_writer as vtk_writer
import time

datafile = 'meshes/cantilever_data.txt'
datafile = 'meshes/template_data.txt'
config = ConfigObj(datafile)

mshfile = '/'.join([datafile.split('/')[0]] + [config['mesh']['file']])
material = config['mat']
boundary = config['boundary']
load = config['load']
material = {int(k):v for k,v in material.items()}
boundary = {int(k):v for k,v in boundary.items()}
load = {int(k):v for k,v in load.items()}

sparse = True
sparse = False

t = time.time()


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

vtufile = mshfile.split('.')[0] + '.vtu'
vtk_id = {1: 3, 2: 5, 4: 10, 15: 1, 3: 9}
Cells = {}
cdata = {}
cvdata = {}
k = 0.0
for g_id, E in elmts.items():
    k += 1.0
    if g_id not in vtk_id:
        raise NotImplementedError('vtk ids not yet implemented. Id: {}'.format(g_id))
    Cells[vtk_id[g_id]] = E[1]
    cdata[vtk_id[g_id]] = k*np.random.rand((E[1].shape[0]))
    cvdata[vtk_id[g_id]] =k*np.random.rand(E[1].shape[0] * 3)
    pdata = np.random.rand(nodes.shape[0], 3)

vtk_writer.write_vtu(Verts=nodes, Cells=Cells, pdata=pdata, cdata=cdata, cvdata=cvdata, fname=vtufile)

