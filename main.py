#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import solverutil as solver
# https://wiki.python.org/moin/ConfigParserShootout
from configobj import ConfigObj
import gmsh.vtk_writer as vtk_writer
import harmonic
import time

datafile = 'meshes/cantilever_data.txt'
datafile = 'meshes/template_data.txt'
datafile = 'meshes/template_eigen_data.txt'
datafile = 'meshes/eigen_beam_data.txt'
config = ConfigObj(datafile)

mshfile = '/'.join([datafile.split('/')[0]] + [config['mesh']['file']])
material = config['mat']
try:
    boundary = config['boundary']
except KeyError:
    boundary = {}
try:
    load = config['load']
except KeyError:
    load = {}
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


#%% Auxiliar variables computation
def complete_disp(IBC, nodes, UG):
    """Fill the displacement vectors with imposed and computed values.

    IBC : ndarray (int)
        IBC (Indicator of Boundary Conditions) indicates if the
        nodes has any type of boundary conditions applied to it.
    UG : ndarray (float)
        Array with the computed displacements.
    nodes : ndarray (float)
        Array with number and nodes coordinates

    Returns
    -------
    UC : (nnodes, 2) ndarray (float)
      Array with the displacements.

    """
    nn = nodes.shape[0]
    UC = np.zeros([nn, 2], dtype=float)
    for i in range(nn):
        for j in range(2):
            kk = IBC[i*2 + j]
            if kk == -1:
                UC[i, j] = 0.0
            else:
                UC[i, j] = UG[kk]

    return UC

UC = complete_disp(IBC, nodes, U)


M = solver.dense_mass_assem(neqs, elmts, nodes, DME, material)

C = np.zeros(M.shape)
sys = harmonic.Solver(M, C, K)
neigs = 6
w0, w0d, psi, vesc = sys.eigen(neigs)

print('undamped eigen: {}'.format(w0))
print('damped eigen: {}'.format(w0d))
#import ipdb; ipdb.set_trace()

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
    cvdata[vtk_id[g_id]] = k*np.random.rand(E[1].shape[0] * 3)
    pname = ['rand1', 'rand2', 'og en tredje']
    pdata = np.random.rand(nodes.shape[0], 3)
    pvdata = np.empty((neqs + neqs//2,neigs))
    # always use 3d coordinates (x,y) -> (x,y,0)
    for i in range(neigs):
        pvdata[::3,i] = vesc[:neqs:2,i]
        pvdata[1::3,i] = vesc[1:neqs:2,i]
        pvdata[2::3,i] = np.zeros((neqs//2))
        # pvdata[:,1] = np.hstack((vesc[:neqs,1], np.zeros((neqs//2))))


#    pvname = ['disp']
    #print('g_id: {}\nE: {}'.format(g_id, E))

vtk_writer.write_vtu(Verts=nodes, Cells=Cells, pdata=pdata, pvdata=pvdata, cdata=cdata, cvdata=cvdata, pname=pname, fname=vtufile)
