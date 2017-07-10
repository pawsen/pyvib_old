#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import solverutil as solver
# https://wiki.python.org/moin/ConfigParserShootout
from configobj import ConfigObj
import gmsh.vtk_writer as vtk_writer
import harmonic
import matplotlib.pyplot as plt
import time

# datafile = 'meshes/cantilever_data.txt'
datafile = 'meshes/template_data.txt'
#datafile = 'meshes/eigen_beam_data.txt'
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
#sparse = False

t = time.time()

fem = solver.Fem(mshfile)

print("Number of nodes: {}".format(fem.nn))
print("Number of elements: {}".format(fem.ne))
print("Number of equations: {}".format(fem.neqs))

# setup loads
fem.load_assem(load)

fem.boundary_assem(boundary)

# assemble
fem.K_assem(material)
fem.K = fem.enforce_boundary(fem.K)
U = solver.static_solve(fem.K,fem.rhs)

elapsed = time.time() - t
print('Elapsed time: {}'.format(elapsed))

print(np.linalg.norm(U))
if not(np.allclose(fem.K.dot(U)/fem.K.max(), fem.rhs/fem.K.max())):
    print("The system is not in equilibrium!")

fem.M_assem(material)
fem.M = fem.enforce_boundary(fem.M, False)
#fem.K = fem.enforce_boundary(fem.K, True)
sys = harmonic.Solver(fem.M, fem.K, 0.0001*fem.M)

damping = False
# harmonic response/steady state
w0 = 1
w1 = 30
A = []
Omega_vec = np.linspace(w0,w1,1000)
for Omega in Omega_vec:
    u = sys.time_harmonic(Omega, fem.rhs, damping)
    A.append(max(np.abs(u)))


# Eigenvalue problem
neigs = 20

w0, w0d, psi, vesc = sys.eigen(neigs, damped=False)
#w0, w0d, psi, vesc = sys.eigen(neigs, damped=True)
f0 = sys.freqs_hz(w0)

print('undamped [rad/s]: {}'.format(w0))
print('undamped [Hz/s]: {}'.format(f0))
print('damped eigen: {}'.format(w0d))

# transient response
Omega = 3
amp = 0.1
for dof in fem.fdof:
    sys.add_force(harmonic.Sine(dof,amp,Omega))

# init conditions
x0, y0 = 0, 0
dt = 0.01
tfinal = np.pi
nsteps = int(np.floor(tfinal/dt))
print("Integration steps: {}".format(nsteps))
t = np.linspace(0, tfinal, nsteps)
x, dx, ddx = sys.integrate(x0, y0, t)

plt.figure(2)
plt.clf()
plt.plot(t, x[:,0], 'r', label='newmark')
plt.legend(loc='best')
plt.xlabel('Time (t)')
plt.ylabel('Distance (m)')

plt.figure(1)
plt.clf()
plt.plot(Omega_vec, np.log10(A))
axes = plt.gca()
ymin, ymax = axes.get_ylim()
for w in w0:
    plt.plot([w,w],[ymin,ymax],'--k')
plt.xlabel('Frequency')
plt.ylabel('Amplitude, log|D|')
plt.show()


vtufile = mshfile.split('.')[0] + '.vtu'
vtk_id = {1: 3, 2: 5, 4: 10, 15: 1, 3: 9}
Cells = {}
cdata = {}
cvdata = {}
for g_id, E in fem.elmts.items():
    if g_id not in vtk_id:
        raise NotImplementedError('vtk ids not yet implemented. Id: {}'.
                                  format(g_id))
    Cells[vtk_id[g_id]] = E[1]
    cdata[vtk_id[g_id]] = E[0]

pvdata = np.empty((fem.neqs + fem.neqs//2,1+neigs))
pvdata[::3,0] = U[:fem.neqs:2]
pvdata[1::3,0] = U[1:fem.neqs:2]
pvdata[2::3,0] = np.zeros((fem.neqs//2))
pvname = ['disp']

# always use 3d coordinates (x,y) -> (x,y,0)
for i in range(neigs):
    pvdata[::3,i+1] = vesc[:fem.neqs:2,i]
    pvdata[1::3,i+1] = vesc[1:fem.neqs:2,i]
    pvdata[2::3,i+1] = np.zeros((fem.neqs//2))
    pvname.extend(['eig{}_f: {:.3f}'.format(i, w0[i])])

vtk_writer.write_vtu(Verts=fem.nodes, Cells=Cells, cdata=cdata, pvdata=pvdata,
                     pvname=pvname, fname=vtufile)

pvdata = np.empty((fem.neqs + fem.neqs//2,1))
pvname = ['disp']
for i in range(len(t)):
    vtufile = mshfile.split('.')[0] + '_anim' + '{:04d}.vtu'.format(i)
    pvdata[::3,0] = x[i,:fem.neqs:2]
    pvdata[1::3,0] = x[i,1:fem.neqs:2]
    pvdata[2::3,0] = np.zeros((fem.neqs//2))

    vtk_writer.write_vtu(Verts=fem.nodes, Cells=Cells, cdata=cdata, pvdata=pvdata,
                         pvname=pvname, fname=vtufile)


# vtk_writer.write_vtu(Verts=nodes, Cells=Cells, pdata=pdata, pvdata=pvdata, cdata=cdata, cvdata=cvdata, pname=pname, fname=vtufile)
