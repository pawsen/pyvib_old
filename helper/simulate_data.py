#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Do a randomPeriodic multisine excitation or sine sweep on the duffing eq:

 y'' + 2*beta*omega*y' + omega**2*y =
    -gam*y**3 - (mu1*y'**2 + mu2)*sign(y') - vrms*force(t)

"""

import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import forcing
import PyDSTool as dst
from matplotlib import pyplot as plt
import numpy as np
import time

# ODE parameters
savedata = True
saveplot = False
saveacc = False
omega2 = 100e3/2
beta = 10
mu1 = 0
mu2 = 0
gamma = 100e7/2
gamma = 0

# forcing parameters
vrms = 100
fs = 750
f1 = 5
f2 = 150
nsper = 8192
nper = 5
vsweep = 10
ftype = 'sweep'
#ftype = 'multisine'
inctype = ''

# get external forcing
if ftype == 'multisine':
    u, t_ran = forcing.randomPeriodic(vrms,fs, f1,f2,nsper, nper)
elif ftype == 'sweep':
    inctype='log'
    u, t_ran, _ = forcing.sineSweep(vrms,fs, f1,f2,vsweep, nper, inctype)
    # we dont use the first value, as it is not repeated when the force is
    # generated. This is taken care of in randomPeriodic.
    nsper = (len(u)-1) // nper
else:
    raise ValueError('Wrong type of forcing', ftype)

#  Initial conditions
y0 = 0  # deflection t = 0
v0 = 0  # velocity at t = 0

print( '\n parameters:')
print( 'ftype      \t = %s' % ftype)
print( 'inctype    \t = %s' % inctype)
print( 'omega(Hz)  \t = %f' % (np.sqrt(omega2)/(2*np.pi)) )
print( 'mu1        \t = %f' % mu1)
print( 'mu2        \t = %f' % mu2)
print( 'beta       \t = %f' % beta)
print( 'gamma      \t = %f' % gamma)
print( 'y0         \t = %f' % y0)
print( 'vrms       \t = %f' % vrms)
print( 'fs         \t = %d' % fs)
print( 'f1         \t = %f' % f1)
print( 'f2         \t = %f' % f2)
print( 'nsper      \t = %d' % nsper)
print( 'nper       \t = %d' % nper)
print( 'ns_tot     \t = %d' % len(u))

def recover_acc(t, y, v):
    """Recover the acceleration from the RHS:
    # a= - 2*beta*v - omega2*y -gamma*y**3 - (mu1*v**2 + mu2)*np.sign(v) - q*np.cos(OMEGA*t)
    # oneliner, but slower than the explicit for loop. Maybe because of mem-init.
    # a = np.array([DS.Rhs(pts['t'][i],state, DS.pars) for i, state in enumerate(pts)])[:,0]
    """
    a = np.empty(len(t))
    for i in range(len(t)):
        a[i] = DS.Rhs(t[i], {'y':y[i], 'v':v[i]}, DS.pars)[0]
    print('accelerations recovered')
    return a

xData = {'force': u}
my_input = dst.InterpolateTable({'tdata': t_ran,
                                 'ics': xData,
                                 'name': 'interp1d',
                                 'method': 'linear', # next 3 not necessary
                                 'checklevel': 1,
                                 'abseps': 1e-6,
                              }).compute('interp1d')

DSargs = dst.args(name='duffing_sweep')
tdomain = [t_ran[0], t_ran[-1]]
DSargs.tdata = tdomain
DSargs.inputs = my_input.variables['force']
DSargs.pars = {'omega2': omega2,
               'mu1': mu1,
               'mu2': mu2,
               'gam': gamma,
               'beta': beta}

DSargs.varspecs = {'y': 'v',
                   'v': '-2*beta*v \
                   -omega2 * y \
                   -gam * y**3 \
                   -force', #  -(mu1*v**2 + mu2)*signum(v) \
                   'inval': 'force'}
DSargs.vars = ['y', 'v']
DSargs.ics = {'y': y0, 'v': v0}
DSargs.algparams = {'init_step' :0.01, 'max_step': 0.01, 'max_pts': 200000}
DSargs.checklevel = 2

python = False
if python:
    DS = dst.Generator.Vode_ODEsystem(DSargs)
else:
    DS = dst.Generator.Dopri_ODEsystem(DSargs)

startTime = time.time()

# Because of the *stupid* way I explicit save the whole forcing repeated times,
# the integration needs to be run in steps for pydstool not to crash when nrep
# is large or the sampling high.
int_time = (t_ran[-1]-t_ran[0])/nper
t0 = 0
t1 = int_time
y, v, t, u_pol, a = [], [], [], [], []
for i in range(nper):
    DS.set(tdata=[t0, t1],
           ics={'y':y0, 'v':v0})
    traj = DS.compute('in-table')
    # Precision sampling seems to crash sometimes, when nper is high. It does
    # not seems to be a problem to run without it.
    #pts = traj.sample(dt=1/fs, precise=True)
    pts = traj.sample(dt=1/fs)
    # Dont save the last point, as it will be the first point for next round
    y.extend(pts['y'][:-1])
    v.extend(pts['v'][:-1])
    t.extend(pts['t'][:-1])
    u_pol.extend(pts['inval'][:-1])
    y0 = pts['y'][-1]
    v0 = pts['v'][-1]
    t0 = pts['t'][-1]
    t1 = t0 + int_time

y.extend([pts['y'][-1]])
v.extend([pts['v'][-1]])
t.extend([pts['t'][-1]])
u_pol.extend([pts['inval'][-1]])

print('Integration done in: {}'.format(time.time()-startTime))

if saveacc:
    a = recover_acc(t, y, v)
abspath =  os.path.dirname(os.path.realpath(sys.argv[0]))
forcing = str(vrms).replace('.','')
relpath =  '/../data/' + 'duffing_' + ftype + forcing
filename = abspath + relpath
if savedata:
    np.savez(
        filename,
        ftype=ftype,
        inctype=inctype,
        beta=beta,
        omega2=omega2,
        mu1=mu1,
        mu2=mu2,
        gamma=gamma,
        vrms=vrms,
        fs=fs,
        f1=f1,
        f2=f2,
        nsper=nsper,
        nper=nper,
        t=t,  #pts['t'],
        y=y,  #pts['y'],
        dy=v,  #pts['v'],
        ddy=a,
        u=u_pol,  #pts['inval'],
        )
    print('data saved as {}'.format(relpath))


plt.figure()
plt.plot(t, y, '-k') #, label = 'disp')
plt.xlabel('Time (t)')
plt.ylabel('Displacement (m)')
plt.title('Force type: {}, periods:{:d}'.format(ftype, nper) )
# plt.legend()

if saveplot == True:

    relpath = '/../plots/' + 'duffing_' + ftype + forcing
    filename = abspath + relpath
    plt.savefig(filename + '.pdf')
    plt.savefig(filename + '.png')
    print('plot saved as {}'.format(relpath))

    # plt.figure()
    # plt.plot(t, u_pol, label='interp')
    # plt.plot(t_ran, u, '--k', label='u')
    # plt.legend()
    # #plt.savefig('error_dopri.png')

if nper <= 10:
    plt.show()
