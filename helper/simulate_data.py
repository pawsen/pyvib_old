#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""Do a swept sine excitation on the duffing eq:

 y'' + 2*beta*omega*y' + omega**2*y =
    -gam*y**3 - (mu1*y'**2 + mu2)*sign(y') - q*cos(OMEGE*t)

"""

import os
import sys
import PyDSTool as dst
import numpy as np
import time

savedata = True
omega2 = 1
beta = 0.01
mu1 = 0
mu2 = 0
q = 0.1
gamma = 2
OMEGA_start = 0.1
OMEGA_stop = 4.0

#  Initial conditions
y0 = 0  # deflection t = 0
v0 = 0.01  # velocity at t = 0

print( '\n parameters:')
print( 'omega2   \t = %f' % omega2)
print( 'q        \t = %f' % q)
print( 'mu1      \t = %f' % mu1)
print( 'mu2      \t = %f' % mu2)
print( 'beta     \t = %f' % beta)
print( 'gamma    \t = %f' % gamma)
print( 'y0       \t = %f' % y0)
print( 'v0       \t = %f' % v0)


# Set simulation parameters:
# NOTE: A lot transient periods are needed in order to get complete steady
# state. If the response curve is not smooth, try raising ntransient to
# something like 500.!

# ntransient = 500  # transient periods of ext excitation
# nsteady = 100  # state state periods of ext excitation
# sampling_rate = 25

ntransient = 200  # transient periods of ext excitation
nsteady = 50  # state state periods of ext excitation
sampling_rate = 20

n_OMEGA = 100  # numbers in excitation freq range

def recover_acc(t, y, v, OMEGA):
    a= - 2*beta*v - omega2*y -gamma*y**3 - (mu1*v**2 + mu2)*np.sign(v) - q*np.cos(OMEGA*t)
    return a

DSargs = dst.args(name='duffing')
DSargs.algparams = {'max_pts': 100000}
DSargs.pars = {'q': q,
               'omega2': omega2,
               'OMEGA': 0,
               'mu1': mu1,
               'mu2': mu2,
               'gam': gamma,
               'beta': beta}

DSargs.varspecs = {'y': 'v',
                   'v': '-2*beta*v \
                   -omega2 * y \
                   -gam * y**3 \
                   -(mu1*v**2 + mu2)*signum(v) \
                   -q*cos(OMEGA*t)'}


#lang = "python"
lang = "c"

# Create ode-object
if (lang == 'python'):
    ode = dst.Generator.Vode_ODEsystem(DSargs)
if (lang == 'c'):
    DSargs['nobuild'] = True
    ode = dst.Generator.Dopri_ODEsystem(DSargs)
    ode.makeLib()  # compile (remove gen files and dirs before recompiling)

# increase/decrease ext excitation freq
if len(sys.argv) > 1:
    arg = sys.argv[1]
    if arg == 'True':
        increasing = True
    else:
        increasing = False
else:
    increasing = False
    increasing = True

# Sweep OMEGA

path =  os.path.dirname(os.path.realpath(sys.argv[0])) +  '/../data/'
if increasing:
    filename = path + 'duffing_inc'
    OMEGA_vec = np.linspace(OMEGA_start, OMEGA_stop, n_OMEGA)
else:
    filename = path + 'duffing_dec'
    OMEGA_vec = np.linspace(OMEGA_stop, OMEGA_start, n_OMEGA)

y = []
v = []
a = []
t = []
sweep_idx = np.zeros(n_OMEGA, dtype=int)
steady_idx = np.zeros(n_OMEGA, dtype=int)

print( '\n looping OMEGA from %f to %f in %d steps' \
       % (OMEGA_vec[0], OMEGA_vec[-1], n_OMEGA))
print(' %d Sweeps with %d transient and %d steady periods.' \
      %(len(OMEGA_vec), ntransient, nsteady))
print(' Each period is sampled in %d steps, a total of %d steps\n' \
      %(sampling_rate, len(OMEGA_vec)*ntransient*nsteady*sampling_rate))

startTime = time.time()
tot = 0
t0 = 0.0  # start time for for the simulation
for i, OMEGA in enumerate(OMEGA_vec):

    print( 'OMEGA=%f' % OMEGA)

    # adjust time domain and timestep:
    ttransient = ntransient*2.0*np.pi/OMEGA + t0  # periods of the excitation force
    tstop = ttransient + nsteady*2.0*np.pi/OMEGA  # periods of the excitation force
    dt = 1 / sampling_rate  # timesteps per period of the excitation force
    #dt = 2*np.pi/OMEGA / sampling_rate  # timesteps per period of the excitation force
    # print((ttransient-t0)/dt,(tstop-ttransient)/dt)

    # set excitation frequency and update time doamain
    ode.set(pars={'OMEGA': OMEGA})

    # solve for transient motion:
    # print 'resolving transient motion'
    ode.set(tdata=[t0, ttransient],
            ics={'y': y0, 'v': v0})
    traj_transient = ode.compute('duffing')  # integrate ODE
    pts = traj_transient.sample(dt=dt, precise=True)  # sampling data for plotting
    #pts['t'] = pts['t']/np.sqrt(np.abs(omega2))  # scale back to real time
    pts_transient = pts

    # solve for steady state motion:
    ode.set(tdata=[ttransient, tstop+0.5*dt],
            ics={'y': pts['y'][-1], 'v': pts['v'][-1]})
    traj_steady = ode.compute('duffing')
    pts = traj_steady.sample(dt=dt, precise=True)
    pts_steady = pts

    # update initial conditions
    y0 = pts['y'][-1]
    v0 = pts['v'][-1]
    t0 = pts['t'][-1]

    # Save time data
    steady_idx[i] = sweep_idx[i-1] + len(pts_transient['t']) - 1
    sweep_idx[i] = steady_idx[i] + len(pts_steady['t']) - 1

    t.extend(pts_transient['t'])
    t.extend(pts_steady['t'])
    y.extend(pts_transient['y'])
    y.extend(pts_steady['y'])
    v.extend(pts_transient['v'])
    v.extend(pts_steady['v'])

    a.extend(recover_acc(pts_transient['t'], pts_transient['y'], pts_transient['v'], OMEGA))
    a.extend(recover_acc(pts_steady['t'], pts_steady['y'], pts_steady['v'], OMEGA))

    ymax = np.max(pts_steady['y'])
    ymin = np.min(pts_steady['y'])
    print( "max A: %0.5f" % (np.abs(0.5*(ymax-ymin))))

totalTime = time.time()-startTime
print('')
print(' Total time: %f, time per sweep: %f' % (totalTime,totalTime/len(OMEGA_vec)))
print(' Total points in time series: %d' % (len(t)))
y = np.asarray(y)
v = np.asarray(v)
t = np.asarray(t)


if savedata:
    np.savez(
        filename,
        beta=beta,
        omega2=omega2,
        q=q,
        mu1=mu1,
        mu2=mu2,
        gamma=gamma,
        dt=dt,
        fs=sampling_rate,
        OMEGA_vec=OMEGA_vec,
        a=a,
        v=v,
        y=y,
        t=t,
        sweep_idx=sweep_idx,
        steady_idx=steady_idx
    )
