#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""Do a stepped sine excitation on the duffing eq:

 y'' + 2*beta*omega*y' + omega**2*y =
    -gam*y**3 - (mu1*y'**2 + mu2)*sign(y') - q*cos(OMEGE*t)

"""

import os
import sys
import PyDSTool as dst
import numpy as np
import time

savedata = True
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

OMEGA_start = f1 *2*np.pi
OMEGA_stop = f2 *2*np.pi

#  Initial conditions
y0 = 0  # deflection t = 0
v0 = 0.01  # velocity at t = 0

print( '\n parameters:')
print( 'omega2   \t = %f' % omega2)
print( 'vrms     \t = %f' % vrms)
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

ntransient = 200  # transient periods of ext excitation
nsteady = 50  # state state periods of ext excitation
n_OMEGA = 100  # numbers in excitation freq range


DSargs = dst.args(name='duffing_stepped')
DSargs.algparams = {'max_pts': 100000}
DSargs.pars = {'vrms': vrms,
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
                   -vrms*cos(OMEGA*t)'}


#lang = "python"
lang = "c"

# Create ode-object
if (lang == 'python'):
    ode = dst.Generator.Vode_ODEsystem(DSargs)
if (lang == 'c'):
    ode = dst.Generator.Dopri_ODEsystem(DSargs)

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

abspath =  os.path.dirname(os.path.realpath(sys.argv[0]))
filename = abspath + '/../data/' + 'duffing_stepped' + str(vrms)
if increasing:
    filename = filename + '_inc'
    OMEGA_vec = np.linspace(OMEGA_start, OMEGA_stop, n_OMEGA)
else:
    filename = filename + '_dec'
    OMEGA_vec = np.linspace(OMEGA_stop, OMEGA_start, n_OMEGA)

y = []
v = []
a = []
t = []
sweep_idx = np.zeros(n_OMEGA, dtype=int)
steady_idx = np.zeros(n_OMEGA, dtype=int)

print( '\n looping OMEGA from %f to %f in %d steps' \
       % (OMEGA_vec[0]/(2*np.pi), OMEGA_vec[-1]/(2*np.pi), n_OMEGA))
print(' %d Sweeps with %d transient and %d steady periods.' \
      %(len(OMEGA_vec), ntransient, nsteady))
# TODO: This is not correct longer:
print(' Each period is sampled in %d steps, a total of %d steps\n' \
      %(fs, len(OMEGA_vec)*ntransient*nsteady*fs))

startTime = time.time()
tot = 0
t0 = 0.0  # start time for for the simulation
for i, OMEGA in enumerate(OMEGA_vec):
    print( 'OMEGA=%f Hz' % (OMEGA/(2*np.pi)))

    # periods of the excitation force
    ttransient = ntransient*2.0*np.pi/OMEGA + t0
    tstop = ttransient + nsteady*2.0*np.pi/OMEGA
    # I prefer constant sampling rate. The other adjust the sampling rate to
    # the excitation frequency. Ie. each period will have the same points.
    dt = 1/fs
    #dt = 2*np.pi/OMEGA / fs

    # set excitation frequency and update time doamain
    ode.set(pars={'OMEGA': OMEGA})
    ode.set(tdata=[t0, ttransient],
            ics={'y': y0, 'v': v0})
    traj_transient = ode.compute('duffing')  # integrate ODE
    pts = traj_transient.sample(dt=dt) #", precise=True)  # sampling data for plotting
    pts_transient = pts

    # solve for steady state motion:
    ode.set(tdata=[ttransient, tstop+0.5*dt],
            ics={'y': pts['y'][-1], 'v': pts['v'][-1]})
    traj_steady = ode.compute('duffing')
    pts = traj_steady.sample(dt=dt)  #, precise=True)
    pts_steady = pts

    # update initial conditions
    y0 = pts['y'][-1]
    v0 = pts['v'][-1]
    t0 = pts['t'][-1]

    # Save index for steady state and new sweep/OMEGA
    # Note that sweep_idx really is one ahead. Ie. sweep_idx[0] is the idx for
    # the second sweep, etc.
    steady_idx[i] = sweep_idx[i-1] + len(pts_transient['t']) - 1
    sweep_idx[i] = steady_idx[i] + len(pts_steady['t']) - 1

    # Dont include the same point twice
    t.extend(pts_transient['t'][:-1])
    t.extend(pts_steady['t'][:-1])
    y.extend(pts_transient['y'][:-1])
    y.extend(pts_steady['y'][:-1])
    v.extend(pts_transient['v'][:-1])
    v.extend(pts_steady['v'][:-1])

    ymax = np.max(pts_steady['y'])
    ymin = np.min(pts_steady['y'])
    print( "max A: %0.5f" % (np.abs(0.5*(ymax-ymin))))

# But save the last point
t.extend([pts_transient['t'][-1]])
t.extend([pts_steady['t'][-1]])
y.extend([pts_transient['y'][-1]])
y.extend([pts_steady['y'][-1]])
v.extend([pts_transient['v'][-1]])
v.extend([pts_steady['v'][-1]])

totalTime = time.time()-startTime
print('')
print(' Total time: %f, time per sweep: %f' % (totalTime,totalTime/len(OMEGA_vec)))
print(' Total points in time series: %d' % (len(t)))


if savedata:
    np.savez(
        filename,
        beta=beta,
        omega2=omega2,
        vrms=vrms,
        mu1=mu1,
        mu2=mu2,
        gamma=gamma,
        dt=dt,
        fs=fs,
        OMEGA_vec=OMEGA_vec,
        a=a,
        v=v,
        y=y,
        t=t,
        sweep_idx=sweep_idx,
        steady_idx=steady_idx,
    )
    print('data saved as {}'.format(filename))
