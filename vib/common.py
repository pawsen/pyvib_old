#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import math
import itertools

class color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def next_pow2(i):
    """
    Find the next power of two

    >>> int(next_pow2(5))
    8
    >>> int(next_pow2(250))
    256
    """
    # do not use NumPy here, math is much faster for single values
    exponent = math.ceil(math.log(i) / math.log(2))
    # the value: int(math.pow(2, exponent))
    return exponent

def prime_factors(n):
    """Find the prime factorization of n

    Efficient implementation. Find the factorization by trial division, using
    the optimization of dividing only by two and the odd integers.

    An improvement on trial division by two and the odd numbers is wheel
    factorization, which uses a cyclic set of gaps between potential primes to
    greatly reduce the number of trial divisions. Here we use a 2,3,5-wheel

    Factoring wheels have the same O(sqrt(n)) time complexity as normal trial
    division, but will be two or three times faster in practice.

    >>> list(factors(90))
    [2, 3, 3, 5]
    """
    f = 2
    increments = itertools.chain([1,2,2], itertools.cycle([4,2,4,2,4,6,2,6]))
    for incr in increments:
        if f*f > n:
            break
        while n % f == 0:
            yield f
            n //= f
        f += incr
    if n > 1:
        yield n

def db(x):
    """relative value in dB

    TODO: Maybe x should be rescaled to ]0..1].?
    log10(0) = inf.
    """

    # dont nag if x=0
    with np.errstate(divide='ignore', invalid='ignore'):
        return 20*np.log10(np.abs(x**2))

def rescale(x, mini=None, maxi=None):
    """Rescale x to 0-1.

    If mini and maxi is given, then they are used as the values that get scaled
    to 0 and 1, respectively

    Notes
    -----
    To 0..1:
    z_i = (x_i− min(x)) / (max(x)−min(x))

    Or custom range:
    a = (maxval-minval) / (max(x)-min(x))
    b = maxval - a * max(x)
    z = a * x + b

    """
    if hasattr(x, "__len__") is False:
        return x

    if mini is None:
        mini = np.min(x)
    if maxi is None:
        maxi = np.max(x)
    return (x - mini) / (maxi - mini)

def meanVar(Y, isnoise=False):
    """
    Y = fft(y)/nsper

    Parameters
    ----------
    Y : ndarray (ndof, nsper, nper)
        Y is the fft of y
    """

    # number of periods
    p = Y.shape[2]

    # average over periods
    Ymean = np.sum(Y,axis=2) / p

    # subtract column mean from y in a broadcast way. Ie: y is 3D matrix and
    # for every 2D slice we subtract y_mean. Python automatically
    # broadcast(repeat) y_mean.
    # https://scipy.github.io/old-wiki/pages/EricsBroadcastingDoc
    Y0 = Y - Ymean[...,None]

    W = []
    # weights. Only used if the signal is noisy and multiple periods are
    # used
    if p > 1 and isnoise:
        W = np.sum(np.abs(Y0)**2, axis=2)/(p-1)

    return Ymean, W
