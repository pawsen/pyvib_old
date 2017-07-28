#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

def db(x):
    """relative value in dB

    TODO: Maybe x should be rescaled to ]0..1].?
    log10(0) = inf.
    """
    return 20*np.log10( np.abs(x **2 ))

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
