#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

def db(x):
    """relative value in dB

    TODO: Maybe x should be rescaled to ]0..1].?
    log10(0) = inf.
    """
    return 20*np.log10( np.abs(x **2 ))
