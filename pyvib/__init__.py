#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

__minimum_python_version__ = "3.7"
if sys.version_info < tuple((int(val) for val in __minimum_python_version__.split('.'))):
    raise UnsupportedPythonError(f"sunpy does not support Python < {__minimum_python_version__}")


__version__ = "0.2.dev1"

# this indicates whether or not we are in the package's setup.py
try:
    _PYVIB_SETUP_
except NameError:
    import builtins
    builtins._PYVIB_SETUP_ = False

if not _PYVIB_SETUP_:
    from pyvib.utils.config import load_config, print_config
    from pyvib.utils.sysinfo import system_info
    # Load user configuration
    config = load_config()

    __all__ = ['config', 'system_info']
