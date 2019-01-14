.. highlight:: sh
.. currentmodule:: pyvib

Installation
============

This tutorial will walk you through the process of installing pyvib. To follow,
you really only need two basic things:

* A working `Python3 <http://www.python.org>`_ installation.
* The python packages *numpy*, *scipy*, *matplotlib*

Step 0: Install prerequisites
-----------------------------
**Recommended**:

The necessary Python packages can be installed via the Anaconda
Python distribution (https://www.anaconda.com/download/). Python 3 is needed.

NumPy, SciPy and matplotlib can be installed in Anaconda via the command::

    $ conda install numpy scipy matplotlib

**Manual**
::
    $ sudo apt install python3 python3-pip
    $ pip3 install numpy scipy matplotlib

Step 1: Download and unpack pyvib
-----------------------------------

`Download pyvib <http://pypi.python.org/pypi/pyvib>`_ and unpack it::

    $ tar xfz pyvib-VERSION.tar.gz

If you're downloading from git, say::

    $ git clone https://github.com/pawsen/pyvib.git

Step 2: Build pyvib
--------------------

Just type::

    $ cd pyvib-VERSION # if you're not there already
    #$ ./configure
    $ python setup.py install

Step 3: Test pyvib
-------------------
Just type::

    $ cd test
    $ py.test

Advanced
--------

Get better speed by linking numpy and scipy with optimised blas libraries. This
is taken care of if you uses the anaconda distribution.
