# Copyright (c) 2021 Ivan Iakoupov
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from matplotlib import rc
import matplotlib as mpl
from cycler import cycler

def do_common_setup():
    # Set margins to 1% (instead of the default 5%)
    mpl.rcParams['axes.xmargin'] = 0.01
    mpl.rcParams['axes.ymargin'] = 0.01
    # Use Computer Modern font for math
    mpl.rcParams['mathtext.fontset'] = 'cm'
    mpl.rcParams['mathtext.rm'] = 'serif'
    # Use Computer Modern font for everything else too
    font = {'family':'serif', 'serif': ['computer modern roman']}
    rc('font',**font)

    mpl.rcParams.update({'axes.labelsize': 10})
    mpl.rcParams.update({'axes.titlesize': 10})
    mpl.rcParams.update({'legend.fontsize': 6})
    mpl.rcParams.update({'font.size': 10})
    mpl.rcParams['axes.prop_cycle'] = cycler('color', ['#aa0000', '#00aa00', '#0000aa', '#00aaaa'])

    rc('text', usetex=True)
