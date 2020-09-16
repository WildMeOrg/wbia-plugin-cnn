# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import utool as ut
from wbia_cnn import __THEANO__ as theano

(print, rrr, profile) = ut.inject2(__name__)

try:
    from Lasagne.lasagne import *  # NOQA
except ImportError as ex:
    print('Lasagne failed to import')
    print(ex)
    print('theano.__version__ = %r' % (theano.__version__,))
    print('theano.__file__ = %r' % (theano.__file__,))
    raise

from Lasagne.lasagne import layers
from Lasagne.lasagne import init
from Lasagne.lasagne import objectives
from Lasagne.lasagne import nonlinearities
