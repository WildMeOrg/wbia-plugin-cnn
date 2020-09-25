#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
References:
    http://deeplearning.net/software/theano/tutorial/using_gpu.html#testing-theano-with-gpu

python `python -c "import os, theano; print os.path.dirname(theano.__file__)"`/misc/check_blas.py
"""
import logging
import theano

# from theano import function, config, shared, sandbox
import theano.tensor as T
import numpy
import time
import utool as ut

(print, rrr, profile) = ut.inject2(__name__)
logger = logging.getLogger()


def test_theano():
    vlen = 10 * 30 * 768  # 10 x #cores x # threads per core
    iters = 1000

    rng = numpy.random.RandomState(22)
    x = theano.shared(numpy.asarray(rng.rand(vlen), theano.config.floatX))
    f = theano.function([], T.exp(x))
    logger.info(f.maker.fgraph.toposort())
    t0 = time.time()
    for i in range(iters):
        r = f()
    t1 = time.time()
    logger.info('Looping %d times took' % iters, t1 - t0, 'seconds')
    logger.info('Result is', r)

    if numpy.any([isinstance(x_.op, T.Elemwise) for x_ in f.maker.fgraph.toposort()]):
        logger.info('Used the cpu')
    else:
        logger.info('Used the gpu')


if __name__ == '__main__':
    """
    python $CODE_DIR/wbia_cnn/test_theano.py
    """
    test_theano()
    # theano.test()
