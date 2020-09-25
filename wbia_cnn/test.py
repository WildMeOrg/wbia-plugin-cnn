#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tests a test set of data using a specified, pre0trained model and weights
"""
import logging
from wbia_cnn import utils
import utool as ut
import six  # NOQA

print, rrr, profile = ut.inject2(__name__)
logger = logging.getLogger()


def test(
    data_fpath, model, weights_fpath, results_dpath=None, labels_fpath=None, **kwargs
):
    """
    Driver function

    Args:
        data_fpath (?):
        labels_fpath (?):
        model (?):
        weights_fpath (?):
    """

    ######################################################################################

    # Load the data
    logger.info('\n[data] loading data...')
    logger.info('data_fpath = %r' % (data_fpath,))
    X_test, y_test = utils.load(data_fpath, labels_fpath)
    if len(X_test.shape) == 3:
        # add channel dimension for implicit grayscale
        X_test.shape = X_test.shape + (1,)

    # return test_data(X_test, y_test, model, weights_fpath, results_dpath, **kwargs)


if __name__ == '__main__':
    """
    CommandLine:
        python -m wbia_cnn.test
        python -m wbia_cnn.test --allexamples
        python -m wbia_cnn.test --allexamples --noface --nosrc

    CommandLine:
        cd %CODE_DIR%/wbia_cnn/code
        cd $CODE_DIR/wbia_cnn/code
        code
        cd wbia_cnn/code
        python test.py

    PythonPrereqs:
        pip install theano
        pip install git+https://github.com/Lasagne/Lasagne.git
        pip install git+git://github.com/lisa-lab/pylearn2.git
        #pip install lasagne
        #pip install pylearn2
        git clone git://github.com/lisa-lab/pylearn2.git
        git clone https://github.com/Lasagne/Lasagne.git
        cd pylearn2
        python setup.py develop
        cd ..
        cd Lesagne
        git checkout 8758ac1434175159e5c1f30123041799c2b6098a
        python setup.py develop
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
