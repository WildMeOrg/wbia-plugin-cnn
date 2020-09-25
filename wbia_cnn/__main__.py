#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import utool as ut

(print, rrr, profile) = ut.inject2(__name__)
logger = logging.getLogger()


# -*- coding: utf-8 -*-
def main():  # nocover
    import wbia_cnn

    logger.info('Looks like the imports worked')
    logger.info('wbia_cnn = {!r}'.format(wbia_cnn))
    logger.info('wbia_cnn.__file__ = {!r}'.format(wbia_cnn.__file__))
    logger.info('wbia_cnn.__version__ = {!r}'.format(wbia_cnn.__version__))

    import wbia

    logger.info('wbia = {!r}'.format(wbia))

    import theano

    logger.info('theano = {!r}'.format(theano))
    logger.info('theano.__file__ = {!r}'.format(theano.__file__))
    logger.info('theano.__version__ = {!r}'.format(theano.__version__))

    import lasagne

    logger.info('lasagne = {!r}'.format(lasagne))
    logger.info('lasagne.__file__ = {!r}'.format(lasagne.__file__))
    logger.info('lasagne.__version__ = {!r}'.format(lasagne.__version__))

    try:
        import cv2

        logger.info('cv2 = {!r}'.format(cv2))
        logger.info('cv2.__file__ = {!r}'.format(cv2.__file__))
        logger.info('cv2.__version__ = {!r}'.format(cv2.__version__))
    except Exception:
        logger.info('OpenCV (cv2) failed to import')


if __name__ == '__main__':
    """
    CommandLine:
       python -m vtool
    """
    main()
