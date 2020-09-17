#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import utool as ut

(print, rrr, profile) = ut.inject2(__name__)


# -*- coding: utf-8 -*-
def main():  # nocover
    import wbia_cnn

    print('Looks like the imports worked')
    print('wbia_cnn = {!r}'.format(wbia_cnn))
    print('wbia_cnn.__file__ = {!r}'.format(wbia_cnn.__file__))
    print('wbia_cnn.__version__ = {!r}'.format(wbia_cnn.__version__))

    import wbia

    print('wbia = {!r}'.format(wbia))

    try:
        import theano

        print('theano = {!r}'.format(theano))
        print('theano.__file__ = {!r}'.format(theano.__file__))
        print('theano.__version__ = {!r}'.format(theano.__version__))
    except Exception:
        print('Theano (theano) failed to import')

    try:
        import lasagne

        print('lasagne = {!r}'.format(lasagne))
        print('lasagne.__file__ = {!r}'.format(lasagne.__file__))
        print('lasagne.__version__ = {!r}'.format(lasagne.__version__))
    except Exception:
        print('Lasagne (lasagne) failed to import')

    try:
        import cv2

        print('cv2 = {!r}'.format(cv2))
        print('cv2.__file__ = {!r}'.format(cv2.__file__))
        print('cv2.__version__ = {!r}'.format(cv2.__version__))
    except Exception:
        print('OpenCV (cv2) failed to import')


if __name__ == '__main__':
    """
    CommandLine:
       python -m vtool
    """
    main()
