#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import utool as ut

(print, rrr, profile) = ut.inject2(__name__)


def wbia_cnn_main():
    ignore_prefix = []
    ignore_suffix = []
    import utool as ut

    try:
        print('[wbia_cnn] Importing wbia_cnn')
        import wbia_cnn  # NOQA

        print('[wbia_cnn] Importing ibeis')
        import wbia  # NOQA

        print('[wbia_cnn] Importing wbia_cnn._plugin')
        import wbia_cnn._plugin  # NOQA

        print('[wbia_cnn] Done importing for __main__')
    except ImportError:
        raise
        pass
    # allow for --tf flags
    ut.main_function_tester('wbia_cnn', ignore_prefix, ignore_suffix)


if __name__ == '__main__':
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    print('Checking wbia_cnn main')
    wbia_cnn_main()
