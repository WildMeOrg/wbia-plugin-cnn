# -*- coding: utf-8 -*-
### __init__.py ###
# flake8: noqa
import logging
import utool as ut

ut.noinject(__name__, '[wbia_cnn.__init__]')
from wbia_cnn import models
from wbia_cnn import process
from wbia_cnn import netrun
from wbia_cnn import utils
from wbia_cnn import sandbox
from wbia_cnn import theano_ext

# from wbia_cnn import _plugin
print, print_, profile = ut.inject2(__name__, '[wbia_cnn]')
logger = logging.getLogger()

try:
    from wbia_cnn._version import __version__
except ImportError:
    __version__ = '0.0.0'


def reassign_submodule_attributes(verbose=True):
    """
    why reloading all the modules doesnt do this I don't know
    """
    import sys

    if verbose and '--quiet' not in sys.argv:
        logger.info('dev reimport')
    # Self import
    import wbia_cnn

    # Implicit reassignment.
    seen_ = set([])
    for tup in IMPORT_TUPLES:
        if len(tup) > 2 and tup[2]:
            continue  # dont import package names
        submodname, fromimports = tup[0:2]
        submod = getattr(wbia_cnn, submodname)
        for attr in dir(submod):
            if attr.startswith('_'):
                continue
            if attr in seen_:
                # This just holds off bad behavior
                # but it does mimic normal util_import behavior
                # which is good
                continue
            seen_.add(attr)
            setattr(wbia_cnn, attr, getattr(submod, attr))


def reload_subs(verbose=True):
    """Reloads wbia_cnn and submodules"""
    rrr(verbose=verbose)

    def fbrrr(*args, **kwargs):
        """fallback reload"""
        pass

    getattr(models, 'rrr', fbrrr)(verbose=verbose)
    getattr(process, 'rrr', fbrrr)(verbose=verbose)
    getattr(netrun, 'rrr', fbrrr)(verbose=verbose)
    getattr(utils, 'rrr', fbrrr)(verbose=verbose)
    rrr(verbose=verbose)
    try:
        # hackish way of propogating up the new reloaded submodule attributes
        reassign_submodule_attributes(verbose=verbose)
    except Exception as ex:
        logger.info(ex)


rrrr = reload_subs

IMPORT_TUPLES = [
    ('ibsplugin', None),
    ('models', None),
    ('process', None),
    ('netrun', None),
    ('utils', None),
]
"""
Regen Command:
    cd /home/joncrall/code/wbia_cnn/wbia_cnn
    makeinit.py -x _grave old_test old_models old_main
"""
# autogenerated __init__.py for: '/home/joncrall/code/wbia_cnn/wbia_cnn'
