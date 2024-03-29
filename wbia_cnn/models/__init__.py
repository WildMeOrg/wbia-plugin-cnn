# -*- coding: utf-8 -*-
# Autogenerated on 11:47:37 2016/08/23
# flake8: noqa
import logging
from wbia_cnn.models import _model_legacy
from wbia_cnn.models import abstract_models
from wbia_cnn.models import aoi2
from wbia_cnn.models import background
from wbia_cnn.models import classifier
from wbia_cnn.models import classifier2
from wbia_cnn.models import labeler
from wbia_cnn.models import dummy
from wbia_cnn.models import mnist
from wbia_cnn.models import pretrained
from wbia_cnn.models import quality
from wbia_cnn.models import siam
from wbia_cnn.models import viewpoint
import utool

print, rrr, profile = utool.inject2(__name__, '[wbia_cnn.models]')
logger = logging.getLogger()


def reassign_submodule_attributes(verbose=True):
    """
    why reloading all the modules doesnt do this I don't know
    """
    import sys

    if verbose and '--quiet' not in sys.argv:
        print('dev reimport')
    # Self import
    import wbia_cnn.models

    # Implicit reassignment.
    seen_ = set([])
    for tup in IMPORT_TUPLES:
        if len(tup) > 2 and tup[2]:
            continue  # dont import package names
        submodname, fromimports = tup[0:2]
        submod = getattr(wbia_cnn.models, submodname)
        for attr in dir(submod):
            if attr.startswith('_'):
                continue
            if attr in seen_:
                # This just holds off bad behavior
                # but it does mimic normal util_import behavior
                # which is good
                continue
            seen_.add(attr)
            setattr(wbia_cnn.models, attr, getattr(submod, attr))


def reload_subs(verbose=True):
    """Reloads wbia_cnn.models and submodules"""
    if verbose:
        print('Reloading submodules')
    rrr(verbose=verbose)

    def wrap_fbrrr(mod):
        def fbrrr(*args, **kwargs):
            """fallback reload"""
            if verbose:
                print('No fallback relaod for mod=%r' % (mod,))
            # Breaks ut.Pref (which should be depricated anyway)
            # import imp
            # imp.reload(mod)

        return fbrrr

    def get_rrr(mod):
        if hasattr(mod, 'rrr'):
            return mod.rrr
        else:
            return wrap_fbrrr(mod)

    def get_reload_subs(mod):
        return getattr(mod, 'reload_subs', wrap_fbrrr(mod))

    get_rrr(_model_legacy)(verbose=verbose)
    get_rrr(abstract_models)(verbose=verbose)
    get_rrr(aoi2)(verbose=verbose)
    get_rrr(background)(verbose=verbose)
    get_rrr(classifier)(verbose=verbose)
    get_rrr(classifier2)(verbose=verbose)
    get_rrr(labeler)(verbose=verbose)
    get_rrr(dummy)(verbose=verbose)
    get_rrr(mnist)(verbose=verbose)
    get_rrr(pretrained)(verbose=verbose)
    get_rrr(quality)(verbose=verbose)
    get_rrr(siam)(verbose=verbose)
    get_rrr(viewpoint)(verbose=verbose)
    rrr(verbose=verbose)
    try:
        # hackish way of propogating up the new reloaded submodule attributes
        reassign_submodule_attributes(verbose=verbose)
    except Exception as ex:
        print(ex)


rrrr = reload_subs

IMPORT_TUPLES = [
    ('_model_legacy', None),
    ('abstract_models', None),
    ('aoi2', None),
    ('background', None),
    ('classifier', None),
    ('classifier2', None),
    ('labeler', None),
    ('dummy', None),
    ('mnist', None),
    ('pretrained', None),
    ('quality', None),
    ('siam', None),
    ('viewpoint', None),
]
"""
python -c "import wbia_cnn.models" --dump-wbia_cnn.models-init
python -c "import wbia_cnn.models" --update-wbia_cnn.models-init
"""
__DYNAMIC__ = True
if __DYNAMIC__:
    # TODO: import all utool external prereqs. Then the imports will not import
    # anything that has already in a toplevel namespace
    # COMMENTED OUT FOR FROZEN __INIT__
    # Dynamically import listed util libraries and their members.
    from utool._internal import util_importer

    # FIXME: this might actually work with rrrr, but things arent being
    # reimported because they are already in the modules list
    import_execstr = util_importer.dynamic_import(__name__, IMPORT_TUPLES)
    exec(import_execstr)
    DOELSE = False
else:
    # Do the nonexec import (can force it to happen no matter what if alwyas set
    # to True)
    DOELSE = True

if DOELSE:
    # <AUTOGEN_INIT>
    pass
    # </AUTOGEN_INIT>
"""
Regen Command:
    cd /home/joncrall/code/wbia_cnn/wbia_cnn/models
    makeinit.py --modname=wbia_cnn.models --star
"""
