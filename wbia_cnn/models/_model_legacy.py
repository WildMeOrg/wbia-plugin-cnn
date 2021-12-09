# -*- coding: utf-8 -*-
import logging
import utool as ut
from six.moves import cPickle as pickle  # NOQA

print, rrr, profile = ut.inject2(__name__)
logger = logging.getLogger()


@ut.reloadable_class
class _ModelLegacy(object):
    """
    contains old functions for backwards compatibility
    that may be eventually be depricated
    """

    def _fix_center_mean_std(model):
        # Hack to preconvert mean / std to 0-1 for old models
        if model.data_params is not None:
            if model.data_params.get('center_std', None) == 255:
                model.data_params['center_std'] = 1.0
                model.data_params['center_mean'] /= 255.0

    def load_old_weights_kw(model, old_weights_fpath):
        logger.info('[model] loading old model state from: %s' % (old_weights_fpath,))
        oldkw = ut.load_cPkl(old_weights_fpath)
        # Model architecture and weight params
        data_shape = oldkw['model_shape'][1:]
        input_shape = (None, data_shape[2], data_shape[0], data_shape[1])
        output_dims = oldkw['output_dims']

        if model.output_dims is None:
            model.output_dims = output_dims

        # Perform checks
        assert input_shape[1:] == model.input_shape[1:], 'architecture disagreement'
        assert output_dims == model.output_dims, 'architecture disagreement'

        model.data_params = {
            'center_mean': oldkw['center_mean'],
            'center_std': oldkw['center_std'],
        }
        model._fix_center_mean_std()
        # Set class attributes
        print(oldkw.keys())
        model.best_results = {
            'epoch': oldkw['best_epoch'],
            'test_accuracy': oldkw['best_test_accuracy'],
            'learn_loss': oldkw['best_train_loss'],
            'valid_accuracy': oldkw['best_valid_accuracy'],
            'valid_loss': oldkw['best_valid_loss'],
            'weights': oldkw['best_weights'],
        }

        # Need to build architecture first
        model.init_arch()

        model.encoder = oldkw.get('encoder', None)

        # Set architecture weights
        weights_list = model.best_results['weights']
        model.set_all_param_values(weights_list)

    def load_old_weights_kw2(model, old_weights_fpath):
        logger.info('[model] loading old model state from: %s' % (old_weights_fpath,))

        oldkw = ut.load_cPkl(old_weights_fpath, n=None)
        # output_dims = model.best_results['weights'][-1][0]

        # Model architecture and weight params
        if model.output_dims is None:
            # model.output_dims = output_dims
            # ut.depth_profile(oldkw['best_weights'])
            model.output_dims = oldkw['best_weights'][-1].shape[0]

        # Set class attributes
        model.data_params = {
            'center_mean': oldkw['data_whiten_mean'],
            'center_std': oldkw['data_whiten_std'],
        }
        model._fix_center_mean_std()
        model.best_results = {
            'epoch': oldkw['best_epoch'],
            'test_accuracy': oldkw['best_valid_accuracy'],
            'learn_loss': oldkw['best_train_loss'],
            'valid_accuracy': oldkw['best_valid_accuracy'],
            'valid_loss': oldkw['best_valid_loss'],
            'weights': oldkw['best_fit_weights'],
        }

        # Need to build architecture first
        model.init_arch()
        model.encoder = oldkw.get('data_label_encoder', None)
        model.batch_size = oldkw['train_batch_size']

        # Set architecture weights
        model.set_all_param_values(model.best_results['weights'])
