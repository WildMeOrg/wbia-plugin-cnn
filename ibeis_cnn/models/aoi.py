# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import functools
import six
import numpy as np
import utool as ut
from ibeis_cnn import ingest_data
from ibeis_cnn.__LASAGNE__ import layers
from ibeis_cnn.__LASAGNE__ import nonlinearities
from ibeis_cnn.__THEANO__ import tensor as T  # NOQA
from ibeis_cnn.models import abstract_models

print, rrr, profile = ut.inject2(__name__)


def augment_parallel(X, y):
    return augment_wrapper(
        [X],
        None if y is None else [y]
    )


def augment_wrapper(Xb, yb=None):
    Xb_ = []
    yb_ = []
    for index in range(len(Xb)):
        X = Xb[index]
        y = yb[index]
        for values in y:
            bbox = values[:4]
            Xb_.append(np.append(X, bbox))
            yb_.append(values[4])
    Xb_ = np.array(Xb_, dtype=Xb.dtype)
    yb_ = np.array(yb_, dtype=yb.dtype)
    return Xb_, yb_


@six.add_metaclass(ut.ReloadingMetaclass)
class AoIModel(abstract_models.AbstractVectorVectorModel):
    def __init__(model, autoinit=False, batch_size=128, data_shape=(64, 64, 3),
                 name='aoi', **kwargs):
        super(AoIModel, model).__init__(batch_size=batch_size,
                                               data_shape=data_shape,
                                               name=name, **kwargs)

    def augment(model, Xb, yb=None, parallel=False):
        if not parallel:
            return augment_wrapper(Xb, yb)
        # Run in parallel
        if yb is None:
            yb = [None] * len(Xb)
        arg_iter = list(zip(Xb, yb))
        result_list = ut.util_parallel.generate2(augment_parallel, arg_iter,
                                                 ordered=True, verbose=False)
        result_list = list(result_list)
        X = [ result[0][0] for result in result_list ]
        X = np.array(X)
        if yb is None:
            y = None
        else:
            y = [ result[1] for result in result_list ]
            y = np.hstack(y)
        return X, y

    def get_aoi_def(model, verbose=ut.VERBOSE, **kwargs):
        # _CaffeNet = abstract_models.PretrainedNetwork('caffenet')
        _P = functools.partial

        hidden_initkw = {
            'nonlinearity' : nonlinearities.LeakyRectify(leakiness=(1. / 10.)),
        }

        network_layers_def = (
            [
                _P(layers.InputLayer, shape=model.input_shape),

                _P(layers.DenseLayer, num_units=1024, name='F0', **hidden_initkw),
                _P(layers.FeaturePoolLayer, pool_size=2, name='FP0'),
                _P(layers.DropoutLayer, p=0.5, name='D0'),

                _P(layers.DenseLayer, num_units=512, name='F1', **hidden_initkw),
                _P(layers.FeaturePoolLayer, pool_size=2, name='FP1'),
                _P(layers.DropoutLayer, p=0.5, name='D1'),

                _P(layers.DenseLayer, num_units=256, name='F2', **hidden_initkw),
                _P(layers.FeaturePoolLayer, pool_size=2, name='FP2'),
                _P(layers.DropoutLayer, p=0.5, name='D2'),

                _P(layers.DenseLayer, num_units=128, name='F3', **hidden_initkw),
                _P(layers.FeaturePoolLayer, pool_size=2, name='FP3'),
                _P(layers.DropoutLayer, p=0.5, name='D3'),

                _P(layers.DenseLayer, num_units=model.output_dims, name='F4', nonlinearity=nonlinearities.sigmoid),
            ]
        )
        return network_layers_def

    def init_arch(model, verbose=ut.VERBOSE, **kwargs):
        r"""
        """
        (_, input_size) = model.input_shape
        if verbose:
            print('[model] Initialize aoi model architecture')
            print('[model]   * batch_size     = %r' % (model.batch_size,))
            print('[model]   * input_width    = %r' % (input_size,))
            print('[model]   * output_dims    = %r' % (model.output_dims,))

        network_layers_def = model.get_aoi_def(verbose=verbose, **kwargs)
        # connect and record layers
        from ibeis_cnn import custom_layers
        network_layers = custom_layers.evaluate_layer_list(network_layers_def, verbose=verbose)
        #model.network_layers = network_layers
        output_layer = network_layers[-1]
        model.output_layer = output_layer
        return output_layer


def train_aoi(output_path, data_fpath, labels_fpath):
    r"""
    CommandLine:
        python -m ibeis_cnn.train --test-train_aoi

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis_cnn.train import *  # NOQA
        >>> result = train_aoi()
        >>> print(result)
    """
    era_size = 16
    batch_size = 128
    max_epochs = 256
    hyperparams = ut.argparse_dict(
        {
            'era_size'            : era_size,
            'learning_rate'       : .01,
            'rate_schedule'       : 0.75,
            'momentum'            : .9,
            'weight_decay'        : 0.0001,
            'augment_on'          : True,
            'augment_on_validate' : True,
            'whiten_on'           : False,
            'max_epochs'          : max_epochs,
            'class_weight'        : None,
        }
    )

    ut.colorprint('[netrun] Ensuring Dataset', 'yellow')
    dataset = ingest_data.get_numpy_dataset2('aoi', data_fpath, labels_fpath, output_path)
    X_train, y_train = dataset.subset('train')
    X_valid, y_valid = dataset.subset('valid')
    print('dataset.training_dpath = %r' % (dataset.training_dpath,))

    input_shape = (batch_size, dataset.data_shape[0] + 4, )
    ut.colorprint('[netrun] Architecture Specification', 'yellow')
    model = AoIModel(
        input_shape=input_shape,
        training_dpath=dataset.training_dpath,
        **hyperparams)

    ut.colorprint('[netrun] Initialize archchitecture', 'yellow')
    model.output_dims = 1
    model.input_shape = (None, dataset.data_shape[0] + 4, )
    model.batch_size = batch_size
    model.init_arch()

    ut.colorprint('[netrun] * Initializing new weights', 'lightgray')
    if model.has_saved_state():
        model.load_model_state()

    ut.colorprint('[netrun] Training Requested', 'yellow')
    # parse training arguments
    config = ut.argparse_dict(dict(
        monitor=True,
        monitor_updates=True,
        show_confusion=True,
        era_size=era_size,
        max_epochs=max_epochs,
    ))
    model.monitor_config.update(**config)

    print('\n[netrun] Model Info')
    model.print_layer_info()

    ut.colorprint('[netrun] Begin training', 'yellow')
    model.fit(X_train, y_train, X_valid=X_valid, y_valid=y_valid)

    model_path = model.save_model_state()
    return model_path


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis_cnn.models.aoi
        python -m ibeis_cnn.models.aoi --allexamples
        python -m ibeis_cnn.models.aoi --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
