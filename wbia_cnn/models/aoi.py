# -*- coding: utf-8 -*-
import logging
import functools
import six
import numpy as np
import utool as ut
from wbia_cnn import ingest_data
from lasagne import layers, nonlinearities
from theano import tensor as T  # NOQA
from wbia_cnn.models import abstract_models

print, rrr, profile = ut.inject2(__name__)
logger = logging.getLogger()


def augment_parallel(X, y, w):
    return augment_wrapper(
        [X],
        None if y is None else [y],
        None if w is None else [w],
    )


def augment_wrapper(Xb, yb=None, wb=None):
    Xb_ = []
    yb_ = []
    wb_ = []
    for index in range(len(Xb)):
        X = Xb[index]
        y = yb[index]
        for values in y:
            bbox = values[:4]
            Xb_.append(np.append(X, bbox))
            yb_.append(values[4])
            wb_.append(1.0)
    Xb_ = np.array(Xb_, dtype=Xb.dtype)
    yb_ = np.array(yb_, dtype=yb.dtype)
    wb_ = np.array(wb_, dtype=np.float32)
    return Xb_, yb_, wb_


@six.add_metaclass(ut.ReloadingMetaclass)
class AoIModel(abstract_models.AbstractVectorVectorModel):
    def __init__(
        model,
        autoinit=False,
        batch_size=128,
        data_shape=(64, 64, 3),
        name='aoi',
        **kwargs
    ):
        super(AoIModel, model).__init__(
            batch_size=batch_size, data_shape=data_shape, name=name, **kwargs
        )

    def augment(model, Xb, yb=None, wb=None, parallel=False):
        if not parallel:
            return augment_wrapper(Xb, yb, wb)
        # Run in parallel
        if yb is None:
            yb = [None] * len(Xb)
        if wb is None:
            wb = [None] * len(Xb)
        arg_iter = list(zip(Xb, yb, wb))
        result_list = ut.util_parallel.generate2(
            augment_parallel, arg_iter, ordered=True, verbose=False
        )
        result_list = list(result_list)
        X = [result[0][0] for result in result_list]
        X = np.array(X)
        if yb is None:
            y = None
        else:
            y = [result[1] for result in result_list]
            y = np.hstack(y)
        if wb is None:
            w = None
        else:
            w = [result[2] for result in result_list]
            w = np.hstack(w)
        return X, y, w

    def get_aoi_def(model, verbose=ut.VERBOSE, **kwargs):
        _P = functools.partial

        hidden_initkw = {
            'nonlinearity': nonlinearities.LeakyRectify(leakiness=(1.0 / 10.0)),
        }

        (_, input_size) = model.input_shape
        network_layers_def = [
            _P(
                layers.InputLayer,
                shape=(
                    None,
                    input_size,
                ),
            ),
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
            _P(
                layers.DenseLayer,
                num_units=model.output_dims,
                name='F4',
                nonlinearity=nonlinearities.sigmoid,
            ),
        ]
        return network_layers_def

    def init_arch(model, verbose=True, **kwargs):
        r""""""
        (_, input_size) = model.input_shape
        if verbose:
            logger.info('[model] Initialize aoi model architecture')
            logger.info('[model]   * batch_size     = %r' % (model.batch_size,))
            logger.info('[model]   * input_width    = %r' % (input_size,))
            logger.info('[model]   * output_dims    = %r' % (model.output_dims,))

        network_layers_def = model.get_aoi_def(verbose=verbose, **kwargs)
        # connect and record layers
        from wbia_cnn import custom_layers

        network_layers = custom_layers.evaluate_layer_list(
            network_layers_def, verbose=verbose
        )
        # model.network_layers = network_layers
        output_layer = network_layers[-1]
        model.output_layer = output_layer
        return output_layer


def train_aoi(output_path, data_fpath, labels_fpath):
    r"""
    CommandLine:
        python -m wbia_cnn.train --test-train_aoi

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia_cnn.models.aoi import *  # NOQA
        >>> result = train_aoi()
        >>> print(result)
    """
    era_size = 256
    batch_size = 16
    max_epochs = era_size * 16
    hyperparams = ut.argparse_dict(
        {
            'era_size': era_size,
            'learning_rate': 0.01,
            'rate_schedule': 0.75,
            'momentum': 0.9,
            'weight_decay': 0.0001,
            'augment_on': True,
            'augment_on_validate': True,
            'whiten_on': False,
            'max_epochs': max_epochs,
            'stopping_patience': max_epochs,
            'class_weight': None,
        }
    )

    ut.colorprint('[netrun] Ensuring Dataset', 'yellow')
    dataset = ingest_data.get_numpy_dataset2('aoi', data_fpath, labels_fpath, output_path)
    X_train, y_train = dataset.subset('train')
    X_valid, y_valid = dataset.subset('valid')
    logger.info('dataset.training_dpath = %r' % (dataset.training_dpath,))

    input_shape = (
        batch_size,
        dataset.data_shape[0] + 4,
    )
    ut.colorprint('[netrun] Architecture Specification', 'yellow')
    model = AoIModel(
        input_shape=input_shape, training_dpath=dataset.training_dpath, **hyperparams
    )

    ut.colorprint('[netrun] Initialize architecture', 'yellow')
    model.output_dims = 1
    model.input_shape = (
        None,
        dataset.data_shape[0] + 4,
    )
    model.batch_size = batch_size
    model.init_arch()

    ut.colorprint('[netrun] * Initializing new weights', 'lightgray')
    if model.has_saved_state():
        model.load_model_state()

    ut.colorprint('[netrun] Training Requested', 'yellow')
    # parse training arguments
    config = ut.argparse_dict(
        dict(
            era_size=era_size,
            max_epochs=max_epochs,
            show_confusion=False,
        )
    )
    model.monitor_config.update(**config)

    logger.info('\n[netrun] Model Info')
    model.print_layer_info()

    ut.colorprint('[netrun] Begin training', 'yellow')
    model.fit(X_train, y_train, X_valid=X_valid, y_valid=y_valid)

    model_path = model.save_model_state()
    return model_path


if __name__ == '__main__':
    """
    CommandLine:
        python -m wbia_cnn.models.aoi
        python -m wbia_cnn.models.aoi --allexamples
        python -m wbia_cnn.models.aoi --allexamples --noface --nosrc
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
