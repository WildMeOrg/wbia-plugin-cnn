# -*- coding: utf-8 -*-
import logging
import functools
import six
import numpy as np
import utool as ut
from wbia_cnn import ingest_data
import lasagne
from lasagne import layers, nonlinearities
from theano import tensor as T  # NOQA
from wbia_cnn.models import abstract_models

print, rrr, profile = ut.inject2(__name__)
logger = logging.getLogger()


class NonlinearityLayerSpatial(lasagne.layers.NonlinearityLayer):
    def __init__(self, incoming, nonlinearity=nonlinearities.rectify, **kwargs):
        """The spatial version of a nonlinearity as applied accross all spatial
        dimensions of a network's output.
        """
        super(NonlinearityLayerSpatial, self).__init__(incoming, **kwargs)
        self.nonlinearity = (
            nonlinearities.identity if nonlinearity is None else nonlinearity
        )
        in_batch, in_channels, in_width, in_height = self.input_shape
        self.reshape_required = in_width == 1 and in_height == 1

    def get_output_for(self, input, **kwargs):
        old_shape = T.shape(input)
        if self.reshape_required:
            input = T.reshape(input, (-1, old_shape[1]))
            return self.nonlinearity(input)
        elif input.ndim == 4:
            input = input.dimshuffle((0, 3, 2, 1))
            temp = T.shape(input)
            input = T.reshape(input, (-1, old_shape[1]))
            activation = self.nonlinearity(input)
            activation = T.reshape(activation, temp)
            activation = activation.dimshuffle((0, 3, 2, 1))  # Transpose
            return activation
        else:
            _super = super(NonlinearityLayerSpatial, self)
            return _super.get_output_for(input, **kwargs)

    def get_output_shape_for(self, input_shape):
        if self.reshape_required:
            return input_shape[:2]
        else:
            _super = super(NonlinearityLayerSpatial, self)
            return _super.get_output_shape_for(input_shape)


@six.add_metaclass(ut.ReloadingMetaclass)
class BackgroundModel(abstract_models.AbstractCategoricalModel):
    def __init__(
        model,
        autoinit=False,
        batch_size=128,
        data_shape=(48, 48, 3),
        num_output=2,
        **kwargs
    ):
        model.num_output = num_output
        super(BackgroundModel, model).__init__(
            batch_size=batch_size, data_shape=data_shape, name='background', **kwargs
        )

    def learning_rate_update(model, x):
        return x / 2.0

    def learning_rate_shock(model, x):
        return x * 2.0

    def augment(model, Xb, yb=None):
        import random
        import cv2

        for index, X in enumerate(Xb):
            if random.uniform(0.0, 1.0) <= 0.5:
                Xb[index] = cv2.flip(X, 1)
        return Xb, yb

    def get_background_def(model, verbose=ut.VERBOSE, **kwargs):
        _P = functools.partial

        hidden_initkw = {
            'nonlinearity': nonlinearities.LeakyRectify(leakiness=(1.0 / 10.0))
        }

        from wbia_cnn import custom_layers

        Conv2DLayer = custom_layers.Conv2DLayer
        MaxPool2DLayer = custom_layers.MaxPool2DLayer
        # DenseLayer = layers.DenseLayer

        network_layers_def = [
            _P(layers.InputLayer, shape=model.input_shape),
            _P(
                Conv2DLayer,
                num_filters=16,
                filter_size=(11, 11),
                name='C0',
                **hidden_initkw
            ),
            _P(layers.DropoutLayer, p=0.1, name='D0'),
            _P(MaxPool2DLayer, pool_size=(2, 2), stride=(2, 2), name='P0'),
            _P(
                Conv2DLayer,
                num_filters=32,
                filter_size=(5, 5),
                name='C1',
                **hidden_initkw
            ),
            _P(layers.DropoutLayer, p=0.2, name='D1'),
            _P(MaxPool2DLayer, pool_size=(2, 2), stride=(2, 2), name='P1'),
            _P(
                Conv2DLayer,
                num_filters=64,
                filter_size=(3, 3),
                name='C2',
                **hidden_initkw
            ),
            _P(layers.DropoutLayer, p=0.3, name='D2'),
            _P(MaxPool2DLayer, pool_size=(2, 2), stride=(2, 2), name='P2'),
            _P(
                Conv2DLayer,
                num_filters=128,
                filter_size=(3, 3),
                name='C4',
                **hidden_initkw
            ),
            _P(layers.DropoutLayer, p=0.4, name='D4'),
            _P(layers.NINLayer, num_units=model.num_output, name='F3', nonlinearity=None),
            _P(NonlinearityLayerSpatial, name='S0', nonlinearity=nonlinearities.softmax),
        ]
        return network_layers_def

    def init_arch(model, verbose=ut.VERBOSE, **kwargs):
        r""""""
        (_, input_channels, input_width, input_height) = model.input_shape
        if verbose:
            logger.info('[model] Initialize background model architecture')
            logger.info('[model]   * batch_size     = %r' % (model.batch_size,))
            logger.info('[model]   * input_width    = %r' % (input_width,))
            logger.info('[model]   * input_height   = %r' % (input_height,))
            logger.info('[model]   * input_channels = %r' % (input_channels,))
            logger.info('[model]   * output_dims    = %r' % (model.output_dims,))

        network_layers_def = model.get_background_def(verbose=verbose, **kwargs)
        # connect and record layers
        from wbia_cnn import custom_layers

        network_layers = custom_layers.evaluate_layer_list(
            network_layers_def, verbose=verbose
        )
        # model.network_layers = network_layers
        output_layer = network_layers[-1]
        model.output_layer = output_layer
        return output_layer


def train_background(output_path, data_fpath, labels_fpath):
    r"""
    CommandLine:
        python -m wbia_cnn.train --test-train_background

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia_cnn.models.background import *  # NOQA
        >>> result = train_background()
        >>> print(result)
    """
    era_size = 8
    max_epochs = 128
    hyperparams = ut.argparse_dict(
        {
            'era_size': era_size,
            'era_clean': True,
            'batch_size': 128,
            'learning_rate': 0.01,
            'momentum': 0.9,
            'weight_decay': 0.0005,
            'augment_on': True,
            'whiten_on': True,
            'max_epochs': max_epochs,
        }
    )

    ut.colorprint('[netrun] Ensuring Dataset', 'yellow')
    dataset = ingest_data.get_numpy_dataset2(
        'background', data_fpath, labels_fpath, output_path
    )
    logger.info('dataset.training_dpath = %r' % (dataset.training_dpath,))

    ut.colorprint('[netrun] Architecture Specification', 'yellow')
    model = BackgroundModel(
        data_shape=dataset.data_shape,
        training_dpath=dataset.training_dpath,
        **hyperparams
    )

    ut.colorprint('[netrun] Initialize archchitecture', 'yellow')
    model.init_arch()

    ut.colorprint('[netrun] * Initializing new weights', 'lightgray')
    if model.has_saved_state():
        model.load_model_state()
    else:
        model.reinit_weights()

    # ut.colorprint('[netrun] Need to initialize training state', 'yellow')
    # X_train, y_train = dataset.subset('train')
    # model.ensure_data_params(X_train, y_train)

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
    X_train, y_train = dataset.subset('train')
    X_valid, y_valid = dataset.subset('valid')

    ut.colorprint('[netrun] Init encoder and convert labels', 'yellow')
    if hasattr(model, 'init_encoder'):
        model.init_encoder(y_train)

    if getattr(model, 'encoder', None) is not None:
        class_list = list(model.encoder.classes_)
        y_train = np.array([class_list.index(_) for _ in y_train])
        y_valid = np.array([class_list.index(_) for _ in y_valid])

    ut.colorprint('[netrun] Begin training', 'yellow')
    model.fit(X_train, y_train, X_valid=X_valid, y_valid=y_valid)

    model_path = model.save_model_state()
    return model_path


if __name__ == '__main__':
    """
    CommandLine:
        python -m wbia_cnn.models.background
        python -m wbia_cnn.models.background --allexamples
        python -m wbia_cnn.models.background --allexamples --noface --nosrc
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
