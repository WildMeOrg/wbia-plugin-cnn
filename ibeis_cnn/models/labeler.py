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
from ibeis_cnn.models import abstract_models, pretrained
import cv2

print, rrr, profile = ut.inject2(__name__)


LABEL_MAPPING_DICT = {
    'left'       : 'right',
    'frontleft'  : 'frontright',
    'front'      : 'front',
    'frontright' : 'frontleft',
    'right'      : 'left',
    'backright'  : 'backleft',
    'back'       : 'back',
    'backleft'   : 'backright',
}


@six.add_metaclass(ut.ReloadingMetaclass)
class LabelerModel(abstract_models.AbstractCategoricalModel):
    def __init__(model, autoinit=False, batch_size=128, data_shape=(64, 64, 3),
                 name='labeler', **kwargs):
        super(LabelerModel, model).__init__(batch_size=batch_size,
                                            data_shape=data_shape,
                                            name=name, **kwargs)

    def augment(model, Xb, yb=None):
        import random
        for index, y in enumerate(yb):
            X = np.copy(Xb[index])
            # Adjust the exposure
            X_Lab = cv2.cvtColor(X, cv2.COLOR_BGR2LAB)
            X_L = X_Lab[:, :, 0].astype(dtype=np.float32)
            # margin = np.min([np.min(X_L), 255.0 - np.max(X_L), 64.0])
            margin = 128.0
            exposure = random.uniform(-margin, margin)
            X_L += exposure
            X_L = np.around(X_L)
            X_L[X_L < 0.0] = 0.0
            X_L[X_L > 255.0] = 255.0
            X_Lab[:, :, 0] = X_L.astype(dtype=X_Lab.dtype)
            X = cv2.cvtColor(X_Lab, cv2.COLOR_LAB2BGR)
            # Rotate and Scale
            h, w, c = X.shape
            degree = random.randint(-30, 30)
            scale = random.uniform(0.80, 1.25)
            padding = np.sqrt((w) ** 2 / 4 - 2 * (w) ** 2 / 16)
            padding /= scale
            padding = int(np.ceil(padding))
            for channel in range(c):
                X_ = X[:, :, channel]
                X_ = np.pad(X_, padding, 'reflect', reflect_type='even')
                h_, w_ = X_.shape
                # Calulate Affine transform
                center = (w_ // 2, h_ // 2)
                A = cv2.getRotationMatrix2D(center, degree, scale)
                X_ = cv2.warpAffine(X_, A, (w_, h_), flags=cv2.INTER_LANCZOS4, borderValue=0)
                X_ = X_[padding: -1 * padding, padding: -1 * padding]
                X[:, :, channel] = X_
            # Horizontal flip
            if random.uniform(0.0, 1.0) <= 0.5:
                X = cv2.flip(X, 1)
                if ':' in y:
                    species, viewpoint = y.split(':')
                    viewpoint = LABEL_MAPPING_DICT[viewpoint]
                    y = '%s:%s' % (species, viewpoint)
            # Blur
            if random.uniform(0.0, 1.0) <= 0.1:
                if random.uniform(0.0, 1.0) <= 0.5:
                    X = cv2.blur(X, (3, 3))
                else:
                    X = cv2.blur(X, (5, 5))
            # Reshape
            X = X.reshape(Xb[index].shape)
            # Show image
            # canvas = np.hstack((Xb[index], X))
            # cv2.imshow('', canvas)
            # cv2.waitKey(0)
            # Save
            Xb[index] = X
            yb[index] = y
        return Xb, yb

    def get_labeler_def(model, verbose=ut.VERBOSE, **kwargs):
        # _CaffeNet = abstract_models.PretrainedNetwork('caffenet')
        _P = functools.partial

        _CaffeNet = pretrained.PretrainedNetwork('caffenet_conv')

        hidden_initkw = {
            'nonlinearity' : nonlinearities.LeakyRectify(leakiness=(1. / 10.))
        }

        from ibeis_cnn import custom_layers

        Conv2DLayer = custom_layers.Conv2DLayer
        MaxPool2DLayer = custom_layers.MaxPool2DLayer
        #DenseLayer = layers.DenseLayer

        network_layers_def = (
            [
                _P(layers.InputLayer, shape=model.input_shape),

                _P(Conv2DLayer, num_filters=32, filter_size=(11, 11), name='C0', W=_CaffeNet.get_pretrained_layer(0), **hidden_initkw),  # NOQA
                _P(Conv2DLayer, num_filters=16, filter_size=(5, 5), name='C1', W=_CaffeNet.get_pretrained_layer(2), **hidden_initkw),  # NOQA
                _P(layers.DropoutLayer, p=0.1, name='D0'),
                _P(MaxPool2DLayer, pool_size=(2, 2), stride=(2, 2), name='P0'),

                _P(Conv2DLayer, num_filters=64, filter_size=(3, 3), name='C2', W=_CaffeNet.get_pretrained_layer(4), **hidden_initkw),  # NOQA
                _P(Conv2DLayer, num_filters=32, filter_size=(3, 3), name='C3', W=_CaffeNet.get_pretrained_layer(6), **hidden_initkw),  # NOQA
                _P(layers.DropoutLayer, p=0.1, name='D1'),
                _P(MaxPool2DLayer, pool_size=(2, 2), stride=(2, 2), name='P1'),

                _P(Conv2DLayer, num_filters=128, filter_size=(3, 3), name='C4', **hidden_initkw),
                _P(Conv2DLayer, num_filters=64, filter_size=(3, 3), name='C5', **hidden_initkw),
                _P(layers.DropoutLayer, p=0.1, name='D2'),
                _P(MaxPool2DLayer, pool_size=(2, 2), stride=(2, 2), name='P2'),

                _P(Conv2DLayer, num_filters=256, filter_size=(3, 3), name='C6', **hidden_initkw),
                _P(Conv2DLayer, num_filters=256, filter_size=(3, 3), name='C7', **hidden_initkw),
                _P(Conv2DLayer, num_filters=128, filter_size=(3, 3), name='C8', **hidden_initkw),
                _P(layers.DropoutLayer, p=0.1, name='D3'),

                _P(layers.DenseLayer, num_units=512, name='F0',  **hidden_initkw),
                _P(layers.FeaturePoolLayer, pool_size=2, name='FP0'),
                _P(layers.DropoutLayer, p=0.5, name='D4'),

                _P(layers.DenseLayer, num_units=512, name='F1', **hidden_initkw),
                _P(layers.FeaturePoolLayer, pool_size=2, name='FP1'),
                _P(layers.DropoutLayer, p=0.5, name='D5'),

                _P(layers.DenseLayer, num_units=model.output_dims, name='F2', nonlinearity=nonlinearities.softmax),
            ]
        )
        return network_layers_def

    def init_arch(model, verbose=ut.VERBOSE, **kwargs):
        r"""
        """
        (_, input_channels, input_width, input_height) = model.input_shape
        if verbose:
            print('[model] Initialize labeler model architecture')
            print('[model]   * batch_size     = %r' % (model.batch_size,))
            print('[model]   * input_width    = %r' % (input_width,))
            print('[model]   * input_height   = %r' % (input_height,))
            print('[model]   * input_channels = %r' % (input_channels,))
            print('[model]   * output_dims    = %r' % (model.output_dims,))

        network_layers_def = model.get_labeler_def(verbose=verbose, **kwargs)
        # connect and record layers
        from ibeis_cnn import custom_layers
        network_layers = custom_layers.evaluate_layer_list(network_layers_def, verbose=verbose)
        #model.network_layers = network_layers
        output_layer = network_layers[-1]
        model.output_layer = output_layer
        return output_layer


def train_labeler(output_path, data_fpath, labels_fpath):
    r"""
    CommandLine:
        python -m ibeis_cnn.train --test-train_labeler

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis_cnn.train import *  # NOQA
        >>> result = train_labeler()
        >>> print(result)
    """
    hyperparams = ut.argparse_dict(
        {
            'era_size'      : 8,
            'era_clean'     : False,
            'batch_size'    : 128,
            'learning_rate' : .01,
            'momentum'      : .9,
            'weight_decay'  : 0.0005,
            'augment_on'    : True,
            'whiten_on'     : True,
        }
    )

    ut.colorprint('[netrun] Ensuring Dataset', 'yellow')
    dataset = ingest_data.get_numpy_dataset2('labeler', data_fpath, labels_fpath, output_path)
    output_dims = len(set(dataset.labels))
    print('dataset.training_dpath = %r' % (dataset.training_dpath,))

    ut.colorprint('[netrun] Architecture Specification', 'yellow')
    model = LabelerModel(
        data_shape=dataset.data_shape,
        training_dpath=dataset.training_dpath,
        output_dims=output_dims,
        **hyperparams)

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
    config = ut.argparse_dict(dict(
        era_size=15,
        max_epochs=120,
        show_confusion=False,
    ))
    model.monitor_config.update(**config)
    ut.embed()
    X_train, y_train = dataset.subset('train')
    X_valid, y_valid = dataset.subset('valid')

    ut.colorprint('[netrun] Init encoder and convert labels', 'yellow')
    if hasattr(model, 'init_encoder'):
        model.init_encoder(y_train)

    if getattr(model, 'encoder', None) is not None:
        class_list = list(model.encoder.classes_)
        y_train = np.array([class_list.index(_) for _ in y_train ])
        y_valid = np.array([class_list.index(_) for _ in y_valid ])

    ut.colorprint('[netrun] Begin training', 'yellow')
    model.fit(X_train, y_train, X_valid=X_valid, y_valid=y_valid)

    model_path = model.save_model_state()
    return model_path


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis_cnn.models.labeler
        python -m ibeis_cnn.models.labeler --allexamples
        python -m ibeis_cnn.models.labeler --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
