# -*- coding: utf-8 -*-
import logging
import functools
import six
import numpy as np
import utool as ut
from wbia_cnn import ingest_data
from lasagne import init, layers, nonlinearities
from theano import tensor as T  # NOQA
from wbia_cnn.models import abstract_models
from os.path import exists  # NOQA

print, rrr, profile = ut.inject2(__name__)
logger = logging.getLogger()


def augment_parallel(X, y, w):
    return augment_wrapper(
        [X],
        None if y is None else [y],
        None if w is None else [w],
    )


def augment_wrapper(Xb, yb=None, wb=None):
    import random
    import cv2

    Xb_ = []
    yb_ = []
    wb_ = []
    for index in range(len(Xb)):
        X_base = np.copy(Xb[index])
        y = None if yb is None else yb[index]

        for xtl, ytl, xbr, ybr, class_ in y:
            X = np.copy(X_base)
            mask = X[:, :, 3]
            X = X[:, :, :3]
            # Adjust the exposure
            X_Lab = cv2.cvtColor(X, cv2.COLOR_BGR2LAB)
            X_L = X_Lab[:, :, 0].astype(dtype=np.float32)
            # margin = np.min([np.min(X_L), 255.0 - np.max(X_L), 64.0])
            margin = 64.0
            exposure = random.uniform(-margin, margin)
            X_L += exposure
            X_L = np.around(X_L)
            X_L[X_L < 0.0] = 0.0
            X_L[X_L > 255.0] = 255.0
            X_Lab[:, :, 0] = X_L.astype(dtype=X_Lab.dtype)
            X = cv2.cvtColor(X_Lab, cv2.COLOR_LAB2BGR)
            # Rotate, Scale, Skew
            h, w, c = X.shape
            degree = random.randint(-15, 15)
            scale = random.uniform(0.90, 1.10)
            skew_x = random.uniform(0.90, 1.10)
            skew_y = random.uniform(0.90, 1.10)
            skew_x_offset = abs(1.0 - skew_x)
            skew_y_offset = abs(1.0 - skew_y)
            skew_offset = np.sqrt(
                skew_x_offset ** skew_x_offset + skew_y_offset ** skew_y_offset
            )
            skew_scale = 1.0 + skew_offset
            padding = np.sqrt((w) ** 2 / 4 - 2 * (w) ** 2 / 16)
            padding /= scale
            padding *= skew_scale
            padding = int(np.ceil(padding))
            # Make mask
            xtl = int(np.around(xtl * w))
            ytl = int(np.around(ytl * h))
            xbr = int(np.around(xbr * w))
            ybr = int(np.around(ybr * h))
            mask[ytl:ybr, xtl:xbr] = 255
            X = np.dstack((X, mask))
            for channel in range(c + 1):
                X_ = X[:, :, channel]
                if channel >= c:
                    X_ = np.pad(X_, padding, 'constant', constant_values=0.0)
                else:
                    X_ = np.pad(X_, padding, 'reflect', reflect_type='even')
                h_, w_ = X_.shape
                # Calculate Affine transform
                center = (w_ // 2, h_ // 2)
                A = cv2.getRotationMatrix2D(center, degree, scale)
                # Add skew
                A[0][0] *= skew_x
                A[1][0] *= skew_x
                A[0][1] *= skew_y
                A[1][1] *= skew_y
                # Apply Affine
                X_ = cv2.warpAffine(
                    X_, A, (w_, h_), flags=cv2.INTER_LANCZOS4, borderValue=0
                )
                X_ = X_[padding : -1 * padding, padding : -1 * padding]
                X[:, :, channel] = X_
            # Horizontal flip
            if random.uniform(0.0, 1.0) <= 0.5:
                X = cv2.flip(X, 1)
            # Blur
            if random.uniform(0.0, 1.0) <= 0.01:
                X[:, :, :3] = cv2.blur(X[:, :, :3], (3, 3))
            # Reshape
            X = X.reshape(Xb[index].shape)
            X = X.astype(Xb[index].dtype)
            # Show image
            canvas_filepath = '/home/jason.parham/Desktop/temp-%s-%d.png' % (
                class_,
                random.randint(0, 100),
            )
            if random.uniform(0.0, 1.0) < 0.01:  # False and not exists(canvas_filepath)
                temp_list = [
                    Xb[index][:, :, :3],
                    cv2.merge((mask, mask, mask)),
                    Xb[index][:, :, :3] * (mask / 255.0).reshape(192, 192, 1),
                    X[:, :, :3],
                    cv2.merge((X[:, :, 3], X[:, :, 3], X[:, :, 3])),
                    X[:, :, :3] * (X[:, :, 3] / 255.0).reshape(192, 192, 1),
                ]
                canvas = np.hstack(temp_list)
                cv2.imwrite(canvas_filepath, canvas)
            # Save
            Xb_.append(X)
            yb_.append(int(class_))
            wb_.append(1.0)

    Xb_ = np.array(Xb_, dtype=np.uint8)
    yb_ = np.array(yb_, dtype=np.uint8)
    wb_ = np.array(wb_, dtype=np.float32)

    return Xb_, yb_, wb_


@six.add_metaclass(ut.ReloadingMetaclass)
class AoI2Model(abstract_models.AbstractCategoricalModel):
    def __init__(
        model,
        autoinit=False,
        batch_size=128,
        data_shape=(64, 64, 3),
        name='aoi2',
        **kwargs
    ):
        super(AoI2Model, model).__init__(
            batch_size=batch_size, data_shape=data_shape, name=name, **kwargs
        )

    def augment(model, Xb, yb=None, wb=None, parallel=True):
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
        X = [result[0] for result in result_list]
        X = np.vstack(X)
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

    def get_aoi2_def(model, verbose=ut.VERBOSE, **kwargs):
        _P = functools.partial

        hidden_initkw = {
            'nonlinearity': nonlinearities.LeakyRectify(leakiness=(1.0 / 10.0)),
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
                W=init.Orthogonal('relu'),
                **hidden_initkw
            ),  # NOQA
            _P(MaxPool2DLayer, pool_size=(2, 2), stride=(2, 2), name='P0'),
            _P(
                Conv2DLayer,
                num_filters=32,
                filter_size=(5, 5),
                name='C1',
                W=init.Orthogonal('relu'),
                **hidden_initkw
            ),  # NOQA
            _P(MaxPool2DLayer, pool_size=(2, 2), stride=(2, 2), name='P1'),
            _P(
                Conv2DLayer,
                num_filters=64,
                filter_size=(3, 3),
                name='C2',
                W=init.Orthogonal('relu'),
                **hidden_initkw
            ),  # NOQA
            _P(MaxPool2DLayer, pool_size=(2, 2), stride=(2, 2), name='P2'),
            _P(
                Conv2DLayer,
                num_filters=128,
                filter_size=(3, 3),
                name='C3',
                W=init.Orthogonal('relu'),
                **hidden_initkw
            ),
            _P(MaxPool2DLayer, pool_size=(2, 2), stride=(2, 2), name='P3'),
            _P(
                Conv2DLayer,
                num_filters=128,
                filter_size=(3, 3),
                name='C4',
                W=init.Orthogonal('relu'),
                **hidden_initkw
            ),
            _P(
                Conv2DLayer,
                num_filters=128,
                filter_size=(3, 3),
                name='C5',
                W=init.Orthogonal('relu'),
                **hidden_initkw
            ),
            _P(layers.DenseLayer, num_units=256, name='F0', **hidden_initkw),
            _P(layers.FeaturePoolLayer, pool_size=2, name='FP0'),
            _P(layers.DropoutLayer, p=0.5, name='D1'),
            _P(layers.DenseLayer, num_units=256, name='F1', **hidden_initkw),
            _P(
                layers.DenseLayer,
                num_units=model.output_dims,
                name='F2',
                nonlinearity=nonlinearities.softmax,
            ),
        ]
        return network_layers_def

    def init_arch(model, verbose=ut.VERBOSE, **kwargs):
        r""""""
        (_, input_channels, input_width, input_height) = model.input_shape
        if verbose or True:
            logger.info('[model] Initialize aoi2 model architecture')
            logger.info('[model]   * batch_size     = %r' % (model.batch_size,))
            logger.info('[model]   * input_width    = %r' % (input_width,))
            logger.info('[model]   * input_height   = %r' % (input_height,))
            logger.info('[model]   * input_channels = %r' % (input_channels,))
            logger.info('[model]   * output_dims    = %r' % (model.output_dims,))

        network_layers_def = model.get_aoi2_def(verbose=verbose, **kwargs)
        # connect and record layers
        from wbia_cnn import custom_layers

        network_layers = custom_layers.evaluate_layer_list(
            network_layers_def, verbose=verbose
        )
        # model.network_layers = network_layers
        output_layer = network_layers[-1]
        model.output_layer = output_layer
        return output_layer


def train_aoi2(output_path, data_fpath, labels_fpath, purge=True):
    r"""
    CommandLine:
        python -m wbia_cnn.train --test-train_aoi2

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia_cnn.models.aoi2 import *  # NOQA
        >>> result = train_aoi2()
        >>> print(result)
    """
    era_size = 16
    max_epochs = 256
    hyperparams = ut.argparse_dict(
        {
            'era_size': era_size,
            'batch_size': 32,
            'learning_rate': 0.01,
            'rate_schedule': 0.5,
            'momentum': 0.9,
            'weight_decay': 0.0001,
            'augment_on': True,
            'augment_on_validate': True,
            'augment_weights': True,
            'label_encode_on': False,
            'whiten_on': True,
            'class_weight': None,
            'max_epochs': max_epochs,
        }
    )

    ut.colorprint('[netrun] Ensuring Dataset', 'yellow')
    dataset = ingest_data.get_numpy_dataset2(
        'aoi2', data_fpath, labels_fpath, output_path, cache=False
    )
    X_train, y_train = dataset.subset('train')
    X_valid, y_valid = dataset.subset('valid')
    logger.info('dataset.training_dpath = %r' % (dataset.training_dpath,))

    if purge:
        model = AoI2Model(
            data_shape=dataset.data_shape,
            training_dpath=dataset.training_dpath,
            **hyperparams
        )
        model.output_dims = 2
        model.init_arch()
        ut.delete(model.arch_dpath)

    ut.colorprint('[netrun] Architecture Specification', 'yellow')
    model = AoI2Model(
        data_shape=dataset.data_shape,
        training_dpath=dataset.training_dpath,
        **hyperparams
    )

    ut.colorprint('[netrun] Initialize archchitecture', 'yellow')
    model.output_dims = 2
    model.init_arch()

    ut.colorprint('[netrun] * Initializing new weights', 'lightgray')
    if model.has_saved_state():
        model.load_model_state()
    # else:
    #     model.reinit_weights()

    # ut.colorprint('[netrun] Need to initialize training state', 'yellow')
    # X_train, y_train = dataset.subset('train')
    # model.ensure_data_params(X_train, y_train)

    ut.colorprint('[netrun] Training Requested', 'yellow')
    # parse training arguments
    config = ut.argparse_dict(
        dict(
            monitor=False,
            monitor_updates=False,
            show_confusion=True,
            era_size=era_size,
            max_epochs=max_epochs,
        )
    )
    model.monitor_config.update(**config)

    if getattr(model, 'encoder', None) is not None:
        class_list = list(model.encoder.classes_)
        y_train = np.array([class_list.index(_) for _ in y_train])
        y_valid = np.array([class_list.index(_) for _ in y_valid])

    logger.info('\n[netrun] Model Info')
    model.print_layer_info()

    ut.colorprint('[netrun] Begin training', 'yellow')
    model.fit(X_train, y_train, X_valid=X_valid, y_valid=y_valid)

    model_path = model.save_model_state()
    return model_path


if __name__ == '__main__':
    """
    CommandLine:
        python -m wbia_cnn.models.aoi2
        python -m wbia_cnn.models.aoi2 --allexamples
        python -m wbia_cnn.models.aoi2 --allexamples --noface --nosrc
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
