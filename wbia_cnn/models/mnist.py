# -*- coding: utf-8 -*-
import logging
from wbia_cnn.models import abstract_models
import utool as ut

print, rrr, profile = ut.inject2(__name__)
logger = logging.getLogger()


class MNISTModel(abstract_models.AbstractCategoricalModel):
    """
    Toy model for testing and playing with mnist

    CommandLine:
        python -m wbia_cnn.models.mnist MNISTModel:0
        python -m wbia_cnn.models.mnist MNISTModel:1

        python -m wbia_cnn _ModelFitting.fit:0 --vd --monitor
        python -m wbia_cnn _ModelFitting.fit:1 --vd

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia_cnn.models.mnist import *  # NOQA
        >>> from wbia_cnn import ingest_data
        >>> dataset = ingest_data.grab_mnist_category_dataset_float()
        >>> model = MNISTModel(batch_size=128, data_shape=dataset.data_shape,
        >>>                    output_dims=len(dataset.unique_labels),
        >>>                    training_dpath=dataset.training_dpath)
        >>> output_layer = model.init_arch()
        >>> model.print_model_info_str()
        >>> model.mode = 'FAST_COMPILE'
        >>> model.build_backprop_func()

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia_cnn.models.mnist import *  # NOQA
        >>> from wbia_cnn.models import mnist
        >>> model, dataset = mnist.testdata_mnist()
        >>> model.init_arch()
        >>> model.print_layer_info()
        >>> model.print_model_info_str()
        >>> #model.reinit_weights()
        >>> X_train, y_train = dataset.subset('train')
        >>> model.fit(X_train, y_train)
        >>> output_layer = model.init_arch()
        >>> model.print_layer_info()
        >>> # parse training arguments
        >>> model.monitor_config.update(**ut.argparse_dict(dict(
        >>>     era_size=100,
        >>>     max_epochs=5,
        >>>     rate_schedule=.8,
        >>> )))
        >>> X_train, y_train = dataset.subset('train')
        >>> model.fit(X_train, y_train)

    """

    def __init__(model, **kwargs):
        model.batch_norm = kwargs.pop('batch_norm', True)
        model.dropout = kwargs.pop('dropout', 0.5)
        super(MNISTModel, model).__init__(**kwargs)

    def fit(model, *args, **kwargs):
        """
        CommandLine:
            python -m wbia_cnn.models.mnist MNISTModel.fit --show

        Example:
            >>> # DISABLE_DOCTEST
            >>> from wbia_cnn.models.mnist import *  # NOQA
            >>> from wbia_cnn.models import mnist
            >>> model, dataset = mnist.testdata_mnist()
            >>> model.init_arch()
            >>> # parse training arguments
            >>> model.monitor_config.update(**ut.argparse_dict(dict(
            >>>     era_size=20,
            >>>     max_epochs=100,
            >>>     rate_schedule=.9,
            >>> )))
            >>> X_train, y_train = dataset.subset('train')
            >>> model.fit(X_train, y_train)
        """
        super(MNISTModel, model).fit(*args, **kwargs)

    def augment(model, Xb, yb=None):
        r"""
        CommandLine:
            python -m wbia_cnn.models.mnist MNISTModel.augment --show

        Example:
            >>> from wbia_cnn.models.mnist import *  # NOQA
            >>> from wbia_cnn.models import mnist
            >>> import numpy as np
            >>> model, dataset = mnist.testdata_mnist()
            >>> model._rng = ut.ensure_rng(model.hyperparams['random_seed'])
            >>> X_valid, y_valid = dataset.subset('test')
            >>> num = 10
            >>> Xb = X_valid[:num]
            >>> yb = None
            >>> Xb = Xb / 255.0 if ut.is_int(Xb) else Xb
            >>> Xb = Xb.astype(np.float32, copy=True)
            >>> yb = None if yb is None else yb.astype(np.int32, copy=True)
            >>> # Rescale the batch data to the range 0 to 1
            >>> Xb_, yb_ = model.augment(Xb.copy())
            >>> yb_ = None
            >>> ut.quit_if_noshow()
            >>> from wbia import plottool as pt
            >>> pt.qt4ensure()
            >>> from wbia_cnn import augment
            >>> augment.show_augmented_patches(Xb, Xb_, yb, yb_, data_per_label=1)
            >>> ut.show_if_requested()
        """
        from wbia_cnn import augment

        rng = model._rng
        affperterb_ranges = dict(
            # zoom_range=(1.1, 0.9),
            zoom_range=(1.0, 1.0),
            max_tx=0,
            max_ty=0,
            max_shear=0,
            max_theta=ut.TAU / 64,
            enable_stretch=False,
            enable_flip=False,
        )
        Xb_, yb_ = augment.augment_affine(
            Xb,
            yb,
            rng=rng,
            inplace=True,
            data_per_label=1,
            affperterb_ranges=affperterb_ranges,
            aug_prop=0.5,
        )
        return Xb_, yb_

    def init_arch(model):
        """

        CommandLine:
            python -m wbia_cnn  MNISTModel.init_arch --verbcnn
            python -m wbia_cnn  MNISTModel.init_arch --verbcnn --show

            python -m wbia_cnn  MNISTModel.init_arch --verbcnn --name=bnorm --show
            python -m wbia_cnn  MNISTModel.init_arch --verbcnn --name=incep --show

        Example:
            >>> # ENABLE_DOCTEST
            >>> from wbia_cnn.models.mnist import *  # NOQA
            >>> verbose = True
            >>> name = ut.get_argval('--name', default='bnorm')
            >>> model = MNISTModel(batch_size=128, data_shape=(28, 28, 1),
            >>>                    output_dims=10, name=name)
            >>> model.init_arch()
            >>> model.print_model_info_str()
            >>> print(model)
            >>> ut.quit_if_noshow()
            >>> model.show_arch()
            >>> ut.show_if_requested()
        """
        logger.info('[model] init_arch')
        if True:
            logger.info('[model] Initialize MNIST model architecture')
            logger.info('[model]   * batch_size     = %r' % (model.batch_size,))
            logger.info('[model]   * input_width    = %r' % (model.input_width,))
            logger.info('[model]   * input_height   = %r' % (model.input_height,))
            logger.info('[model]   * input_channels = %r' % (model.input_channels,))
            logger.info('[model]   * output_dims    = %r' % (model.output_dims,))
        name = model.name

        if name is None:
            name = 'lenet'

        if name.startswith('incep'):
            network_layers_def = model.get_inception_def()
        elif name.startswith('resnet'):
            network_layers_def = model.get_resnet_def()
        elif name.startswith('mnist'):
            network_layers_def = model.get_mnist_def()
        else:
            network_layers_def = model.get_lenet_def()
        from wbia_cnn import custom_layers

        network_layers = custom_layers.evaluate_layer_list(network_layers_def)
        model.output_layer = network_layers[-1]
        return model.output_layer

    def get_mnist_def(model):
        """
        Follows https://github.com/Lasagne/Lasagne/blob/master/examples/mnist.py

        python -m wbia_cnn MNISTModel.init_arch --verbcnn --name=mnist --show
        python -m wbia_cnn.models.mnist MNISTModel.fit:0 --name=mnist --vd --monitor
        """
        import lasagne
        from wbia_cnn import custom_layers

        batch_norm = False

        bundles = custom_layers.make_bundles(
            nonlinearity=lasagne.nonlinearities.rectify,
            batch_norm=batch_norm,
            W=lasagne.init.Orthogonal('relu'),
        )
        b = ut.DynStruct(copy_dict=bundles)

        network_layers_def = [
            b.InputBundle(shape=model.input_shape),
            b.ConvBundle(num_filters=32, filter_size=(5, 5), pool=True),
            b.ConvBundle(num_filters=32, filter_size=(5, 5), pool=True),
            # A fully-connected layer and 50% dropout of its inputs
            b.DenseBundle(num_units=256, dropout=0.5),
            # And, finally, the 10-unit output layer with 50% dropout on its inputs
            b.SoftmaxBundle(num_units=model.output_dims, dropout=0.5),
        ]
        return network_layers_def

    def get_lenet_def(model):
        """
        python -m wbia_cnn MNISTModel.init_arch --verbcnn --name=lenet --show
        python -m wbia_cnn.models.mnist MNISTModel.fit:0 --name=lenet --vd --monitor
        """
        import lasagne
        from wbia_cnn import custom_layers

        batch_norm = model.batch_norm
        dropout = model.dropout

        W = lasagne.init.Orthogonal('relu')

        bundles = custom_layers.make_bundles(
            nonlinearity=lasagne.nonlinearities.rectify,
            batch_norm=batch_norm,
            W=W,
        )
        b = ut.DynStruct(copy_dict=bundles)

        N = 128

        network_layers_def = [
            b.InputBundle(shape=model.input_shape, noise=False),
            b.ConvBundle(num_filters=N, filter_size=(3, 3), pool=True),
            b.ConvBundle(num_filters=N, filter_size=(3, 3), pool=False),
            b.ConvBundle(num_filters=N, filter_size=(3, 3), pool=True),
            b.ConvBundle(num_filters=N, filter_size=(3, 3), pool=False),
            b.ConvBundle(num_filters=N, filter_size=(2, 2), pool=False),
            # A fully-connected layer and 50% dropout of its inputs
            b.DenseBundle(num_units=N * 2, dropout=dropout),
            # A fully-connected layer and 50% dropout of its inputs
            b.DenseBundle(num_units=N * 2, dropout=dropout),
            # And, finally, the 10-unit output layer with 50% dropout on its inputs
            b.SoftmaxBundle(num_units=model.output_dims, dropout=dropout),
        ]
        return network_layers_def

    def get_resnet_def(model):
        """
        A residual network with pre-activations

        python -m wbia_cnn MNISTModel.init_arch --verbcnn --name=resnet --show
        python -m wbia_cnn.models.mnist MNISTModel.fit:0 --name=resnet --vd --monitor
        """
        import lasagne
        from wbia_cnn import custom_layers

        batch_norm = model.batch_norm

        W = lasagne.init.HeNormal(gain='relu')
        bundles = custom_layers.make_bundles(
            filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.rectify,
            batch_norm=batch_norm,
            W=W,
        )
        b = ut.DynStruct(copy_dict=bundles)

        N = 16

        network_layers_def = [
            b.InputBundle(shape=model.input_shape, noise=False),
            b.ConvBundle(num_filters=N, pool=False),
            b.ResidualBundle(num_filters=N, stride=(2, 2), preactivate=False),
            b.ResidualBundle(num_filters=N),
            b.ResidualBundle(num_filters=N, stride=(2, 2)),
            # b.ResidualBundle(num_filters=N),
            b.ResidualBundle(num_filters=N, stride=(2, 2), dropout=0.2),
            b.ResidualBundle(
                num_filters=N, stride=(2, 2), dropout=0.5, postactivate=True
            ),
            b.AveragePool(),
            b.SoftmaxBundle(num_units=model.output_dims, dropout=0.5),
        ]
        return network_layers_def

    def get_inception_def(model):
        """
        python -m wbia_cnn MNISTModel.init_arch --verbcnn --name=resnet --show
        python -m wbia_cnn.models.mnist MNISTModel.fit:0 --name=resnet --vd --monitor

        """
        import lasagne
        from wbia_cnn import custom_layers

        batch_norm = model.batch_norm
        if model.dropout is None:
            dropout = 0 if batch_norm else 0.5
        else:
            dropout = model.dropout

        bundles = custom_layers.make_bundles(
            nonlinearity=lasagne.nonlinearities.rectify,
            batch_norm=batch_norm,
        )
        b = ut.DynStruct(copy_dict=bundles)

        N = 64

        network_layers_def = [
            b.InputBundle(shape=model.input_shape, noise=False),
            b.ConvBundle(num_filters=N, filter_size=(3, 3), pool=False),
            b.ConvBundle(num_filters=N, filter_size=(3, 3), pool=True),
            b.InceptionBundle(
                branches=[
                    dict(t='c', s=(1, 1), r=00, n=N),
                    dict(t='c', s=(3, 3), r=N // 2, n=N),
                    dict(t='c', s=(3, 3), r=N // 4, n=N // 2, d=2),
                    dict(t='p', s=(3, 3), n=N // 2),
                ],
            ),
            b.InceptionBundle(
                branches=[
                    dict(t='c', s=(1, 1), r=00, n=N),
                    dict(t='c', s=(3, 3), r=N // 2, n=N),
                    dict(t='c', s=(3, 3), r=N // 4, n=N // 2, d=2),
                    dict(t='p', s=(3, 3), n=N // 2),
                ],
                dropout=dropout,
                pool=True,
            ),
            # ---
            b.DenseBundle(num_units=N, dropout=dropout),
            b.DenseBundle(num_units=N, dropout=dropout),
            # And, finally, the 10-unit output layer with 50% dropout on its inputs
            b.SoftmaxBundle(num_units=model.output_dims, dropout=dropout),
            # b.GlobalPool
            # b.NonlinearitySoftmax(),
        ]
        return network_layers_def


def testdata_mnist(defaultname='lenet', batch_size=128, dropout=None):
    from wbia_cnn import ingest_data
    from wbia_cnn.models import mnist

    dataset = ingest_data.grab_mnist_category_dataset()
    name = ut.get_argval('--name', default=defaultname)
    if name.startswith('lenet'):
        batch_norm = True
        dropout = 0.5
    elif name == 'bnorm':
        batch_norm = True
        dropout = dropout
    elif name == 'dropout':
        batch_norm = False
        dropout = 0.5
    else:
        batch_norm = True
        dropout = dropout
    output_dims = len(dataset.unique_labels)
    model = mnist.MNISTModel(
        batch_size=batch_size,
        data_shape=dataset.data_shape,
        name=name,
        output_dims=output_dims,
        batch_norm=batch_norm,
        dropout=dropout,
        dataset_dpath=dataset.dataset_dpath,
    )
    model.monitor_config['monitor'] = True
    model.monitor_config['showprog'] = False
    model.monitor_config['slowdump_freq'] = 10

    model.learn_state['learning_rate'] = 0.01
    model.hyperparams['weight_decay'] = 0.001
    model.hyperparams['era_size'] = 20
    model.hyperparams['rate_schedule'] = [0.9]

    if model.name is None:
        pass
    elif model.name == 'bnorm':
        model.hyperparams['era_size'] = 4
        model.hyperparams['rate_schedule'] = [0.9]
    elif name.startswith('mnist'):
        model.hyperparams['rate_schedule'] = [1.0]
        model.learn_state['learning_rate'] = 0.01
        model.hyperparams['weight_decay'] = 0
        model.hyperparams['augment_on'] = False
        model.hyperparams['whiten_on'] = False
    elif name.startswith('resnet'):
        model.hyperparams['era_size'] = 10
        model.hyperparams['rate_schedule'] = [0.5]
        model.learn_state['learning_rate'] = 0.01
        model.hyperparams['weight_decay'] = 0
        model.hyperparams['augment_on'] = True
    return model, dataset


if __name__ == '__main__':
    """
    CommandLine:
        python -m wbia_cnn.models.mnist
        python -m wbia_cnn.models.mnist --allexamples
        python -m wbia_cnn.models.mnist --allexamples --noface --nosrc
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
