# -*- coding: utf-8 -*-
import logging
import utool as ut
from wbia_cnn.models import abstract_models

print, rrr, profile = ut.inject2(__name__)
logger = logging.getLogger()


@ut.reloadable_class
class DummyModel(abstract_models.AbstractCategoricalModel):
    def __init__(model, batch_size=8, data_shape=(4, 4, 1), **kwargs):
        # kwargs['autoinit'] = kwargs.get('autoinit', True)
        kwargs['output_dims'] = kwargs.get('output_dims', 3)
        kwargs['showprog'] = kwargs.get('showprog', False)
        super(DummyModel, model).__init__(
            data_shape=data_shape, batch_size=batch_size, **kwargs
        )

    def init_arch(model, verbose=True):
        """
        CommandLine:
            python -m wbia_cnn DummyModel.init_arch --verbcnn --show

        Example:
            >>> # ENABLE_DOCTEST
            >>> from wbia_cnn.models.dummy import *  # NOQA
            >>> model = DummyModel(autoinit=True)
            >>> model.print_model_info_str()
            >>> print(model)
            >>> ut.quit_if_noshow()
            >>> model.show_arch()
            >>> ut.show_if_requested()
        """
        import lasagne
        from wbia_cnn import custom_layers

        if verbose:
            logger.info('init arch')

        bundles = custom_layers.make_bundles(
            nonlinearity=lasagne.nonlinearities.rectify,
            batch_norm=False,
        )
        b = ut.DynStruct(copy_dict=bundles)

        network_layers_def = [
            b.InputBundle(shape=model.input_shape),
            b.ConvBundle(num_filters=5, filter_size=(3, 3)),
            b.DenseBundle(num_units=8),
            b.SoftmaxBundle(num_units=model.output_dims),
        ]

        from wbia_cnn import custom_layers

        network_layers = custom_layers.evaluate_layer_list(network_layers_def)
        # model.network_layers = network_layers
        model.output_layer = network_layers[-1]
        if ut.VERBOSE:
            model.print_arch_str()
            model.print_layer_info()
        return model.output_layer


if __name__ == '__main__':
    """
    CommandLine:
        python -m wbia_cnn.models.dummy
        python -m wbia_cnn.models.dummy --allexamples
        python -m wbia_cnn.models.dummy --allexamples --noface --nosrc
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
