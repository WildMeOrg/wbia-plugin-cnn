# -*- coding: utf-8 -*-
import logging
import functools
import numpy as np
import utool as ut
from six.moves import cPickle as pickle  # NOQA
from wbia_cnn import net_strs

print, rrr, profile = ut.inject2(__name__)
logger = logging.getLogger()


class PretrainedNetwork(object):
    """
    TODO: move to new class

    Intialize weights from a specified (Caffe) pretrained network layers

    Args:
        layer (int) : int

    CommandLine:
        python -m wbia_cnn --tf PretrainedNetwork:0
        python -m wbia_cnn --tf PretrainedNetwork:1

    Example0:
        >>> # DISABLE_DOCTEST
        >>> from wbia_cnn.models import *  # NOQA
        >>> self = PretrainedNetwork('caffenet', show_network=True)
        >>> print('done')

    Example1:
        >>> # DISABLE_DOCTEST
        >>> from wbia_cnn.models import *  # NOQA
        >>> self = PretrainedNetwork('vggnet', show_network=True)
        >>> print('done')
    """

    def __init__(self, model_key=None, show_network=False):
        from wbia_cnn._plugin_grabmodels import ensure_model

        self.model_key = model_key
        weights_path = ensure_model(model_key)
        try:
            self.pretrained_weights = ut.load_cPkl(weights_path)
        except Exception:
            raise IOError('The specified model was not found: %r' % (weights_path,))
        if show_network:
            net_strs.print_pretrained_weights(self.pretrained_weights, weights_path)

    def get_num_layers(self):
        return len(self.pretrained_weights)

    def get_layer_num_filters(self, layer_index):
        assert layer_index <= len(
            self.pretrained_weights
        ), 'Trying to specify a layer that does not exist'
        shape = self.pretrained_weights[layer_index].shape
        fanout, fanin, height, width = shape
        return fanout

    def get_layer_filter_size(self, layer_index):
        assert layer_index <= len(
            self.pretrained_weights
        ), 'Trying to specify a layer that does not exist'
        shape = self.pretrained_weights[layer_index].shape
        fanout, fanin, height, width = shape
        return (height, width)

    def get_conv2d_layer(self, layer_index, name=None, **kwargs):
        """Assumes requested layer is convolutional

        Returns:
            lasagne.layers.Layer: Layer
        """
        if name is None:
            name = '%s_layer%r' % (self.model_key, layer_index)
        W = self.get_pretrained_layer(layer_index)
        try:
            b = self.get_pretrained_layer(layer_index + 1)
            assert W.shape[0] == b.shape[0]
        except Exception:
            b = None
        logger.info(W.shape)
        logger.info(b.shape)
        num_filters = self.get_layer_num_filters(layer_index)
        filter_size = self.get_layer_filter_size(layer_index)

        from wbia_cnn import custom_layers

        Conv2DLayer = custom_layers.Conv2DLayer
        # MaxPool2DLayer = custom_layers.MaxPool2DLayer

        Layer = functools.partial(
            Conv2DLayer,
            num_filters=num_filters,
            filter_size=filter_size,
            W=W,
            b=b,
            name=name,
            **kwargs
        )
        return Layer

    def get_pretrained_layer(self, layer_index, rand=False):
        import lasagne

        assert layer_index <= len(
            self.pretrained_weights
        ), 'Trying to specify a layer that does not exist'
        pretrained_layer = self.pretrained_weights[layer_index]

        class _PretrainedLayerInitializer(lasagne.init.Initializer):
            def __init__(pt, pretrained_layer, model_key, layer_index):
                pt.pretrained_layer = pretrained_layer
                pt.model_key = model_key
                pt.layer_index = layer_index

            def sample(pt, shape):
                args = (
                    pt.layer_index,
                    pt.model_key,
                )
                logger.info(
                    '[pretrained] Sampling layer %d from pretrained model %r' % args
                )
                if len(shape) == 1:
                    assert shape[0] <= pt.pretrained_layer.shape[0]
                    pretrained_weights = pt.pretrained_layer[: shape[0]]
                else:
                    is_conv = len(shape) == 4
                    assert len(shape) == len(
                        pt.pretrained_layer.shape
                    ), 'Layer shape mismatch. Expected %r got %r' % (
                        pt.pretrained_layer.shape,
                        shape,
                    )
                    fanout, fanin = shape[:2]
                    fanout_, fanin_ = pt.pretrained_layer.shape[:2]
                    assert (
                        fanout <= fanout_
                    ), 'Cannot increase weight fan-out dimension from %d to %d' % (
                        fanout,
                        fanout_,
                    )  # NOQA
                    assert (
                        fanin <= fanin_
                    ), 'Cannot increase weight fan-in dimension from %d to %d' % (
                        fanin,
                        fanin_,
                    )  # NOQA
                    if is_conv:
                        height, width = shape[2:]
                        height_, width_ = pt.pretrained_layer.shape[2:]
                        assert height == height_, 'Layer height must equal Weight height'
                        assert width == width_, 'Layer width must equal Weight width'
                    if is_conv:
                        pretrained_weights = pt.pretrained_layer[:fanout, :fanin, :, :]
                    else:
                        pretrained_weights = pt.pretrained_layer[:fanout, :fanin]
                pretrained_sample = lasagne.utils.floatX(pretrained_weights)
                return pretrained_sample

        weights_initializer = _PretrainedLayerInitializer(
            pretrained_layer, self.model_key, layer_index
        )
        if rand:
            np.random.shuffle(weights_initializer)
        return weights_initializer
