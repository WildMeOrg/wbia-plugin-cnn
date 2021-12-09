# -*- coding: utf-8 -*-
import logging
import numpy as np
import warnings
import six
import theano
import functools
from theano import tensor as T  # NOQA
import lasagne
from lasagne import init, layers, nonlinearities
from wbia_cnn import utils
import utool as ut

(print, rrr, profile) = ut.inject2(__name__)
logger = logging.getLogger()


FORCE_CPU = False  # ut.get_argflag('--force-cpu')
USING_GPU = False
try:
    if FORCE_CPU:
        raise ImportError('GPU is forced off')
    # use cuda_convnet for a speed improvement
    # will not be available without a GPU

    conv_impl = 'cuDNN'
    # conv_impl = 'cuda_convnet'
    if ut.get_computer_name().lower() == 'hyrule':
        # cuda_convnet seems broken on hyrule
        conv_impl = 'cuDNN'

    # http://lasagne.readthedocs.org/en/latest/modules/layers/conv.html#layers.Conv2DLayer

    if conv_impl == 'cuda_convnet':
        # cannot handle non-square images (pylearn2 module)
        import layers.cuda_convnet

        Conv2DLayer = layers.cuda_convnet.Conv2DCCLayer
        MaxPool2DLayer = layers.cuda_convnet.MaxPool2DCCLayer
    elif conv_impl == 'cuDNN':
        import layers.dnn

        Conv2DLayer = layers.dnn.Conv2DDNNLayer
        MaxPool2DLayer = layers.dnn.MaxPool2DDNNLayer
        """
        Need cuda convnet for background model otherwise
        <type 'exceptions.ValueError'>: GpuReshape: cannot reshape input of shape (128, 12, 26, 26) to shape (128, 676).
        Apply node that caused the error: GpuReshape{2}(GpuElemwise{Composite{((i0 * (i1 + i2)) + (i3 * Abs((i1 + i2))))}}[(0, 1)].0, TensorConstant{[128 676]})
        Toposort index: 36
        Inputs types: [CudaNdarrayType(float32, 4D), TensorType(int64, vector)]
        Inputs shapes: [(128, 12, 26, 26), (2,)]
        Inputs strides: [(676, 86528, 26, 1), (8,)]
        Inputs values: ['not shown', array([128, 676])]
        Outputs clients: [[GpuDot22(GpuReshape{2}.0, GpuReshape{2}.0)]]
        """
    elif conv_impl == 'gemm':
        # Dont use gemm
        import layers.corrmm

        Conv2DLayer = layers.corrmm.Conv2DLayer
        MaxPool2DLayer = layers.corrmm.Conv2DLayer
    else:
        raise NotImplementedError('conv_impl = %r' % (conv_impl,))

    USING_GPU = True
except (Exception, ImportError) as ex:
    Conv2DLayer = layers.Conv2DLayer
    MaxPool2DLayer = layers.MaxPool2DLayer

    if utils.VERBOSE_CNN:
        logger.info('Conv2DLayer = %r' % (Conv2DLayer,))
        logger.info('MaxPool2DLayer = %r' % (MaxPool2DLayer,))

    if theano.config.device != 'cpu':
        ut.printex(ex, 'WARNING: GPU seems unavailable', iswarning=True)

if utils.VERBOSE_CNN:
    logger.info(
        'lasagne.__version__ = %r' % getattr(lasagne, '__version__', None),
    )
    logger.info('lasagne.__file__ = %r' % (getattr(lasagne, '__file__', None),))
    logger.info('theano.__version__ = %r' % (getattr(theano, '__version__', None),))
    logger.info('theano.__file__ = %r' % (getattr(theano, '__file__', None),))


class L1NormalizeLayer(layers.Layer):
    def __init__(self, input_layer, *args, **kwargs):
        super(L1NormalizeLayer, self).__init__(input_layer, *args, **kwargs)

    def get_output_for(self, input_, *args, **kwargs):
        ell1_norm = T.abs_(input_).sum(axis=1)
        output_ = input_ / ell1_norm[:, None]
        return output_


@six.add_metaclass(ut.ReloadingMetaclass)
class LocallyConnected2DLayer(layers.Layer):
    """
    Copy of the Conv2D layer that needs to be adapted into a locally connected layer

    Args:
        incoming (layers.Layer):
        num_filters (?):
        filter_size (?):
        stride (tuple): (default = (1, 1))
        pad (int): (default = 0)
        untie_biases (bool): (default = False)
        W (GlorotUniform): (default = <init.GlorotUniform object at 0x7f551a3537d0>)
        b (Constant): (default = <init.Constant object at 0x7f551a33ecd0>)
        nonlinearity (function): (default = <function rectify at 0x7f55307989b0>)
        convolution (function): (default = <function conv2d at 0x7f55330148c0>)

    CommandLine:
        python -m wbia_cnn.custom_layers --exec-__init__

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia_cnn.custom_layers import *  # NOQA
        >>> incoming = testdata_input_layer(item_shape=(3,8,8), batch_size=4)
        >>> num_filters = 64
        >>> filter_size = (3, 3)
        >>> stride = (1, 1)
        >>> pad = 0
        >>> untie_biases = False
        >>> W = init.GlorotUniform()
        >>> b = init.Constant(0.)
        >>> nonlinearity = nonlinearities.rectify
        >>> convolution = T.nnet.conv2d
        >>> self = LocallyConnected2DLayer(incoming, num_filters, filter_size,
        >>>                                stride, pad, untie_biases, W, b,
        >>>                                nonlinearity, convolution)

    Ignore:
        self.get_output_rc(self.input_shape)
    """

    def __init__(
        self,
        incoming,
        num_filters,
        filter_size,
        stride=(1, 1),
        pad=0,
        untie_biases=False,
        W=init.GlorotUniform(),
        b=init.Constant(0.0),
        nonlinearity=nonlinearities.rectify,
        convolution=T.nnet.conv2d,
        **kwargs
    ):
        super(LocallyConnected2DLayer, self).__init__(incoming, **kwargs)
        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity

        self.num_filters = num_filters
        self.filter_size = lasagne.utils.as_tuple(filter_size, 2)
        self.stride = lasagne.utils.as_tuple(stride, 2)
        self.untie_biases = untie_biases
        self.convolution = convolution

        if pad == 'same':
            if any(s % 2 == 0 for s in self.filter_size):
                raise NotImplementedError('`same` padding requires odd filter size.')

        if pad == 'valid':
            self.pad = (0, 0)
        elif pad in ('full', 'same'):
            self.pad = pad
        else:
            self.pad = lasagne.utils.as_tuple(pad, 2, int)

        self.W = self.add_param(W, self.get_W_shape(), name='W')
        if b is None:
            self.b = None
        else:
            if self.untie_biases:
                biases_shape = (num_filters, self.output_shape[2], self.output_shape[3])
            else:
                biases_shape = (num_filters,)
            self.b = self.add_param(b, biases_shape, name='b', regularizable=False)

    def get_W_shape(self):
        """Get the shape of the weight matrix `W`.

        Returns
        -------
        tuple of int
            The shape of the weight matrix.

            (should have a different conv matrix for each output node)
            (ie NO WEIGHT SHARING)
        """
        num_input_channels = self.input_shape[1]
        output_rows, output_cols = self.get_output_rc(self.input_shape)
        return (
            self.num_filters,
            num_input_channels,
            self.filter_size[0],
            self.filter_size[1],
            output_rows,
            output_cols,
        )

    def get_output_rc(self, input_shape):
        pad = self.pad if isinstance(self.pad, tuple) else (self.pad,) * 2

        output_rows = layers.conv.conv_output_length(
            input_shape[2], self.filter_size[0], self.stride[0], pad[0]
        )

        output_columns = layers.conv.conv_output_length(
            input_shape[3], self.filter_size[1], self.stride[1], pad[1]
        )

        return output_rows, output_columns

    def get_output_shape_for(self, input_shape):
        output_rows, output_columns = self.get_output_rc(input_shape)
        return (input_shape[0], self.num_filters, output_rows, output_columns)

    def get_output_for(self, input, input_shape=None, **kwargs):
        # The optional input_shape argument is for when get_output_for is
        # called directly with a different shape than self.input_shape.
        if input_shape is None:
            input_shape = self.input_shape

        if self.stride == (1, 1) and self.pad == 'same':
            # simulate same convolution by cropping a full convolution
            conved = self.convolution(
                input,
                self.W,
                subsample=self.stride,
                image_shape=input_shape,
                filter_shape=self.get_W_shape(),
                border_mode='full',
            )
            crop_x = self.filter_size[0] // 2
            crop_y = self.filter_size[1] // 2
            conved = conved[:, :, crop_x : -crop_x or None, crop_y : -crop_y or None]
        else:
            # no padding needed, or explicit padding of input needed
            if self.pad == 'full':
                border_mode = 'full'
                pad = [(0, 0), (0, 0)]
            elif self.pad == 'same':
                border_mode = 'valid'
                pad = [
                    (self.filter_size[0] // 2, self.filter_size[0] // 2),
                    (self.filter_size[1] // 2, self.filter_size[1] // 2),
                ]
            else:
                border_mode = 'valid'
                pad = [(self.pad[0], self.pad[0]), (self.pad[1], self.pad[1])]
            if pad != [(0, 0), (0, 0)]:
                input = lasagne.theano_extensions.padding.pad(input, pad, batch_ndim=2)
                input_shape = (
                    input_shape[0],
                    input_shape[1],
                    None
                    if input_shape[2] is None
                    else input_shape[2] + pad[0][0] + pad[0][1],
                    None
                    if input_shape[3] is None
                    else input_shape[3] + pad[1][0] + pad[1][1],
                )
            conved = self.convolution(
                input,
                self.W,
                subsample=self.stride,
                image_shape=input_shape,
                filter_shape=self.get_W_shape(),
                border_mode=border_mode,
            )

        if self.b is None:
            activation = conved
        elif self.untie_biases:
            activation = conved + self.b.dimshuffle('x', 0, 1, 2)
        else:
            activation = conved + self.b.dimshuffle('x', 0, 'x', 'x')

        return self.nonlinearity(activation)


# @six.add_metaclass(ut.ReloadingMetaclass)
class L2NormalizeLayer(layers.Layer):
    """
    Normalizes the outputs of a layer to have an L2 norm of 1.
    This is useful for siamese networks who's outputs will be comparsed using
    the L2 distance.

    """

    def __init__(self, input_layer, axis=1, **kwargs):
        super(L2NormalizeLayer, self).__init__(input_layer, **kwargs)
        self.axis = axis

    def get_output_for(self, input_, axis=None, T=T, **kwargs):
        """
        CommandLine:
            python -m wbia_cnn.custom_layers --test-L2NormalizeLayer.get_output_for

        Example0:
            >>> # ENABLE_DOCTEST
            >>> from wbia_cnn.custom_layers import *  # NOQA
            >>> # l2 normalization on batches of vector encodings
            >>> input_layer = testdata_input_layer(item_shape=(8,), batch_size=4)
            >>> inputdata_ = np.random.rand(*input_layer.shape).astype(np.float32)
            >>> axis = 1
            >>> self = L2NormalizeLayer(input_layer, axis=axis)
            >>> # Test numpy version
            >>> T = np
            >>> input_ = inputdata_
            >>> output_np = self.get_output_for(inputdata_, T=np)
            >>> assert np.all(np.isclose(np.linalg.norm(output_np, axis=axis), 1.0))
            >>> # Test theano version
            >>> T = theano.tensor
            >>> input_expr = input_ = T.matrix(name='vector_input')
            >>> output_expr = self.get_output_for(input_expr, T=T)
            >>> output_T = output_expr.eval({input_expr: inputdata_})
            >>> print(output_T)
            >>> assert np.all(np.isclose(output_T, output_np)), 'theano and numpy diagree'

        Example1:
            >>> # ENABLE_DOCTEST
            >>> from wbia_cnn.custom_layers import *  # NOQA
            >>> # l2 normalization on batches of image filters
            >>> input_layer = testdata_input_layer(item_shape=(3, 2, 2), batch_size=4)
            >>> inputdata_ = np.random.rand(*input_layer.shape).astype(np.float32)
            >>> axis = 2
            >>> self = L2NormalizeLayer(input_layer, axis=axis)
            >>> # Test numpy version
            >>> T = np
            >>> input_ = inputdata_
            >>> output_np = self.get_output_for(inputdata_, T=np)
            >>> output_flat_np = output_np.reshape(np.prod(input_layer.shape[0:2]), np.prod(input_layer.shape[2:4]))
            >>> assert np.all(np.isclose(np.linalg.norm(output_flat_np, axis=1), 1.0))
            >>> # Test theano version
            >>> T = theano.tensor
            >>> input_expr = input_ = T.tensor4(name='image_filter_input')
            >>> output_expr = self.get_output_for(input_expr, T=T)
            >>> output_T = output_expr.eval({input_expr: inputdata_})
            >>> print(output_T)
            >>> assert np.all(np.isclose(output_T, output_np)), 'theano and numpy diagree'
            >>> #output_T = utils.evaluate_symbolic_layer(self.get_output_for, inputdata_, T.tensor4, T=theano.tensor)
        """
        if axis is None:
            axis = self.axis

        input_shape = input_.shape
        batch_shape = input_shape[0:axis]
        rest_shape = input_shape[axis:]
        batch_size = T.prod(batch_shape)
        rest_size = T.prod(rest_shape)

        # reshape to two dimensions
        input_reshaped_ = input_.reshape((batch_size, rest_size))
        # if T is np:
        #    #input_reshaped_ = input_.reshape(batch_shape + (rest_size,))
        # else:
        #    # hack because I don't know how to get ndim yet
        #    if axis == 1:
        #        input_reshaped_ = input_.reshape(batch_shape + (rest_size,), ndim=2)
        #    elif axis == 2:
        #        input_reshaped_ = input_.reshape(batch_shape + (rest_size,), ndim=3)

        ell2_norm = T.sqrt(T.power(input_reshaped_, 2).sum(axis=-1))
        if T is np:
            # outputreshaped_ = input_reshaped_ / ell2_norm[..., None]
            outputreshaped_ = input_reshaped_ / ell2_norm[:, None]
            output_ = outputreshaped_.reshape(input_shape)
        else:
            outputreshaped_ = input_reshaped_ / ell2_norm[:, None]
            output_ = outputreshaped_.reshape(input_shape)
            output_.name = 'l2normalized(%s)' % (input_.name)
            # .dimshuffle(0, 'x', 1)
        return output_


class L2SquaredDistanceLayer(layers.Layer):
    def __init__(self, input_layer, *args, **kwargs):
        super(L2SquaredDistanceLayer, self).__init__(input_layer, *args, **kwargs)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0] // 2,) + input_shape[1:]

    def get_output_for(self, input_, *args, **kwargs):
        # Split batch into pairs
        G1, G2 = input_[0::2], input_[1::2]
        E = T.power((G1 - G2), 2).sum(axis=1)
        return E


class L1DistanceLayer(layers.Layer):
    def __init__(self, input_layer, *args, **kwargs):
        super(L1DistanceLayer, self).__init__(input_layer, *args, **kwargs)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0] // 2,) + input_shape[1:]

    def get_output_for(self, input_, *args, **kwargs):
        # Split batch into pairs
        G1, G2 = input_[0::2], input_[1::2]
        E = T.abs_((G1 - G2)).sum(axis=1)
        return E


def testdata_input_layer(item_shape=(3, 32, 32), batch_size=128):
    input_shape = (batch_size,) + item_shape
    input_layer = layers.InputLayer(shape=input_shape)
    return input_layer


class SiameseConcatLayer(layers.Layer):
    """

    TODO checkout layers.merge.ConcatLayer

    Takes two network representations in the batch and combines them along an axis.
    """

    def __init__(self, input_layer, data_per_label=2, axis=1, **kwargs):
        super(SiameseConcatLayer, self).__init__(input_layer, **kwargs)
        self.data_per_label = data_per_label
        self.axis = axis

    def get_output_shape_for(self, input_shape, axis=None):
        r"""

        Args:
            input_shape: shape being fed into this layer
            axis: overrideable for tests

        CommandLine:
            python -m wbia_cnn.custom_layers --test-get_output_shape_for

        Example0:
            >>> # ENABLE_DOCTEST
            >>> from wbia_cnn.custom_layers import *  # NOQA
            >>> input_layer = testdata_input_layer(item_shape=(3, 8, 16))
            >>> self = SiameseConcatLayer(input_layer)
            >>> input_shape = input_layer.shape
            >>> output_shape_list = [self.get_output_shape_for(input_shape, axis) for axis in [1, 2, 3, -3, -2, -1]]
            >>> result = str(output_shape_list[0:3]) + '\n' + str(output_shape_list[3:])
            >>> print(result)
            [(64, 6, 8, 16), (64, 3, 16, 16), (64, 3, 8, 32)]
            [(64, 6, 8, 16), (64, 3, 16, 16), (64, 3, 8, 32)]

        Example1:
            >>> # ENABLE_DOCTEST
            >>> from wbia_cnn.custom_layers import *  # NOQA
            >>> input_layer = testdata_input_layer(item_shape=(1024,))
            >>> self = SiameseConcatLayer(input_layer)
            >>> input_shape = input_layer.shape
            >>> output_shape_list = [self.get_output_shape_for(input_shape, axis) for axis in [1, -1]]
            >>> result = output_shape_list
            >>> print(result)
            [(64, 2048), (64, 2048)]
        """
        if axis is None:
            # allow override for tests
            axis = self.axis
        assert self.axis != 0, 'self.axis=%r cannot be 0' % (self.axis,)
        new_batch_shape = (input_shape[0] // self.data_per_label,)
        new_shape_middle = (input_shape[axis] * self.data_per_label,)
        if axis >= 0:
            shape_front = input_shape[1:axis]
            shape_end = input_shape[axis + 1 :]
        else:
            shape_front = input_shape[1:axis]
            shape_end = input_shape[len(input_shape) + axis + 1 :]
        output_shape = new_batch_shape + shape_front + new_shape_middle + shape_end
        return output_shape

    def get_output_for(self, input_, T=T, **kwargs):
        """
        CommandLine:
            python -m wbia_cnn.custom_layers --test-SiameseConcatLayer.get_output_for:1 --show

        Example0:
            >>> # ENABLE_DOCTEST
            >>> from wbia_cnn.custom_layers import *  # NOQA
            >>> input_shape = (128, 1024)
            >>> input_layer = layers.InputLayer(shape=input_shape)
            >>> self = SiameseConcatLayer(input_layer)
            >>> np.random.seed(0)
            >>> input_ = np.random.rand(*input_shape)
            >>> T = np
            >>> output_ = self.get_output_for(input_, T=T)
            >>> target_shape = self.get_output_shape_for(input_shape)
            >>> result = output_.shape
            >>> print(result)
            >>> assert target_shape == result
            (64, 2048)

        Example1:
            >>> # DISABLE_DOCTEST
            >>> from wbia_cnn.custom_layers import *  # NOQA
            >>> from wbia_cnn import utils
            >>> from wbia_cnn import draw_net
            >>> import theano
            >>> import numpy as np
            >>> input_layer = layers.InputLayer(shape=(4, 3, 32, 32))
            >>> cs_layer = CenterSurroundLayer(input_layer)
            >>> # Make sure that this can concat center surround properly
            >>> self = SiameseConcatLayer(cs_layer, axis=2, data_per_label=4)
            >>> data = utils.testdata_imglist()[0]
            >>> inputdata_ = utils.convert_cv2_images_to_theano_images(data)
            >>> outputdata_ = cs_layer.get_output_for(inputdata_, T=np)
            >>> input_ = outputdata_
            >>> output_ = self.get_output_for(input_, T=np)
            >>> ut.quit_if_noshow()
            >>> img_list = utils.convert_theano_images_to_cv2_images(output_)
            >>> interact_image_list(img_list, num_per_page=2)

        """
        data_per_label = self.data_per_label
        split_inputs = [input_[count::data_per_label] for count in range(data_per_label)]
        output_ = T.concatenate(split_inputs, axis=self.axis)
        # input1, input2 = input_[0::2], input_[1::2]
        # output_ =  T.concatenate([input1, input2], axis=1)
        return output_


def interact_image_list(img_list, num_per_page=1):
    # from wbia.viz import viz_helpers as vh
    from wbia import plottool as pt

    nRows, nCols = pt.get_square_row_cols(num_per_page)
    chunked_iter = list(ut.ichunks(img_list, num_per_page))
    for img_chunks in ut.InteractiveIter(chunked_iter, display_item=False):
        pnum_ = pt.make_pnum_nextgen(nRows, nCols)
        pt.figure(fnum=1, doclf=True)
        for img in img_chunks:
            pt.imshow(img, pnum=pnum_())
        # pt.draw_border(pt.gca(), color=vh.get_truth_color(label))
        # pt.imshow(patch2, pnum=(1, 2, 2))
        # pt.draw_border(pt.gca(), color=vh.get_truth_color(label))
        pt.update()


def testdata_centersurround(item_shape):
    input_layer = testdata_input_layer(item_shape)
    data = utils.testdata_imglist(item_shape)[0]
    self = CenterSurroundLayer(input_layer)
    inputdata_ = utils.convert_cv2_images_to_theano_images(data)
    return self, inputdata_


class CenterSurroundLayer(layers.Layer):
    def __init__(self, input_layer, *args, **kwargs):
        # self.name = kwargs.pop('name', None)
        super(CenterSurroundLayer, self).__init__(input_layer, *args, **kwargs)

    def get_output_shape_for(self, input_shape):
        batch_size, channels, height, width = input_shape
        if height % 2 == 1 or width % 2 == 1:
            warnings.warn(
                'input layer to CenterSurroundLayer should ideally have an even width and height.'
            )
        output_shape = (batch_size * 2, channels, height // 2, width // 2)
        return output_shape

    def get_output_for(self, input_expr, T=T, **kwargs):
        r"""
        CommandLine:
            python -m wbia_cnn.custom_layers --test-CenterSurroundLayer.get_output_for:0 --show
            python -m wbia_cnn.custom_layers --test-CenterSurroundLayer.get_output_for:1 --show

        Example0:
            >>> # DISABLE_DOCTEST
            >>> from wbia_cnn.custom_layers import *  # NOQA
            >>> import theano
            >>> #item_shape = (32, 32, 3)
            >>> item_shape = (41, 41, 3)
            >>> self, inputdata_ = testdata_centersurround(item_shape)
            >>> # Test the actual symbolic expression
            >>> output_T = utils.evaluate_symbolic_layer(self.get_output_for, inputdata_, T.tensor4, T=theano.tensor)
            >>> output_T = output_T.astype(np.uint8)
            >>> ut.quit_if_noshow()
            >>> img_list = utils.convert_theano_images_to_cv2_images(output_T)
            >>> interact_image_list(img_list, num_per_page=8)

        Example1:
            >>> # DISABLE_DOCTEST
            >>> from wbia_cnn.custom_layers import *  # NOQA
            >>> import numpy as np
            >>> #item_shape = (32, 32, 3)
            >>> item_shape = (41, 41, 3)
            >>> self, input_expr = testdata_centersurround(item_shape)
            >>> # Test using just numpy
            >>> output_np = self.get_output_for(input_expr, T=np)
            >>> print('results agree')
            >>> ut.quit_if_noshow()
            >>> img_list = utils.convert_theano_images_to_cv2_images(output_np)
            >>> interact_image_list(img_list, num_per_page=8)

        Ignore:
            from wbia_cnn import draw_net
            #draw_net.draw_theano_symbolic_expression(result)
            assert np.all(output_np == output_T)
            np.stack = np.vstack
            T = np
        """
        # Create a center and surround for each input patch
        # return input_
        input_shape = input_expr.shape
        batch_size, channels, height, width = input_shape

        left_h = height // 4
        left_w = width // 4
        right_h = left_h * 3
        right_w = left_w * 3
        # account for odd patches
        total_h = left_h * 4
        total_w = left_w * 4

        center = input_expr[:, :, left_h:right_h, left_w:right_w]
        surround = input_expr[:, :, 0:total_h:2, 0:total_w:2]

        output_shape = self.get_output_shape_for(input_shape)

        if T is theano.tensor:
            center.name = 'center'
            surround.name = 'surround'
            # production theano version
            output_expr = T.alloc(0.0, *output_shape)
            output_expr.name = 'center_surround_alloc'
            set_subtensor = functools.partial(T.set_subtensor)
            # set_subtensor = functools.partial(T.set_subtensor, inplace=True, tolerate_inplace_aliasing=True)
            output_expr = set_subtensor(output_expr[::2], center)
            output_expr = set_subtensor(output_expr[1::2], surround)
            output_expr.name = 'center_surround_output'
            # from wbia_cnn import draw_net
            # draw_net.draw_theano_symbolic_expression(output_expr)
        else:
            # debugging numpy version
            output_expr = np.empty(output_shape, dtype=input_expr.dtype)
            output_expr[::2] = center
            output_expr[1::2] = surround
        # output_expr = T.concatenate([center, surround], axis=0)
        return output_expr


class MultiImageSliceLayer(layers.Layer):
    """
    orig CyclicSliceLayer
    References:
        https://github.com/benanne/kaggle-ndsb/blob/master/dihedral.py#L89

    This layer stacks rotations of 0, 90, 180, and 270 degrees of the input
    along the batch dimension.
    If the input has shape (batch_size, num_channels, r, c),
    then the output will have shape (4 * batch_size, num_channels, r, c).
    Note that the stacking happens on axis 0, so a reshape to
    (4, batch_size, num_channels, r, c) will separate the slice axis.
    """

    def __init__(self, input_layer):
        super(MultiImageSliceLayer, self).__init__(input_layer)

    def get_output_shape_for(self, input_shape):
        return (4 * input_shape[0],) + input_shape[1:]

    def get_output_for(self, input_, *args, **kwargs):
        return lasagne.utils.concatenate(
            [
                # array_tf_0(input_),
                # array_tf_90(input_),
                # array_tf_180(input_),
                # array_tf_270(input_),
            ],
            axis=0,
        )


class MultiImageRollLayer(layers.Layer):
    """
    orig CyclicConvRollLayer


    This layer turns (n_views * batch_size, num_channels, r, c) into
    (n_views * batch_size, n_views * num_channels, r, c) by rolling
    and concatenating feature maps.
    It also applies the correct inverse transforms to the r and c
    dimensions to align the feature maps.

    References:
        https://github.com/benanne/kaggle-ndsb/blob/master/dihedral.py#L224
    """

    def __init__(self, input_layer):
        super(MultiImageRollLayer, self).__init__(input_layer)
        self.inv_tf_funcs = []  # array_tf_0, array_tf_270, array_tf_180, array_tf_90]
        self.compute_permutation_matrix()

    def compute_permutation_matrix(self):
        map_identity = np.arange(4)
        map_rot90 = np.array([1, 2, 3, 0])

        valid_maps = []
        current_map = map_identity
        for k in range(4):
            valid_maps.append(current_map)
            current_map = current_map[map_rot90]

        self.perm_matrix = np.array(valid_maps)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], 4 * input_shape[1]) + input_shape[2:]

    def get_output_for(self, input_, *args, **kwargs):
        s = input_.shape
        input_unfolded = input_.reshape((4, s[0] // 4, s[1], s[2], s[3]))

        permuted_inputs = []
        for p, inv_tf in zip(self.perm_matrix, self.inv_tf_funcs):
            input_permuted = inv_tf(input_unfolded[p].reshape(s))
            permuted_inputs.append(input_permuted)

        return lasagne.utils.concatenate(
            permuted_inputs, axis=1
        )  # concatenate long the channel axis


class CyclicPoolLayer(layers.Layer):
    """
    Utility layer that unfolds the viewpoints dimension and pools over it.
    Note that this only makes sense for dense representations, not for
    feature maps (because no inverse transforms are applied to align them).
    """

    def __init__(self, input_layer, pool_function=T.mean):
        super(CyclicPoolLayer, self).__init__(input_layer)
        self.pool_function = pool_function

    def get_output_shape_for(self, input_shape):
        return (input_shape[0] // 4, input_shape[1])

    def get_output_for(self, input_, *args, **kwargs):
        unfolded_input = input_.reshape((4, input_.shape[0] // 4, input_.shape[1]))
        return self.pool_function(unfolded_input, axis=0)


class BatchNormLayer2(layers.BatchNormLayer):
    """
    Adds a nonlinearity to batch norm layer to reduce number of layers
    """

    def __init__(self, incoming, nonlinearity=None, **kwargs):
        # this_class_now = ut.fix_super_reload_error(BatchNormLayer2, self)
        this_class_now = BatchNormLayer2
        super(this_class_now, self).__init__(incoming, **kwargs)
        # super(BatchNormLayer2, self).__init__(incoming, **kwargs)
        self.nonlinearity = (
            nonlinearities.identity if nonlinearity is None else nonlinearity
        )

    def get_output_for(self, input, **kwargs):
        # this_class_now = ut.fix_super_reload_error(BatchNormLayer2, self)
        this_class_now = BatchNormLayer2
        normalized = super(this_class_now, self).get_output_for(input, **kwargs)
        # normalized = super(BatchNormLayer2, self).get_output_for(input, **kwargs)
        normalized_activation = self.nonlinearity(normalized)
        return normalized_activation


class FlipLayer(layers.Layer):
    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_output_for(self, input, **kwargs):
        return input[:, :, ::-1, ::-1]


def load_json_arch_def(arch_json_fpath):
    """
    layer_list = layers.get_all_layers(output_layer)

    from wbia_cnn import net_strs
    layer_json_list = [net_strs.make_layer_json_dict(layer)
                       for layer in layer_list]

    arch_json_fpath = '/media/raid/work/WS_ALL/_ibsdb/_ibeis_cache/nets/injur-shark_10056_224x224x3_auqbfhle/models/arch_injur-shark_o2_d11_c688_acioqbst/saved_sessions/fit_session_2016-08-26T173854+5/fit_arch_info.json'
    """
    from wbia_cnn import custom_layers
    import layers.dnn

    # FIXME: Need to redo the saved arch json file.
    # Need to give layers identifiers and specify their inputs / outputs
    # They are not implicit
    arch_json = ut.load_json(arch_json_fpath)
    layer_json_list = arch_json['layers']

    network_def_list = []
    for layer_json in layer_json_list:
        classname = layer_json['type']
        classkw = ut.delete_dict_keys(layer_json.copy(), ['type', 'output_shape'])

        # Rectify nonlinearity definition
        if 'nonlinearity' in classkw:
            nonlin = classkw['nonlinearity']
            nonlinclass = getattr(lasagne.nonlinearities, nonlin['type'], None)
            nonlinkw = ut.delete_dict_keys(nonlin.copy(), ['type'])
            if nonlin['type'] in ['softmax', 'linear']:
                classkw['nonlinearity'] = nonlinclass
            else:
                classkw['nonlinearity'] = nonlinclass(**nonlinkw)

        classtype = None
        if classtype is None:
            classtype = getattr(lasagne.layers, classname, None)
        if classtype is None:
            classtype = getattr(layers.dnn, classname, None)
        if classtype is None:
            classtype = getattr(custom_layers, classname, None)

        if classtype is not None:
            layer_func = ut.NamedPartial(classtype, **classkw)
            network_def_list.append(layer_func)
        else:
            network_def_list.append(None)

    layer_list = custom_layers.evaluate_layer_list(network_def_list)
    # Hack, remove all biases before batch norm
    for layer in layer_list:
        if isinstance(layer, layers.BatchNormLayer):
            in_layer = layer.input_layer
            if in_layer is not None:
                if getattr(in_layer, 'b') is not None:
                    del in_layer.params[in_layer.b]
                    in_layer.b = None

    output_layer = layer_list[-1]
    return output_layer


def evaluate_layer_list(network_layers_def, verbose=None):
    r"""
    compiles a sequence of partial functions into a network
    """
    if verbose is None:
        verbose = utils.VERBOSE_CNN
    total = len(network_layers_def)
    network_layers = []
    if verbose:
        logger.info('Evaluting List of %d Layers' % (total,))
    layer_fn_iter = iter(network_layers_def)
    layer = None
    try:
        with ut.Indenter(' ' * 4, enabled=verbose):
            next_args = tuple()
            for count, layer_fn in enumerate(layer_fn_iter, start=1):
                if verbose:
                    logger.info(
                        'Evaluating layer %d/%d (%s) '
                        % (
                            count,
                            total,
                            ut.get_funcname(layer_fn),
                        )
                    )
                with ut.Timer(verbose=False) as tt:
                    layer = layer_fn(*next_args)
                next_args = (layer,)
                network_layers.append(layer)
                if verbose:
                    logger.info('  * took %.4fs' % (tt.toc(),))
                    logger.info('  * layer = %r' % (layer,))
                    if hasattr(layer, 'input_shape'):
                        logger.info('  * layer.input_shape = %r' % (layer.input_shape,))
                    if hasattr(layer, 'shape'):
                        logger.info('  * layer.shape = %r' % (layer.shape,))
                    logger.info('  * layer.output_shape = %r' % (layer.output_shape,))
    except Exception as ex:
        keys = [
            'layer_fn',
            'layer_fn.func',
            'layer_fn.args',
            'layer_fn.keywords',
            'layer_fn.__dict__',
            'layer',
            'count',
        ]
        ut.printex(ex, ('Error building layers.\n' 'layer=%r') % (layer,), keys=keys)
        raise
    return network_layers


# Bundle common layers together


def make_bundles(
    nonlinearity='lru',
    batch_norm=True,
    filter_size=(3, 3),
    stride=(1, 1),
    pool_stride=(2, 2),
    pool_size=(2, 2),
    branches=None,
    W=None,
):

    # FIXME; dropout is a pre-operation
    import itertools
    import six

    if W is None:
        # W = init.GlorotUniform()
        W = init.Orthogonal('relu')

    # Rectify default inputs
    if nonlinearity == 'lru':
        nonlinearity = nonlinearities.LeakyRectify(leakiness=(1.0 / 10.0))
    nonlinearity = nonlinearity
    namer = ut.partial(lambda x: str(six.next(x)), itertools.count(1))

    bundles = {}

    def register_bundle(class_):
        classname = ut.get_classname(class_, local=True)
        bundles[classname] = class_
        return class_

    class Bundle(object):
        def __init__(self):
            self.suffix = namer()
            self.name = self.suffix

        def debug_layer(self, layer):
            if False:
                logger.info('layer = %r' % layer)
                if hasattr(layer, 'name'):
                    logger.info('  * layer.name = %r' % layer.name)
                if hasattr(layer, 'input_shape'):
                    logger.info('  * layer.input_shape = %r' % (layer.input_shape,))
                if hasattr(layer, 'shape'):
                    logger.info('  * layer.shape = %r' % (layer.shape,))
                logger.info('  * layer.output_shape = %r' % (layer.output_shape,))

        def apply_dropout(self, layer):
            # change name standard
            outgoing = layers.DropoutLayer(layer, p=self.dropout, name='D' + self.name)
            return outgoing

        def apply_batch_norm(self, layer, **kwargs):
            # change name standard
            nonlinearity = getattr(layer, 'nonlinearity', None)
            if nonlinearity is not None:
                layer.nonlinearity = nonlinearities.identity
            if hasattr(layer, 'b') and layer.b is not None:
                del layer.params[layer.b]
                layer.b = None
            # bn_name = (kwargs.pop('name', None) or
            #           (getattr(layer, 'name', None) and self.name + '/bn'))
            bn_name = layer.name + '/bn'
            layer = BatchNormLayer2(
                layer, name=bn_name, nonlinearity=nonlinearity, **kwargs
            )
            # layer._is_main_layer = False
            # if nonlinearity is not None:
            #    nonlin_name = 'g' + self.name
            #    layer = layers.special.NonlinearityLayer(layer,
            #                                                     nonlinearity,
            #                                                     name=nonlin_name)
            # outgoing = layers.normalization.batch_norm(layer, **kwargs)
            # outgoing.input_layer.name = 'bn' + self.name
            # outgoing.name = 'nl' + self.name
            outgoing = layer
            return outgoing

    @register_bundle
    class InputBundle(Bundle):
        def __init__(self, shape, noise=False):
            self.shape = shape
            self.noise = noise
            super(InputBundle, self).__init__()

        def __call__(self):
            outgoing = layers.InputLayer(shape=self.shape, name='I' + self.name)
            if self.noise:
                outgoing = layers.GaussianNoiseLayer(outgoing, name='N' + self.name)
            return outgoing

    @register_bundle
    class ConvBundle(Bundle):
        def __init__(
            self,
            num_filters,
            filter_size=filter_size,
            stride=stride,
            nonlinearity=nonlinearity,
            batch_norm=batch_norm,
            pool_size=pool_size,
            W=W,
            pool_stride=pool_stride,
            dropout=None,
            pool=False,
            preactivate=False,
            pad=0,
            name=None,
        ):
            self.num_filters = num_filters
            self.filter_size = filter_size
            self.stride = stride
            self.nonlinearity = nonlinearity
            self.batch_norm = batch_norm
            self.pool_size = pool_size
            self.pool_stride = pool_stride
            self.dropout = dropout
            self.preactivate = preactivate
            self.pool = pool
            self.pad = pad
            self.W = W
            if name is None:
                super(ConvBundle, self).__init__()
                self.name = 'C' + self.suffix
            else:
                self.name = name

        def __call__(self, incoming):
            outgoing = incoming

            if self.preactivate or self.batch_norm:
                b = None
                nonlinearity = None
            else:
                b = init.Constant(0)
                nonlinearity = self.nonlinearity

            if self.preactivate:
                outgoing = BatchNormLayer2(
                    outgoing, nonlinearity=self.nonlinearity, name=self.name + '/_bn'
                )

            if self.dropout is not None and self.dropout > 0:
                outgoing = self.apply_dropout(outgoing)

            outgoing = Conv2DLayer(
                outgoing,
                num_filters=self.num_filters,
                filter_size=self.filter_size,
                stride=self.stride,
                name=self.name,
                pad=self.pad,
                W=W,
                nonlinearity=nonlinearity,
                b=b,
            )

            if self.batch_norm and not self.preactivate:
                outgoing = BatchNormLayer2(
                    outgoing, nonlinearity=self.nonlinearity, name=self.name + '/bn_'
                )

            if self.pool:
                outgoing = MaxPool2DLayer(
                    outgoing,
                    pool_size=self.pool_size,
                    name=self.name + '/P',
                    stride=self.pool_stride,
                )
            return outgoing

    @register_bundle
    class ResidualBundle(Bundle):
        def __init__(
            self,
            num_filters,
            filter_size=filter_size,
            stride=stride,
            nonlinearity=nonlinearity,
            pool_size=pool_size,
            W=W,
            pool_stride=pool_stride,
            dropout=None,
            pool=False,
            preactivate=True,
            postactivate=False,
        ):
            self.num_filters = num_filters
            self.filter_size = filter_size
            self.stride = stride
            self.nonlinearity = nonlinearity
            self.pool_size = pool_size
            self.pool_stride = pool_stride
            self.dropout = dropout
            self.pool = pool
            self.W = W
            self.preactivate = preactivate
            self.postactivate = postactivate
            super(ResidualBundle, self).__init__()
            self.name = 'R' + self.name

        # def projectionA(l_inp):
        #    n_filters = l_inp.output_shape[1] * 2

        #    def ceildiv(a, b):
        #        return -(-a // b)

        #    l = layers.ExpressionLayer(
        #        l_inp,
        #        lambda X: X[:, :, ::2, ::2],
        #        lambda s: (s[0], s[1], ceildiv(s[2], 2), ceildiv(s[3], 2)))
        #    l = layers.PadLayer(l, [n_filters // 4, 0, 0], batch_ndim=1)
        #    return l

        def projectionB(self, incoming):
            """
            The projection shortcut in Eqn.(2) is used to match dimensions
            (done by 1x1 convolutions). When the shortcuts go across feature
            maps of two sizes, they are performed with a stride of 2.
            """
            # Projection is a strided 1x1 convolution.
            # I think preactivation should not trigger any nonlinearities. Just
            # batch normalization. But I haven't been able to confirm.
            projector = ConvBundle(
                filter_size=(1, 1),
                num_filters=self.num_filters,
                stride=self.stride,
                W=self.W,
                pad='same',
                dropout=self.dropout,
                name=self.name + '/proj',
                nonlinearity=None,
                preactivate=self.preactivate,
            )
            shortcut = projector(incoming)
            return shortcut

        def __call__(self, incoming):
            """
            https://github.com/Lasagne/Lasagne/issues/531
            https://github.com/alrojo/lasagne_residual_network/search?utf8=%E2%9C%93&q=residual
            https://github.com/FlorianMuellerklein/Identity-Mapping-ResNet-Lasagne/blob/master/models.py
            """
            # Check if this bundle is going to reduce the spatial dimensions
            size_reduced = np.prod(self.stride) != 1

            # Define convolvers
            convkw = dict(
                W=self.W,
                pad='same',
                dropout=self.dropout,
                batch_norm=False,
                filter_size=self.filter_size,
                num_filters=self.num_filters,
            )

            # Do not preactivate if this is the first layer in the network
            convolver1 = ConvBundle(
                stride=self.stride,
                preactivate=self.preactivate,
                name=self.name + '/C1',
                **convkw
            )
            convolver2 = ConvBundle(
                stride=(1, 1), preactivate=True, name=self.name + '/C2', **convkw
            )

            branch = incoming
            branch = convolver1(branch)
            branch = convolver2(branch)
            branch._is_main_layer = False

            # Need to project the shortcut branch
            if size_reduced:
                shortcut = self.projectionB(incoming)
            else:
                shortcut = incoming

            outgoing = layers.ElemwiseSumLayer(
                [branch, shortcut], name=self.name + '/sum'
            )

            # Postactivate if this is the last residual layer.
            if self.postactivate:
                outgoing = BatchNormLayer2(
                    outgoing, name=self.name + '/bn_', nonlinearity=nonlinearity
                )

            if self.pool:
                outgoing = MaxPool2DLayer(
                    outgoing,
                    pool_size=self.pool_size,
                    name=self.name + '/P',
                    stride=self.pool_stride,
                )

            return outgoing

    @register_bundle
    class InceptionBundle(Bundle):
        # https://github.com/317070/lasagne-googlenet/blob/master/googlenet.py

        def __init__(
            self,
            branches=branches,
            nonlinearity=nonlinearity,
            batch_norm=batch_norm,
            dropout=None,
            pool=False,
            pool_size=pool_size,
            pool_stride=pool_stride,
            W=W,
        ):
            # standard
            self.branches = branches
            self.nonlinearity = nonlinearity
            self.dropout = dropout
            self.batch_norm = batch_norm
            self.pool = pool
            self.pool_size = pool_size
            self.W = W
            self.pool_stride = pool_stride
            super(InceptionBundle, self).__init__()
            self.name = 'INCEP' + self.name

        def __call__(self, incoming):
            in_ = incoming
            if self.dropout is not None and self.dropout > 0:
                in_ = self.apply_dropout(in_)
            name = self.name

            # branches = self.inception_v0(in_)
            if self.branches is not None:
                branches = []
                for b in self.branches:
                    logger.info(b)
                    if b['t'] == 'c':
                        branch = self.conv_branch(
                            in_, b['s'], b['n'], b['r'], b.get('d', 1)
                        )
                    elif b['t'] == 'p':
                        branch = self.proj_branch(in_, b['s'], b['n'])
                    else:
                        logger.info('b = %r' % (b,))
                        assert False
                    branches.append(branch)
            else:
                # branches = self.inception_v3_A(in_)
                branches = self.inception_v0(in_)

            # logger.info(branches)
            # for b in branches:
            #    logger.info(b.output_shape)

            outgoing = layers.ConcatLayer(branches, name=name + '/cat')
            outgoing._is_main_layer = True
            if self.pool:
                outgoing = MaxPool2DLayer(
                    outgoing,
                    pool_size=self.pool_size,
                    name='P' + self.name,
                    stride=self.pool_stride,
                )
            return outgoing

        def conv_branch(self, incoming, filter_size, num_filters, num_reduce, depth=1):
            name = self.name
            name_aug = 'x'.join([str(s) for s in filter_size])
            if num_reduce > 0:
                if False:
                    bias = 0.1
                    redu = layers.NINLayer(
                        incoming,
                        num_units=num_reduce,
                        W=self.W,
                        b=init.Constant(bias),
                        nonlinearity=self.nonlinearity,
                        name=name + '/' + name_aug + '_reduce',
                    )

                else:
                    redu = Conv2DLayer(
                        incoming,
                        num_filters=num_reduce,
                        filter_size=(1, 1),
                        pad=0,
                        stride=(1, 1),
                        nonlinearity=self.nonlinearity,
                        W=self.W,
                        name=name + '/' + name_aug + '_reduce',
                    )
                if self.batch_norm:
                    redu = self.apply_batch_norm(redu)
            else:
                redu = incoming
            conv = redu
            for d in range(depth):
                pad = min(filter_size) // 2
                conv = Conv2DLayer(
                    conv,
                    num_filters=num_filters,
                    filter_size=filter_size,
                    pad=pad,
                    stride=(1, 1),
                    nonlinearity=self.nonlinearity,
                    W=self.W,
                    name=name + '/' + name_aug + '_' + str(d),
                )
                if d > 0:
                    conv._is_main_layer = False
                if self.batch_norm:
                    conv = self.apply_batch_norm(conv)
            # if num_reduce > 0:
            #    conv._is_main_layer = False
            return conv

        def proj_branch(self, incoming, pool_size, num_proj):
            name = self.name
            flipped = FlipLayer(incoming, name=name + '/Flip')
            pool = MaxPool2DLayer(
                flipped,
                pool_size=pool_size,
                stride=(1, 1),
                pad=(1, 1),
                name=name + '/pool',
            )
            unflipped = FlipLayer(pool, name=name + '/Unflip')

            project = Conv2DLayer(
                unflipped,
                num_filters=num_proj,
                filter_size=(1, 1),
                pad=0,
                stride=(1, 1),
                nonlinearity=self.nonlinearity,
                name=name + '/proj',
            )
            if self.batch_norm:
                project = self.apply_batch_norm(project)
            return project

        def inception_v0(self, in_):
            # Define the 3 convolutional branches
            conv_branches = [
                self.conv_branch(in_, (1, 1), 64, 0),
                self.conv_branch(in_, (3, 3), 128, 96),
                self.conv_branch(in_, (5, 5), 16, 16),
            ]
            # Define the projection branch
            proj_branches = [self.proj_branch(in_, (3, 3), 32)]
            branches = conv_branches + proj_branches
            return branches

        def inception_v3_A(self, in_):
            conv_branches = [
                self.conv_branch(in_, (1, 1), 64, 0),
                self.conv_branch(in_, (3, 3), 128, 96),
                self.conv_branch(in_, (3, 3), 16, 16, depth=2),
            ]
            proj_branches = [self.proj_branch(in_, (3, 3), 32)]
            branches = conv_branches + proj_branches
            return branches

    @register_bundle
    class DenseBundle(Bundle):
        def __init__(
            self,
            num_units,
            batch_norm=batch_norm,
            nonlinearity=nonlinearity,
            W=W,
            dropout=None,
        ):
            self.num_units = num_units
            self.batch_norm = batch_norm
            self.nonlinearity = nonlinearity
            self.dropout = dropout
            self.W = W
            super(DenseBundle, self).__init__()

        def __call__(self, incoming):
            outgoing = incoming
            if self.dropout is not None and self.dropout > 0:
                outgoing = self.apply_dropout(outgoing)
            outgoing = layers.DenseLayer(
                outgoing,
                num_units=self.num_units,
                name='F' + self.name,
                nonlinearity=self.nonlinearity,
                W=self.W,
            )
            if self.batch_norm:
                outgoing = self.apply_batch_norm(outgoing)
            return outgoing

    @register_bundle
    class SoftmaxBundle(Bundle):
        def __init__(self, num_units, dropout=None, W=W):
            self.num_units = num_units
            self.batch_norm = batch_norm
            self.dropout = dropout
            self.W = W
            super(SoftmaxBundle, self).__init__()

        def __call__(self, incoming):
            outgoing = incoming
            if self.dropout is not None and self.dropout > 0:
                outgoing = self.apply_dropout(outgoing)
            outgoing = layers.DenseLayer(
                outgoing,
                num_units=self.num_units,
                name='F' + self.name,
                W=self.W,
                nonlinearity=nonlinearities.softmax,
            )
            return outgoing

    @register_bundle
    class NonlinearitySoftmax(Bundle):
        def __call__(self, incoming):
            return layers.NonlinearityLayer(
                incoming,
                nonlinearity=nonlinearities.softmax,
                name='Softmax' + self.name,
            )

    @register_bundle
    class GlobalPool(Bundle):
        def __call__(
            self,
            incoming,
        ):
            outgoing = layers.GlobalPoolLayer(
                incoming,
                name='GP' + self.name,
            )
            outgoing._is_main_layer = True
            return outgoing

    @register_bundle
    class AveragePool(Bundle):
        def __call__(
            self,
            incoming,
        ):
            outgoing = layers.GlobalPoolLayer(
                incoming,
                name='AP' + self.name,
            )
            outgoing._is_main_layer = True
            return outgoing

    @register_bundle
    class MaxPool2D(Bundle):
        def __init__(self, pool_size=pool_size, pool_stride=pool_stride):
            self.pool_size = pool_size
            self.pool_stride = pool_stride
            super(MaxPool2D, self).__init__()

        def __call__(self, incoming):
            return MaxPool2DLayer(
                incoming,
                pool_size=self.pool_size,
                stride=self.pool_stride,
                name='P' + self.name,
            )

    # def inception_module(l_in, num_1x1, reduce_3x3, num_3x3, reduce_5x5,
    #                     num_5x5, gain=1.0, bias=0.1):
    #    """
    #    inception module (without the 3x3x1 pooling and projection because
    #    that's difficult in Theano right now)

    #    http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Szegedy_Going_Deeper_With_2015_CVPR_paper.pdf
    #    """
    return bundles


if __name__ == '__main__':
    """
    CommandLine:
        python -m wbia_cnn.custom_layers
        python -m wbia_cnn.custom_layers --allexamples
        python -m wbia_cnn.custom_layers --allexamples --noface --nosrc
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
