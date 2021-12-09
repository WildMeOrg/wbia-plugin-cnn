# -*- coding: utf-8 -*-
import logging
import warnings
import six
import functools
import utool as ut

print, rrr, profile = ut.inject2(__name__)
logger = logging.getLogger()


def make_layer_str(layer, with_name=True):
    r"""
    Args:
        layer (lasagne.Layer): a network layer
    """
    # layer_dict = make_layer_json_dict(layer, extra=with_name)
    # layer_str = layer_dict['type']

    layer_info = get_layer_info(layer)
    layer_type = layer_info['classname']
    layer_attrs = layer_info['layer_attrs']
    attr_str_list = ['%s=%s' % item for item in layer_attrs.items()]

    # attr_key_list = layer_info['layer_attrs']
    # attr_val_list = [getattr(layer, attr) for attr in attr_key_list]
    # attr_str_list = ['%s=%r' % item for item in zip(attr_key_list, attr_val_list)]

    if with_name:
        layer_name = getattr(layer, 'name', None)
        if layer_name is not None:
            attr_str_list = ['name=%s' % (layer_name,)] + attr_str_list

    if hasattr(layer, 'nonlinearity'):
        try:
            nonlinearity = layer.nonlinearity.__name__
        except AttributeError:
            nonlinearity = layer.nonlinearity.__class__.__name__
        attr_str_list.append('nonlinearity={0}'.format(nonlinearity))
    attr_str = ','.join(attr_str_list)

    layer_str = layer_type + '(' + attr_str + ')'
    return layer_str


def make_layers_json(layer_list, extra=True):
    # Make persistant ids
    layer_to_id = {layer: count for count, layer in enumerate(layer_list)}

    layer_info_list = [get_layer_info(layer) for layer in layer_list]

    layer_json_list = []
    for layer_info, layer in zip(layer_info_list, layer_list):
        json_dict = make_layer_json_dict(layer, layer_info, layer_to_id, extra=extra)
        layer_json_list.append(json_dict)
    return layer_json_list


def make_layer_json_dict(layer, layer_info, layer_to_id, extra=True):
    """
    >>> from wbia_cnn.net_strs import *  # NOQA
    """
    # layer_type = layer_info['classname']
    # attr_key_list = layer_info['layer_attrs']
    json_dict = ut.odict([])
    if extra:
        json_dict['name'] = layer_info['name']

    json_dict['type'] = layer_info['classname']
    json_dict['id'] = layer_to_id[layer]

    if hasattr(layer, 'input_layer'):
        # json_dict['input_layer'] = layer.input_layer.name
        # HACK FOR WHEN DROPOUT LAYERS DONT EXIST
        def get_mrp_id(in_layer):
            if in_layer in layer_to_id:
                return layer_to_id[in_layer]
            else:
                return get_mrp_id(in_layer.input_layer)

        json_dict['input_layer'] = get_mrp_id(layer.input_layer)
    if hasattr(layer, 'input_layers'):
        # json_dict['input_layers'] = [l.name for l in layer.input_layers]
        json_dict['input_layers'] = [layer_to_id[layer] for layer in layer.input_layers]

    json_dict.update(**layer_info['layer_attrs'])
    nonlin = layer_info.get('nonlinearity', None)
    if nonlin is not None:
        json_dict['nonlinearity'] = nonlin

    json_params = []
    for param_info in layer_info['param_infos']:
        p = ut.odict()
        p['name'] = param_info['basename']
        if extra:
            init = param_info.get('init', None)
            if init is not None:
                p['init'] = init
        tags = param_info.get('tags', None)
        if tags is not None:
            if extra:
                p['tags'] = list(tags)
            else:
                if len(tags) > 0:
                    p['tags'] = list(tags)
        json_params.append(p)
    if len(json_params) > 0:
        json_dict['params'] = json_params
    return json_dict


def print_pretrained_weights(pretrained_weights, lbl=''):
    r"""
    Args:
        pretrained_weights (list of ndarrays): represents layer weights
        lbl (str): label
    """
    logger.info('Initialization network: %r' % (lbl))
    logger.info('Total memory: %s' % (ut.get_object_size_str(pretrained_weights)))
    for index, layer_ in enumerate(pretrained_weights):
        logger.info(
            ' layer {:2}: shape={:<18}, memory={}'.format(
                index, layer_.shape, ut.get_object_size_str(layer_)
            )
        )


def count_bytes(output_layer):
    import lasagne

    layers = lasagne.layers.get_all_layers(output_layer)
    info_list = [get_layer_info(layer) for layer in layers]
    total_bytes = sum([info['total_bytes'] for info in info_list])
    # logger.info('total_bytes = %s' % (ut.byte_str2(total_bytes),))
    # logger.info(ut.repr2(info_list, nl=2, hack_liststr=True))
    return total_bytes


def surround(str_, b='{}'):
    return b[0] + str_ + b[1]


def tagstr(tags):
    # return surround(','.join([t[0] for t in tags]), '{}')
    return ','.join([t[0] for t in tags])


def shapestr(shape):
    return ','.join(map(str, shape))


def param_basename(layer, p):
    pname = p.name
    if layer.name is not None:
        if pname.startswith(layer.name + '.'):
            pname = pname[len(layer.name) + 1 :]
    return pname


def paramstr(layer, p, tags):
    inner_parts = [shapestr(p.get_value().shape)]
    pname = param_basename(layer, p)
    if tags:
        inner_parts.append(tagstr(tags))
    return pname + surround('; '.join(inner_parts), '()')


def get_layer_info(layer):
    r"""
    Args:
        layer (?):

    Returns:
        ?: layer_info

    CommandLine:
        python -m wbia_cnn.net_strs get_layer_info --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia_cnn.net_strs import *  # NOQA
        >>> from wbia_cnn import models
        >>> model = models.mnist.MNISTModel(batch_size=8, data_shape=(28, 28, 1), output_dims=10)
        >>> model.init_arch()
        >>> nn_layers = model.get_all_layers()
        >>> for layer in nn_layers:
        >>>     layer_info = get_layer_info(layer)
        >>>     print(ut.repr3(layer_info, nl=1))
    """
    import operator
    import lasagne

    # Information that contributes to RAM usage
    import numpy as np

    # Get basic layer infos
    output_shape = lasagne.layers.get_output_shape(layer)
    input_shape = getattr(layer, 'input_shape', [])
    # Get number of outputs ignoring the batch size
    num_outputs = functools.reduce(operator.mul, output_shape[1:])
    if len(input_shape):
        num_inputs = functools.reduce(operator.mul, input_shape[1:])
    else:
        num_inputs = 0
    # TODO: if we can ever support non float32 calculations this must change
    # layer_type = 'float32'
    layer_dtype = np.dtype('float32')

    # Get individual param infos
    param_infos = []
    for param, tags in layer.params.items():
        value = param.get_value()
        pbasename = param_basename(layer, param)
        param_info = ut.odict(
            [
                ('name', param.name),
                ('basename', pbasename),
                ('tags', tags),
                ('shape', value.shape),
                ('size', value.size),
                ('itemsize', value.dtype.itemsize),
                ('dtype', str(value.dtype)),
                ('bytes', value.size * value.dtype.itemsize),
            ]
        )

        def initializer_info(initclass):
            initclassname = initclass.__class__.__name__
            if initclassname == 'Constant':
                spec = initclass.val
            else:
                spec = ut.odict()
                spec['type'] = initclassname
                for key, val in initclass.__dict__.items():
                    if isinstance(val, lasagne.init.Initializer):
                        spec[key] = initializer_info(val)
                    elif isinstance(val, type) and issubclass(
                        val, lasagne.init.Initializer
                    ):
                        spec[key] = val.__name__
                        # initializer_info(val())
                    else:
                        spec[key] = val
            return spec

        if hasattr(layer, '_initializers'):
            # logger.info('layer = %r' % (layer,))
            initclass = layer._initializers[param]
            spec = initializer_info(initclass)
            param_info['init'] = spec

        param_infos.append(param_info)
    # Combine param infos
    param_str = surround(
        ', '.join([paramstr(layer, p, tags) for p, tags in layer.params.items()]), '[]'
    )
    param_type_str = surround(
        ', '.join([repr(p.type) for p, tags in layer.params.items()]), '[]'
    )
    num_params = sum([info['size'] for info in param_infos])

    classalias_map = {
        'ElemwiseSumLayer': 'ElemwiseSum',
        'Conv2DCCLayer': 'Conv2D',
        'Conv2DDNNLayer': 'Conv2D',
        'Conv2DLayer': 'Conv2D',
        'MaxPool2DLayer': 'MaxPool2D',
        'MaxPool2DCCLayer': 'MaxPool2D',
        'MaxPool2DDNNLayer': 'MaxPool2D',
        'LeakyRectify': 'LReLU',
        'InputLayer': 'Input',
        'GaussianNoiseLayer': 'Noise',
        'DropoutLayer': 'Dropout',
        'DenseLayer': 'Dense',
        'NonlinearityLayer': 'Nonlinearity',
        'FlattenLayer': 'Flatten',
        'L2NormalizeLayer': 'L2Norm',
        'BatchNormLayer': 'BatchNorm',
        'BatchNormLayer2': 'BatchNorm',
    }
    layer_attrs_ignore_dict = {
        'MaxPool2D': ['mode', 'ignore_border'],
        'Dropout': ['rescale'],
        'Conv2D': ['convolution'],
        'BatchNorm': ['epsilon', 'mean', 'inv_std', 'axes', 'beta', 'gamma'],
        'BatchNorm2': ['epsilon', 'mean', 'inv_std', 'axes', 'beta', 'gamma'],
        #'ElemwiseSum': ['merge_function', 'cropping'],
        #'ElemwiseSum': [],
        'FeaturePoolLayer': ['axis'],
    }
    layer_attrs_dict = {
        #'ElemwiseSum': ['coeffs'],
        #'ElemwiseSum': ['coeffs', 'merge_function', 'cropping'],
        'Noise': ['sigma'],
        'Input': ['shape'],
        'Dropout': ['p', 'shared_axes'],
        'Conv2D': ['num_filters', 'filter_size', 'stride', 'output_shape', 'num_groups'],
        'MaxPool2D': ['stride', 'pool_size', 'output_shape'],  # 'mode'],
        'Dense': ['num_units', 'num_leading_axes'],
        'SoftMax': ['num_units', 'num_leading_axes'],
        'L2Norm': ['axis'],
        'BatchNorm': ['alpha'],
        'BatchNorm2': ['alpha'],
        'FeaturePoolLayer': ['pool_size', 'pool_function'],
    }
    # layer_attrs_dict = {}
    all_ignore_attrs = [
        'nonlinearity',
        'b',
        'W',
        'get_output_kwargs',
        'name',
        'input_shapes',
        'input_layers',
        'input_shape',
        'input_layer',
        'input_var',
        'untie_biases',
        '_initializers',
        'flip_filters',
        'pad',
        'params',
        'n',
        '_is_main_layer',
    ]

    classname = layer.__class__.__name__
    classalias = classalias_map.get(classname, classname)
    # if classalias == 'FeaturePoolLayer' and ut.get_funcname(layer.pool_function) == 'max':
    #    classalias = 'MaxOut'
    if classalias == 'Dense' and ut.get_funcname(layer.nonlinearity) == 'softmax':
        classalias = 'SoftMax'

    layer_attrs = ut.odict(
        [(key, getattr(layer, key)) for key in layer_attrs_dict.get(classalias, [])]
    )
    ignore_attrs = all_ignore_attrs + layer_attrs_ignore_dict.get(classalias, [])

    if classalias not in layer_attrs_dict or (
        classalias == classname and len(layer_attrs) == 0
    ):
        layer_attrs = layer.__dict__.copy()
        ut.delete_dict_keys(layer_attrs, ignore_attrs)

    for key in list(layer_attrs.keys()):
        val = layer_attrs[key]
        if ut.is_funclike(val):
            layer_attrs[key] = ut.get_funcname(val)

    attr_key_list = list(layer_attrs.keys())
    missing_keys = set(layer.__dict__.keys()) - set(ignore_attrs) - set(attr_key_list)
    missing_keys = [k for k in missing_keys if not k.startswith('_')]

    # if layer_type == 'Conv2DCCLayer':
    #    ut.embed()
    DEBUG = True
    if DEBUG and len(missing_keys) > 0:
        logger.info('---')
        logger.info(' * ' + classname)
        logger.info(' * missing keys: %r' % (missing_keys,))
        logger.info(' * has keys: %r' % (attr_key_list,))
        if True:
            # import utool
            # with utool.embed_on_exception_context:
            # raise AssertionError('MISSING KEYS')
            pass

    # handle None batch sizes
    if output_shape[0] is None:
        size = np.prod(output_shape[1:])
    else:
        size = np.prod(output_shape)

    layer_info = ut.odict(
        [
            ('name', layer.name),
            ('classname', classname),
            ('classalias', classalias),
            ('output_shape', output_shape),
            ('input_shape', input_shape),
            ('num_outputs', num_outputs),
            ('num_inputs', num_inputs),
            ('size', size),
            ('itemsize', layer_dtype.itemsize),
            ('dtype', str(layer_dtype)),
            ('num_params', num_params),
            ('param_infos', param_infos),
            ('param_str', param_str),
            ('param_type_str', param_type_str),
            ('layer_attrs', layer_attrs),
            ('nonlinearity', None),
        ]
    )

    if hasattr(layer, 'nonlinearity'):
        try:
            nonlinearity = layer.nonlinearity.__name__
        except AttributeError:
            nonlinearity = layer.nonlinearity.__class__.__name__
        layer_info['nonlinearity'] = ut.odict([])
        layer_info['nonlinearity']['type'] = nonlinearity
        layer_info['nonlinearity'].update(layer.nonlinearity.__dict__)
        # attr_str_list.append('nonlinearity={0}'.format(nonlinearity))

    param_bytes = sum([info['bytes'] for info in param_infos])
    layer_bytes = layer_info['size'] * layer_info['itemsize']
    # if classname in ['BatchNormLayer', 'NonlinearityLayer']:
    #    layer_bytes = 0
    layer_info['bytes'] = layer_bytes
    layer_info['param_bytes'] = param_bytes
    layer_info['total_bytes'] = layer_bytes + param_bytes
    layer_info['total_memory'] = ut.byte_str2(layer_info['total_bytes'])
    return layer_info


def get_layer_info_str(output_layer, batch_size=128):
    r"""
    Args:
        output_layer (lasagne.layers.Layer):

    CommandLine:
        python -m wbia_cnn.net_strs --test-get_layer_info_str:0
        python -m wbia_cnn.net_strs --test-get_layer_info_str:1

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia_cnn.net_strs import *  # NOQA
        >>> from wbia_cnn import models
        >>> model = models.DummyModel(data_shape=(28, 28, 3), autoinit=True)
        >>> output_layer = model.output_layer
        >>> result = get_layer_info_str(output_layer)
        >>> result = '\n'.join([x.rstrip() for x in result.split('\n')])
        >>> print(result)
        Network Structure:
         index  Layer    Outputs    Bytes OutShape         Params
         0      Input      1,728   55,296 (8, 3, 24, 24)   []
         1      Conv2D     7,744  249,600 (8, 16, 22, 22)  [W(16,3,3,3, {t,r}), b(16, {t})]
         2      Conv2D     7,056  229,952 (8, 16, 21, 21)  [W(16,16,2,2, {t,r}), b(16, {t})]
         3      Dense          8  226,080 (8, 8)           [W(7056,8, {t,r}), b(8, {t})]
         4      Dense          5      340 (8, 5)           [W(8,5, {t,r}), b(5, {t})]
        ...this model has 57,989 learnable parameters
        ...this model will use 761,268 bytes = 743.43 KB

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia_cnn.net_strs import *  # NOQA
        >>> from wbia_cnn import models
        >>> model = models.mnist.MNISTModel(batch_size=128, output_dims=10,
        >>>                                 data_shape=(28, 28, 3))
        >>> model.init_arch()
        >>> output_layer = model.output_layer
        >>> result = get_layer_info_str(output_layer)
        >>> result = '\n'.join([x.rstrip() for x in result.split('\n')])
        >>> print(result)
    """
    import lasagne

    info_lines = []
    _print = info_lines.append
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', '.*topo.*')
        nn_layers = lasagne.layers.get_all_layers(output_layer)
        _print('Network Structure:')

        columns_ = ut.ddict(list)

        for index, layer in enumerate(nn_layers):

            layer_info = get_layer_info(layer)

            columns_['index'].append(index)
            columns_['name'].append(layer_info['name'])
            # columns_['type'].append(getattr(layer, 'type', None))
            # columns_['layer'].append(layer_info['classname'])
            columns_['layer'].append(layer_info['classalias'])
            columns_['num_outputs'].append('{:,}'.format(int(layer_info['num_outputs'])))
            columns_['output_shape'].append(str(layer_info['output_shape']))
            columns_['params'].append(layer_info['param_str'])
            columns_['param_type'].append(layer_info['param_type_str'])
            columns_['mem'].append(layer_info['total_memory'])
            columns_['bytes'].append('{:,}'.format(int(layer_info['total_bytes'])))
            # ut.embed()

        header_nice = {
            'index': 'index',
            'name': 'Name',
            'layer': 'Layer',
            'type': 'Type',
            'num_outputs': 'Outputs',
            'output_shape': 'OutShape',
            'params': 'Params',
            'param_type': 'ParamType',
            'mem': 'Mem',
            'bytes': 'Bytes',
        }

        header_align = {
            'index': '<',
            'params': '<',
            'bytes': '>',
            'num_outputs': '>',
        }

        def get_col_maxval(key):
            header_len = len(header_nice[key])
            val_len = max(list(map(len, map(str, columns_[key]))))
            return max(val_len, header_len)

        header_order = ['index']
        if len(ut.filter_Nones(columns_['name'])) > 0:
            header_order += ['name']
        header_order += ['layer', 'num_outputs']
        # header_order += ['mem']
        header_order += ['bytes']
        header_order += ['output_shape', 'params']
        #'param_type']

        max_len = {
            key: str(get_col_maxval(key) + 1) for key, col in six.iteritems(columns_)
        }

        fmtstr = ' ' + ' '.join(
            [
                '{:' + align + len_ + '}'
                for align, len_ in zip(
                    ut.dict_take(header_align, header_order, '<'),
                    ut.dict_take(max_len, header_order),
                )
            ]
        )
        _print(fmtstr.format(*ut.dict_take(header_nice, header_order)))

        row_list = zip(*ut.dict_take(columns_, header_order))
        for row in row_list:
            try:
                row = ['' if _ is None else _ for _ in row]
                str_ = fmtstr.format(*row)
                _print(str_)
            except TypeError:
                logger.info(
                    'Error printing %r with args %r'
                    % (
                        fmtstr,
                        row,
                    )
                )

        total_bytes = count_bytes(output_layer)
        num_params = lasagne.layers.count_params(output_layer)

        _print('...this model has {:,} learnable parameters'.format(num_params))
        _print(
            '...this model will use ~{:,} bytes = {} per input'.format(
                total_bytes, ut.byte_str2(total_bytes)
            )
        )
        _print(
            '...this model will use ~{:,} bytes = {} per batch with a batch size of {}'.format(
                total_bytes * batch_size,
                ut.byte_str2(total_bytes * batch_size),
                batch_size,
            )
        )
    info_str = '\n'.join(info_lines)
    return info_str


def print_layer_info(output_layer):
    str_ = get_layer_info_str(output_layer)
    str_ = ut.indent('[info] ' + str_)
    logger.info('\n' + str_)


if __name__ == '__main__':
    """
    CommandLine:
        python -m wbia_cnn.net_strs
        python -m wbia_cnn.net_strs --allexamples
        python -m wbia_cnn.net_strs --allexamples --noface --nosrc
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
