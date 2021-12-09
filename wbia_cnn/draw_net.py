# -*- coding: utf-8 -*-
"""
Functions to create network diagrams from a list of Layers.

References:
    # Adapted from
    https://github.com/ebenolson/Lasagne/blob/master/examples/draw_net.py

    # TODO:
    https://github.com/dnouri/nolearn/blob/master/nolearn/lasagne/visualize.py
"""
import logging
from operator import itemgetter
from os.path import join, exists
import numpy as np
import utool as ut
from wbia_cnn import utils

print, rrr, profile = ut.inject2(__name__)
logger = logging.getLogger()


def imwrite_theano_symbolic_graph(thean_expr):
    import theano

    graph_dpath = '.'
    graph_fname = 'symbolic_graph.png'
    graph_fpath = ut.unixjoin(graph_dpath, graph_fname)
    ut.ensuredir(graph_dpath)
    theano.printing.pydotprint(thean_expr, outfile=graph_fpath, var_with_name_simple=True)
    ut.startfile(graph_fpath)
    return graph_fpath


def draw_neural_net(ax, left, right, bottom, top, layer_sizes):
    """

    References:
        # Taken from
        https://gist.github.com/craffel/2d727968c3aaebd10359

    Draw a neural network cartoon using matplotilb.

    Example:
        >>> import matplotlib.pyplot as plt
        >>> fig = plt.figure(figsize=(12, 12))
        >>> draw_neural_net(fig.gca(), .1, .9, .1, .9, [4, 7, 2])

    :parameters:
        - ax : matplotlib.axes.AxesSubplot
            The axes on which to plot the cartoon (get e.g. by plt.gca())
        - left : float
            The center of the leftmost node(s) will be placed here
        - right : float
            The center of the rightmost node(s) will be placed here
        - bottom : float
            The center of the bottommost node(s) will be placed here
        - top : float
            The center of the topmost node(s) will be placed here
        - layer_sizes : list of int
            List of layer sizes, including input and output dimensionality
    """
    import matplotlib.pyplot as plt

    # n_layers = len(layer_sizes)
    v_spacing = (top - bottom) / float(max(layer_sizes))
    h_spacing = (right - left) / float(len(layer_sizes) - 1)
    # Nodes
    for n, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing * (layer_size - 1) / 2.0 + (top + bottom) / 2.0
        for m in range(layer_size):
            circle = plt.Circle(
                (n * h_spacing + left, layer_top - m * v_spacing),
                v_spacing / 4.0,
                color='w',
                ec='k',
                zorder=4,
            )
            ax.add_artist(circle)
    # Edges
    for n, (layer_size_a, layer_size_b) in enumerate(
        zip(layer_sizes[:-1], layer_sizes[1:])
    ):
        layer_top_a = v_spacing * (layer_size_a - 1) / 2.0 + (top + bottom) / 2.0
        layer_top_b = v_spacing * (layer_size_b - 1) / 2.0 + (top + bottom) / 2.0
        for m in range(layer_size_a):
            for o in range(layer_size_b):
                line = plt.Line2D(
                    [n * h_spacing + left, (n + 1) * h_spacing + left],
                    [layer_top_a - m * v_spacing, layer_top_b - o * v_spacing],
                    c='k',
                )
                ax.add_artist(line)


def show_arch_nx_graph(layers, fnum=None, fullinfo=True):
    r"""

    CommandLine:
        python -m wbia_cnn.draw_net show_arch_nx_graph:0 --show
        python -m wbia_cnn.draw_net show_arch_nx_graph:1 --show

    Example0:
        >>> # ENABLE_DOCTEST
        >>> from wbia_cnn.draw_net import *  # NOQA
        >>> from wbia_cnn import models
        >>> model = models.mnist.MNISTModel(batch_size=128, output_dims=10,
        >>>                                 data_shape=(28, 28, 3))
        >>> model.init_arch()
        >>> layers = model.get_all_layers()
        >>> show_arch_nx_graph(layers)
        >>> ut.quit_if_noshow()
        >>> from wbia import plottool as pt
        >>> ut.show_if_requested()

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from wbia_cnn.draw_net import *  # NOQA
        >>> from wbia_cnn import models
        >>> model = models.SiameseCenterSurroundModel(autoinit=True)
        >>> layers = model.get_all_layers()
        >>> show_arch_nx_graph(layers)
        >>> ut.quit_if_noshow()
        >>> from wbia import plottool as pt
        >>> ut.show_if_requested()

    """
    import networkx as nx
    from wbia import plottool as pt
    import lasagne

    # from matplotlib import offsetbox
    # import matplotlib as mpl

    REMOVE_BATCH_SIZE = True
    from wbia_cnn import net_strs

    def get_hex_color(layer_type):
        if 'Input' in layer_type:
            return '#A2CECE'
        if 'Conv2D' in layer_type:
            return '#7C9ABB'
        if 'Dense' in layer_type:
            return '#6CCF8D'
        if 'Pool' in layer_type:
            return '#9D9DD2'
        if 'SoftMax' in layer_type:
            return '#7E9FD9'
        else:
            return '#{0:x}'.format(hash(layer_type + 'salt') % 2 ** 24)

    node_dict = {}
    edge_list = []
    edge_attrs = ut.ddict(dict)

    # Make layer ids (ensure no duplicates)
    layer_to_id = {
        layer: repr(layer) if layer.name is None else layer.name for layer in set(layers)
    }
    keys_ = layer_to_id.keys()
    dups = ut.find_duplicate_items(layer_to_id.values())
    for dupval, dupidxs in dups.items():
        newval_fmt = dupval + '_%d'
        for layer in ut.take(keys_, dupidxs):
            newid = ut.get_nonconflicting_string(newval_fmt, layer_to_id.values())
            layer_to_id[layer] = newid

    def layerid(layer):
        return layer_to_id[layer]

    main_nodes = []

    for i, layer in enumerate(layers):
        layer_info = net_strs.get_layer_info(layer)
        layer_type = layer_info['classalias']

        key = layerid(layer)

        color = get_hex_color(layer_info['classalias'])
        # Make label
        lines = []
        if layer_info['name'] is not None:
            lines.append(layer_info['name'])
        if fullinfo:
            lines.append(layer_info['classalias'])
            for attr, val in layer_info['layer_attrs'].items():
                if attr == 'shape' and REMOVE_BATCH_SIZE:
                    val = val[1:]
                if attr == 'output_shape' and REMOVE_BATCH_SIZE:
                    val = val[1:]
                lines.append('{0}: {1}'.format(attr, val))

            nonlinearity = layer_info.get('nonlinearity')
            if nonlinearity is not None:
                alias_map = {
                    'LeakyRectify': 'LReLU',
                }
                val = layer_info['nonlinearity']['type']
                val = alias_map.get(val, val)
                lines.append('nonlinearity:\n{0}'.format(val))

        label = '\n'.join(lines)

        # append node
        is_main_layer = len(layer.params) > 0
        # is_main_layer = len(lasagne.layers.get_all_params(layer, trainable=True)) > 0
        if layer_info['classname'] in lasagne.layers.normalization.__all__:
            is_main_layer = False
        if layer_info['classname'] in lasagne.layers.special.__all__:
            is_main_layer = False
        if layer_info['classname'].startswith('BatchNorm'):
            is_main_layer = False
        if layer_info['classname'].startswith('ElemwiseSum'):
            is_main_layer = True

        if layer_type == 'Input':
            is_main_layer = True

        if hasattr(layer, '_is_main_layer'):
            is_main_layer = layer._is_main_layer

        # if getattr(layer, 'name', '') is not None and getattr(layer, 'name', '') .endswith('/sum'):
        #    is_main_layer = True

        node_attr = dict(
            name=key,
            label=label,
            color=color,
            fillcolor=color,
            style='filled',
            is_main_layer=is_main_layer,
        )

        node_attr['is_main_layer'] = is_main_layer
        if is_main_layer:
            main_nodes.append(key)
        node_attr['classalias'] = layer_info['classalias']

        if is_main_layer or node_attr['classalias'].startswith('Conv'):
            if hasattr(layer, 'shape'):
                if len(layer.shape) == 3:
                    node_attr['out_size'] = (layer.shape[2], layer.shape[1])
                    node_attr['depth'] = layer.output_shape[0]
            if hasattr(layer, 'output_shape'):
                if len(layer.output_shape) == 4:
                    depth = layer.output_shape[1]
                    width, height = (layer.output_shape[3], layer.output_shape[2])
                    xshift = -width * (0.1 / (depth ** (1 / 3))) / 3
                    yshift = height * (0.1 / (depth ** (1 / 3))) / 2
                    node_attr['depth'] = depth
                    node_attr['xshift'] = xshift
                    node_attr['yshift'] = yshift
                    node_attr['out_size'] = (width, height)

                if len(layer.output_shape) == 2:
                    node_attr['out_size'] = (1, layer.output_shape[1])

        node_dict[key] = node_attr

        _input_layers = []
        if hasattr(layer, 'input_layers'):
            _input_layers += layer.input_layers
        if hasattr(layer, 'input_layer'):
            _input_layers += [layer.input_layer]

        for input_layer in _input_layers:
            parent_key = layerid(input_layer)
            edge = (parent_key, key)
            edge_list.append(edge)

    main_size_ = np.array((100, 100)) * 4
    sub_size = np.array((75, 50)) * 4

    # Setup scaled width and heights
    out_size_list = [v['out_size'] for v in node_dict.values() if 'out_size' in v]
    out_size_list = np.array(out_size_list)
    # out_size_list = out_size_list[out_size_list.T[0] > 1]
    area_arr = np.prod(out_size_list, axis=1)
    main_outsize = np.array(out_size_list[area_arr.argmax()])
    # main_outsize = np.array(out_size_list[area_arr.argmin()])
    scale = main_size_ / main_outsize

    scale_dense_max = 0.25
    scale_dense_min = 8

    for k, v in node_dict.items():
        if v['is_main_layer'] or v['classalias'].startswith('Conv'):
            if 'out_size' in v:
                # Make dense layers more visible
                if v['classalias'] == 'Dense':
                    v['shape'] = 'rect'
                    v['width'] = scale_dense_min
                    if v['out_size'][1] > main_outsize[1]:
                        v['height'] = v['out_size'][1] * scale[1] * scale_dense_max
                    elif v['out_size'][1] < scale_dense_min:
                        v['height'] = scale_dense_min * v['out_size'][1]
                    else:
                        v['height'] = v['out_size'][1]
                elif v['classalias'].startswith('Conv'):
                    v['shape'] = 'stack'
                    # v['shape'] = 'rect'
                    v['width'] = v['out_size'][0] * scale[0]
                    v['height'] = v['out_size'][1] * scale[1]
                else:
                    v['shape'] = 'rect'
                    v['width'] = v['out_size'][0] * scale[0]
                    v['height'] = v['out_size'][1] * scale[1]
            else:
                v['shape'] = 'rect'
                v['width'] = main_size_[0]
                v['height'] = main_size_[1]
        else:
            # v['shape'] = 'ellipse'
            v['shape'] = 'rect'
            v['style'] = 'rounded'
            v['width'] = sub_size[0]
            v['height'] = sub_size[1]

    key_order = ut.take(layer_to_id, layers)
    node_dict = ut.dict_subset(node_dict, key_order)

    # logger.info('node_dict = ' + ut.repr3(node_dict))

    # Create the networkx graph structure
    G = nx.DiGraph()
    G.add_nodes_from(node_dict.items())
    G.add_edges_from(edge_list)
    for key, val in edge_attrs.items():
        nx.set_edge_attributes(G, key, val)

    # Add invisible structure
    # main_nodes = [key for key, val in
    #              nx.get_node_attributes(G, 'is_main_layer').items() if val]

    main_children = ut.odict()

    # for n1, n2 in ut.itertwo(main_nodes):
    #    logger.info('n1, n2 = %r %r' % (n1, n2))
    #    import utool
    #    utool.embed()
    #    children = ut.nx_all_nodes_between(G, n1, n2)
    #    if n1 in children:
    #        children.remove(n1)
    #    if n2 in children:
    #        children.remove(n2)
    #    main_children[n1] = children

    #    #pass
    # main_children[main_nodes[-1]] = []

    for n1 in main_nodes:
        main_children[n1] = []
        # Main nodes only place constraints on nodes in the next main group.
        # Not their own
        next_main = None
        G.node[n1]['group'] = n1
        for (_, n2) in nx.bfs_edges(G, n1):
            if next_main is None:
                if n2 in main_nodes:
                    next_main = n2
                else:
                    G.node[n2]['group'] = n1
                    main_children[n1].append(n2)
            else:
                if n2 not in list(nx.descendants(G, next_main)):
                    G.node[n2]['group'] = n1
                    main_children[n1].append(n2)

    # Custom positioning
    x = 0
    y = 1000
    # logger.info('main_children = %s' % (ut.repr3(main_children),))

    # main_nodes = ut.isect(list(nx.topological_sort(G)), main_nodes)
    xpad = main_size_[0] * 0.3
    ypad = main_size_[1] * 0.3

    # Draw each main node, and then put its children under it
    # Then move to the left and draw the next main node.
    cumwidth = 0
    for n1 in main_nodes:
        cumheight = 0

        maxwidth = G.node[n1]['width']
        for n2 in main_children[n1]:
            maxwidth = max(maxwidth, G.node[n2]['width'])

        cumwidth += xpad
        cumwidth += maxwidth / 2

        pos = np.array([x + cumwidth, y - cumheight])
        G.node[n1]['pos'] = pos
        G.node[n1]['pin'] = 'true'

        height = G.node[n1]['height']
        cumheight += height / 2

        for n2 in main_children[n1]:
            height = G.node[n2]['height']
            cumheight += ypad
            cumheight += height / 2
            pos = np.array([x + cumwidth, y - cumheight])
            G.node[n2]['pos'] = pos
            G.node[n2]['pin'] = 'true'
            cumheight += height / 2

        cumwidth += maxwidth / 2

    # Pin everybody
    nx.set_node_attributes(G, 'pin', 'true')
    layoutkw = dict(prog='neato', splines='line')
    # layoutkw = dict(prog='neato', splines='spline')
    layoutkw = dict(prog='neato', splines='ortho')
    G_ = G.copy()
    # delete lables for positioning
    _labels = nx.get_node_attributes(G_, 'label')
    ut.nx_delete_node_attr(G_, 'label')
    nx.set_node_attributes(G_, 'label', '')
    nolayout = False
    if nolayout:
        G_.remove_edges_from(list(G_.edges()))
    else:
        layout_info = pt.nx_agraph_layout(G_, inplace=True, **layoutkw)  # NOQA
    # reset labels
    if not nolayout:
        nx.set_node_attributes(G_, 'label', _labels)
    _ = pt.show_nx(G_, fontsize=8, arrow_width=0.3, layout='custom', fnum=fnum)  # NOQA
    # pt.adjust_subplots(top=1, bot=0, left=0, right=1)
    pt.plt.tight_layout()


def pydot_to_image(pydot_graph):
    """
    References:
        http://stackoverflow.com/questions/4596962/display-graph-without-saving-using-pydot
    """
    from PIL import Image
    from six.moves import StringIO

    # from cStringIO import StringIO
    png_str = pydot_graph.create_png(prog='dot')
    sio = StringIO()
    sio.write(png_str)
    sio.seek(0)
    pil_img = Image.open(sio)
    img = np.asarray(pil_img.convert('RGB'))
    img = img[..., ::-1]  # to bgr
    pil_img.close()
    sio.close()
    return img


# def make_architecture_image(layers, **kwargs):
#    """
#    Args:
#        layers (list): List of the layers, as obtained from lasagne.layers.get_all_layers

#    Kwargs:
#        see docstring of make_architecture_pydot_graph for other options

#    References:
#        http://stackoverflow.com/questions/4596962/display-graph-without-saving-using-pydot

#    CommandLine:
#        python -m wbia_cnn.draw_net --test-make_architecture_image --show

#    Example:
#        >>> # ENABLE_DOCTEST
#        >>> from wbia_cnn.draw_net import *  # NOQA
#        >>> from wbia_cnn import models
#        >>> model = models.SiameseCenterSurroundModel(autoinit=True)
#        >>> #model = models.DummyModel(autoinit=True)
#        >>> layers = model.get_all_layers()
#        >>> # execute function
#        >>> kwargs = {}
#        >>> img = make_architecture_image(layers, **kwargs)
#        >>> print(img.shape)
#        >>> ut.quit_if_noshow()
#        >>> from wbia import plottool as pt
#        >>> pt.imshow(img)
#        >>> ut.show_if_requested()
#    """
#    # from IPython.display import Image  # needed to render in notebook
#    pydot_graph = make_architecture_pydot_graph(layers, **kwargs)
#    img = pydot_to_image(pydot_graph)
#    return img


# def imwrite_arch(layers, fpath, **kwargs):
#    """
#    Draws a network diagram to a file

#    Args:
#        layers (list): List of the layers, as obtained from lasagne.layers.get_all_layers
#        fpath (str): The fpath to save output to.

#        Kwargs:
#            see docstring of make_architecture_pydot_graph for other options

#    CommandLine:
#        python -m wbia_cnn.draw_net --test-imwrite_arch --show

#    Example:
#        >>> # ENABLE_DOCTEST
#        >>> from wbia_cnn.draw_net import *  # NOQA
#        >>> from wbia_cnn import models
#        >>> #model = models.DummyModel(autoinit=True)
#        >>> model = models.SiameseCenterSurroundModel(autoinit=True)
#        >>> layers = model.get_all_layers()
#        >>> fpath = ut.unixjoin(ut.ensure_app_resource_dir('wbia_cnn'), 'tmp.png')
#        >>> # execute function
#        >>> imwrite_arch(layers, fpath)
#        >>> ut.quit_if_noshow()
#        >>> ut.startfile(fpath)
#    """
#    pydot_graph = make_architecture_pydot_graph(layers, **kwargs)

#    ext = fpath[fpath.rfind('.') + 1:]
#    with open(fpath, 'w') as fid:
#        fid.write(pydot_graph.create(format=ext))


def occlusion_heatmap(net, x, target, square_length=7):
    """An occlusion test that checks an image for its critical parts.
    In this function, a square part of the image is occluded (i.e. set
    to 0) and then the net is tested for its propensity to predict the
    correct label. One should expect that this propensity shrinks of
    critical parts of the image are occluded. If not, this indicates
    overfitting.
    Depending on the depth of the net and the size of the image, this
    function may take awhile to finish, since one prediction for each
    pixel of the image is made.
    Currently, all color channels are occluded at the same time. Also,
    this does not really work if images are randomly distorted by the
    batch iterator.
    See paper: Zeiler, Fergus 2013
    Parameters
    ----------
    net : NeuralNet instance
      The neural net to test.
    x : np.array
      The input data, should be of shape (1, c, x, y). Only makes
      sense with image data.
    target : int
      The true value of the image. If the net makes several
      predictions, say 10 classes, this indicates which one to look
      at.
    square_length : int (default=7)
      The length of the side of the square that occludes the image.
      Must be an odd number.
    Results
    -------
    heat_array : np.array (with same size as image)
      An 2D np.array that at each point (i, j) contains the predicted
      probability of the correct class if the image is occluded by a
      square with center (i, j).
    """
    from lasagne.layers import get_output_shape

    if (x.ndim != 4) or x.shape[0] != 1:
        raise ValueError(
            'This function requires the input data to be of '
            'shape (1, c, x, y), instead got {}'.format(x.shape)
        )
    if square_length % 2 == 0:
        raise ValueError(
            'Square length has to be an odd number, instead '
            'got {}.'.format(square_length)
        )

    num_classes = get_output_shape(net.layers_[-1])[1]
    img = x[0].copy()
    bs, col, s0, s1 = x.shape

    heat_array = np.zeros((s0, s1))
    pad = square_length // 2 + 1
    x_occluded = np.zeros((s1, col, s0, s1), dtype=img.dtype)
    probs = np.zeros((s0, s1, num_classes))

    # generate occluded images
    for i in range(s0):
        # batch s1 occluded images for faster prediction
        for j in range(s1):
            x_pad = np.pad(img, ((0, 0), (pad, pad), (pad, pad)), 'constant')
            x_pad[:, i : i + square_length, j : j + square_length] = 0.0
            x_occluded[j] = x_pad[:, pad:-pad, pad:-pad]
        y_proba = net.predict_proba_Xb(x_occluded)
        probs[i] = y_proba.reshape(s1, num_classes)

    # from predicted probabilities, pick only those of target class
    for i in range(s0):
        for j in range(s1):
            heat_array[i, j] = probs[i, j, target]
    return heat_array


def _plot_heat_map(net, Xb, figsize, get_heat_image):
    from wbia import plottool as pt

    if Xb.ndim != 4:
        raise ValueError(
            'This function requires the input data to be of '
            'shape (b, c, x, y), instead got {}'.format(Xb.shape)
        )

    num_images = Xb.shape[0]
    if figsize[1] is None:
        figsize = (figsize[0], num_images * figsize[0] / 3)
    figs, axes = pt.plt.subplots(num_images, 3, figsize=figsize)

    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')

    for n in range(num_images):
        heat_img = get_heat_image(net, Xb[n : n + 1, :, :, :], n)

        ax = axes if num_images == 1 else axes[n]
        img = Xb[n, :, :, :].mean(0)
        ax[0].imshow(-img, interpolation='nearest', cmap='gray')
        ax[0].set_title('image')
        ax[1].imshow(-heat_img, interpolation='nearest', cmap='Reds')
        ax[1].set_title('critical parts')
        ax[2].imshow(-img, interpolation='nearest', cmap='gray')
        ax[2].imshow(-heat_img, interpolation='nearest', cmap='Reds', alpha=0.6)
        ax[2].set_title('super-imposed')
    return pt.plt


def plot_occlusion(net, Xb, target, square_length=7, figsize=(9, None)):
    """Plot which parts of an image are particularly import for the
    net to classify the image correctly.
    See paper: Zeiler, Fergus 2013
    Parameters
    ----------
    net : NeuralNet instance
      The neural net to test.
    Xb : numpy.array
      The input data, should be of shape (b, c, 0, 1). Only makes
      sense with image data.
    target : list or numpy.array of ints
      The true values of the image. If the net makes several
      predictions, say 10 classes, this indicates which one to look
      at. If more than one sample is passed to Xb, each of them needs
      its own target.
    square_length : int (default=7)
      The length of the side of the square that occludes the image.
      Must be an odd number.
    figsize : tuple (int, int)
      Size of the figure.
    Plots
    -----
    Figure with 3 subplots: the original image, the occlusion heatmap,
    and both images super-imposed.
    """
    return _plot_heat_map(
        net,
        Xb,
        figsize,
        lambda net, Xb, n: occlusion_heatmap(net, Xb, target[n], square_length),
    )


def plot_saliency(net, Xb, figsize=(9, None)):
    def saliency_map(input, output, pred, Xb):
        import theano.tensor as T
        from lasagne.objectives import binary_crossentropy

        score = -binary_crossentropy(output[:, pred], np.array([1])).sum()
        heat_map_ = np.abs(T.grad(score, input).eval({input: Xb}))
        return heat_map_

    def saliency_map_net(net, Xb):
        from lasagne.layers import get_output

        input = net.layers_[0].input_var
        output = get_output(net.layers_[-1])
        pred = output.eval({input: Xb}).argmax(axis=1)
        heat_map_ = saliency_map(input, output, pred, Xb)
        heat_img = heat_map_[0].transpose(1, 2, 0).squeeze()
        return heat_img

    return _plot_heat_map(net, Xb, figsize, lambda net, Xb, n: -saliency_map_net(net, Xb))


class Dream(object):
    """
    https://groups.google.com/forum/#!topic/lasagne-users/UxZpNthZfq0
    http://arxiv.org/pdf/1312.6034.pdf
    http://igva2012.wikispaces.asu.edu/file/view/Erhan+2009+Visualizing+higher+layer+features+of+a+deep+network.pdf

    #TODO
    https://arxiv.org/pdf/1605.09304v3.pdf

    Class model visualization. Sort of like a deep-dream

    CommandLine:
        python -m wbia_cnn.draw_net Dream --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> # Assumes mnist is trained
        >>> from wbia_cnn.draw_net import *  # NOQA
        >>> from wbia_cnn.models import mnist
        >>> model, dataset = mnist.testdata_mnist(dropout=.5)
        >>> model.init_arch()
        >>> model.load_model_state()
        >>> target_labels = 3
        >>> ut.quit_if_noshow()
        >>> from wbia import plottool as pt
        >>> #pt.qt4ensure()
        >>> dream = Dream(model, niters=200)
        >>> img = dream.make_class_images(target_labels)
        >>> pt.imshow(img)
        >>> ut.show_if_requested()
    """

    def saliency(dream, Xb, yb):
        """
        num = 10
        Xb = model.prepare_data(X_test[0:num])
        yb = y_test[0:num]

        dpath = ''
        dataset = None
        """
        dpath = '.'

        import theano.tensor as T
        import lasagne
        import vtool as vt
        import theano

        model = dream.model

        # Use current weights to find the score of a particular class
        Xb_shared = theano.shared(Xb)
        yb_shared = theano.shared(yb.astype(np.int32))

        # Get the final layer and remove the softmax nonlinearity to access the
        # pre-activation. (Softmax encourages minimization of other classes)
        import copy

        # softmax = copy.copy(model.output_layer)
        # softmax.nonlinearity = lasagne.nonlinearities.identity
        softmax = copy.copy(model.output_layer)

        class_probs = lasagne.layers.get_output(softmax, Xb_shared, deterministic=True)

        # werid way to index into position of target
        flat_idx = (T.arange(yb_shared.shape[0]) * class_probs.shape[1]) + yb_shared
        class_probs_target = T.flatten(class_probs)[flat_idx]

        # Get derivative of scores for the target class wrt the input
        d_score_wrt_input = theano.grad(class_probs_target.mean(), Xb_shared)
        w = np.array(d_score_wrt_input.eval())
        saliency = w.max(axis=1, keepdims=True)

        outs = saliency.transpose((0, 2, 3, 1))
        X = Xb.transpose((0, 2, 3, 1))

        for count in range(len(Xb)):
            img = X[count]
            y = yb[count]
            out = vt.norm01(outs[count])
            overlay = vt.blend_images_multiply(out, img)

            vt.imwrite(
                join(dpath, 'out%d_A_image_t=%s.jpg' % (count, y)),
                vt.rectify_to_uint8(img),
            )
            vt.imwrite(
                join(dpath, 'out%d_B_heat_t=%s.jpg' % (count, y)),
                vt.rectify_to_uint8(out),
            )
            vt.imwrite(
                join(dpath, 'out%d_C_overlay_t=%s.jpg' % (count, y)),
                vt.rectify_to_uint8(overlay),
            )

    def __init__(
        dream, model, init='gauss', niters=100, update_rate=1e-2, weight_decay=1e-5
    ):
        dream.model = model
        dream.init = init
        dream.niters = niters
        dream.update_rate = update_rate
        dream.weight_decay = weight_decay
        # FIXME: cached vars assumes not much changes
        dream.shared_images = None
        dream.step_fn = None

    def make_class_images(dream, target_labels):
        import theano
        from theano import tensor as T  # NOQA
        import utool as ut

        was_scalar = not ut.isiterable(target_labels)
        target_labels = ut.ensure_iterable(target_labels)

        if True:
            # We are forcing a batch size for this visualization
            input_shape = (len(target_labels),) + dream.model.input_shape[1:]
        else:
            # Maybe some cnn layers cant take variable batches?
            input_shape = dream.model.input_shape
        b, c, w, h = input_shape
        assert len(target_labels) <= b, 'batch size too small'

        initial_state = dream._make_init_state()

        # make image a shared variable that you can update
        if dream.shared_images is None:
            dream.shared_images = theano.shared(initial_state)
        else:
            dream.shared_images.set_value(initial_state)

        if dream.step_fn is None:
            dream.step_fn = dream._make_objective(dream.shared_images, target_labels)

        # Optimize objective via backpropogation for a few iterations
        for _ in ut.ProgIter(range(dream.niters), lbl='making class model img', bs=True):
            dream.step_fn()
            # logger.info('objective = %r' % (objective,))

        out = dream._postprocess_class_image(
            dream.shared_images, target_labels, was_scalar
        )
        return out

    def _make_init_state(dream):
        r"""

        CommandLine:
            python -m wbia_cnn.draw_net _make_init_state --show

        Example:
            >>> # DISABLE_DOCTEST
            >>> # Assumes mnist is trained
            >>> from wbia_cnn.draw_net import *  # NOQA
            >>> from wbia_cnn.models import mnist
            >>> model, dataset = mnist.testdata_mnist(dropout=.5)
            >>> model.init_arch()
            >>> dream = Dream(model, init='rgauss', niters=200)
            >>> ut.quit_if_noshow()
            >>> from wbia import plottool as pt
            >>> import vtool as vt
            >>> #pt.qt4ensure()
            >>> initial_state = dream._make_init_state().transpose((0, 2, 3, 1))[0]
            >>> pt.imshow(initial_state, pnum=(1, 2, 1), fnum=1)
            >>> pt.imshow(vt.norm01(initial_state), pnum=(1, 2, 2), fnum=1)
            >>> ut.show_if_requested()
        """
        init = dream.init
        input_shape = dream.model.input_shape
        b, c, w, h = input_shape

        rng = np.random.RandomState(0)

        if init is None or init == 'zeros':
            # intializing to zeros seems to do nothing on mnist data
            initial_state = np.zeros(input_shape, dtype=np.float32)
        if init in ['rand', 'random']:
            initial_state = rng.rand(*input_shape)
        elif init in ['randn']:
            initial_state = np.abs(rng.randn(*input_shape)) / 6
            initial_state = np.clip(initial_state, 0, 1)
        elif init in ['gauss']:
            import vtool as vt

            initial_state = np.array(
                [[vt.gaussian_patch((h, w), sigma=None) for _ in range(c)]] * b
            )
            # initial_state /= initial_state.max()
        elif init in ['rgauss']:
            import vtool as vt

            initial_state = np.array(
                [[vt.gaussian_patch((h, w), sigma=None) for _ in range(c)]] * b
            )
            # initial_state /= initial_state.max()
            raug = np.abs(rng.randn(*input_shape)) * (initial_state.max() / 12)
            initial_state += raug
            initial_state = np.clip(initial_state, 0, 1)

        elif init in ['perlin']:
            import vtool as vt

            b, c, w, h = input_shape
            initial_state = np.array(
                [[vt.perlin_noise((w, h), rng=rng) for _ in range(c)]] * b
            )
            initial_state = initial_state.astype(np.float32) / 255
        initial_state = initial_state.astype(np.float32)
        return initial_state

    def _make_objective(dream, shared_images, target_labels):
        r"""
        The goal is to optimize
        S_c = score of class c before softmax nonlinearity
        argmax_{I} S_c(I) - \lambda \elltwo{I}
        max(S_c(I) - lambda * norm(I, 2))
        """
        import lasagne
        import copy
        import theano
        from theano import tensor as T  # NOQA

        logger.info('Making dream objective')
        # Get the final layer and remove the softmax nonlinearity to access the
        # pre-activation. (Softmax encourages minimization of other classes)
        softmax = copy.copy(dream.model.output_layer)
        softmax.nonlinearity = lasagne.nonlinearities.identity

        # Overwrite lasagne's InputLayer with the image
        # Build expression to represent class scores wrt the image
        class_scores = lasagne.layers.get_output(
            softmax, shared_images, deterministic=True
        )

        # Get the class score that represents our class of interest
        # simultaniously generate as many classes as were requested.
        max_term_batch = [class_scores[bx, y] for bx, y in enumerate(target_labels)]
        max_term = T.mean(max_term_batch)

        # Get the squared L2 norm of the image values
        flat_img = T.reshape(
            shared_images, (shared_images.shape[0], T.prod(shared_images.shape[1:]))
        )
        reg_term_batch = (flat_img ** 2).sum(axis=1)
        reg_term = T.mean(reg_term_batch)

        objective = max_term - dream.weight_decay * reg_term

        # Compute the gradient of the maximization objective
        # with respect to the image
        grads = theano.grad(objective, wrt=shared_images)

        # compile a function that does this update
        step_fn = theano.function(
            inputs=[],
            # outputs could be empty, but returning objective allows us to
            # monitor progress
            outputs=[objective],
            updates={shared_images: shared_images + dream.update_rate * grads},
        )
        return step_fn

    def _postprocess_class_image(dream, shared_images, target_labels, was_scalar):
        # return final state of the image
        Xb = shared_images.get_value()
        X = Xb.transpose((0, 2, 3, 1))
        out_ = X[0 : len(target_labels)]
        import vtool as vt  # NOQA

        out = out_.copy()
        out = vt.norm01(out) * 255
        out = np.round(out).astype(np.uint8)
        if was_scalar:
            out = out[0]
        return out

    def generate_class_images(dream, target_labels):
        """
        from wbia import plottool as pt
        fnum = None
        kw = dict(init='gauss', niters=500, update_rate=.05, weight_decay=1e-4)
        target_labels = list(range(model.output_dims))
        dream = draw_net.Dream(model, **kw)
        target_labels = 8
        images = list(dream.generate_class_images(target_labels))

        vid = vt.make_video(images, 'dynimg.pimj', fps=1, is_color=False, format='PIM1')
        vid = vt.make_video2(images, 'dynimg')

        import matplotlib.pyplot as plt
        ims = []
        for img in imgs:
            im = plt.imshow(img[:, :, 0], interpolation='nearest', cmap='gray')
            ims.append([im])

        import matplotlib.animation as animation
        fig = plt.figure()
        ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                        repeat_delay=1000)
        ani.save('dynamic_images.mp4')
        ut.startfile('dynamic_images.mp4')
        plt.show()
        """
        import theano
        from theano import tensor as T  # NOQA
        import utool as ut

        input_shape = dream.model.input_shape
        b, c, w, h = input_shape
        was_scalar = not ut.isiterable(target_labels)
        target_labels = ut.ensure_iterable(target_labels)
        assert len(target_labels) <= b, 'batch size too small'
        initial_state = dream._make_init_state()
        shared_images = theano.shared(initial_state.astype(np.float32))
        step_fn = dream._make_objective(shared_images, target_labels)
        out = dream._postprocess_class_image(shared_images, target_labels, was_scalar)
        yield out
        for _ in ut.ProgIter(range(dream.niters), lbl='class dream', bs=True):
            step_fn()
            # objective = step_fn()
            # logger.info('objective = %r' % (objective,))
            out = dream._postprocess_class_image(shared_images, target_labels, was_scalar)
            yield out


def show_saliency_heatmap(model, dataset):
    """
    https://github.com/dnouri/nolearn/blob/master/nolearn/lasagne/visualize.py

    Example:
        >>> # DISABLE_DOCTEST
        >>> # Assumes mnist is trained
        >>> from wbia_cnn.draw_net import *  # NOQA
        >>> from wbia_cnn import ingest_data
        >>> from wbia_cnn.models import MNISTModel
        >>> dataset = ingest_data.grab_mnist_category_dataset()
        >>> model = MNISTModel(batch_size=128, data_shape=dataset.data_shape,
        >>>                    name='bnorm',
        >>>                    output_dims=len(dataset.unique_labels),
        >>>                    batch_norm=True,
        >>>                    dataset_dpath=dataset.dataset_dpath)
        >>> model.encoder = None
        >>> model.init_arch()
        >>> model.load_model_state()
        >>> from wbia import plottool as pt
        >>> pt.qt4ensure()
        >>> show_saliency_heatmap(model, dataset)
        >>> ut.show_if_requested()
    """
    if dataset.has_subset('valid'):
        X_train, y_train = dataset.subset('train')
        _, _, X_valid, y_valid = model._prefit(X_train, y_train)
    else:
        X_valid, y_valid = dataset.subset('valid')
    net = model
    num = 4
    start = 0
    X = X_valid[start : start + num]
    y = y_valid[start : start + num]
    Xb = net.prepare_input(X)
    plot_occlusion(net, Xb, y)
    # plot_saliency(net, Xb)


def show_convolutional_weights(
    all_weights, use_color=None, limit=144, fnum=None, pnum=(1, 1, 1)
):
    r"""
    Args:
        all_weights (?):
        use_color (bool):
        limit (int):

    CommandLine:
        python -m wbia_cnn.draw_net --test-show_convolutional_weights --show
        python -m wbia_cnn.draw_net --test-show_convolutional_weights --show --index=1

        # Need to fit mnist first
        python -m wbia_cnn _ModelFitting.fit:1 --vd


    Example:
        >>> # DISABLE_DOCTEST
        >>> # Assumes mnist is trained
        >>> from wbia_cnn.draw_net import *  # NOQA
        >>> from wbia_cnn import ingest_data
        >>> from wbia_cnn.models import MNISTModel
        >>> dataset = ingest_data.grab_mnist_category_dataset()
        >>> model = MNISTModel(batch_size=128, data_shape=dataset.data_shape,
        >>>                    name='bnorm',
        >>>                    output_dims=len(dataset.unique_labels),
        >>>                    batch_norm=True,
        >>>                    dataset_dpath=dataset.dataset_dpath)
        >>> model.encoder = None
        >>> model.init_arch()
        >>> model.load_model_state()
        >>> nn_layers = model.get_all_layers()
        >>> weighted_layers = [layer for layer in nn_layers if hasattr(layer, 'W')]
        >>> all_weights = weighted_layers[0].W.get_value()
        >>> print('all_weights.shape = %r' % (all_weights.shape,))
        >>> use_color = None
        >>> limit = 64
        >>> fig = show_convolutional_weights(all_weights, use_color, limit)
        >>> ut.quit_if_noshow()
        >>> from wbia import plottool as pt
        >>> pt.qt4ensure()
        >>> fig = show_convolutional_weights(all_weights, use_color, limit)
        >>> ut.show_if_requested()

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia_cnn.draw_net import *  # NOQA
        >>> from wbia_cnn import models
        >>> from lasagne import layers
        >>> model = models.SiameseCenterSurroundModel(autoinit=True)
        >>> output_layer = model.get_output_layer()
        >>> nn_layers = layers.get_all_layers(output_layer)
        >>> weighted_layers = [layer for layer in nn_layers if hasattr(layer, 'W')]
        >>> index = ut.get_argval('--index', type_=int, default=0)
        >>> all_weights = weighted_layers[index].W.get_value()
        >>> print('all_weights.shape = %r' % (all_weights.shape,))
        >>> use_color = None
        >>> limit = 64
        >>> fig = show_convolutional_weights(all_weights, use_color, limit)
        >>> ut.show_if_requested()
    """
    from wbia import plottool as pt

    if fnum is None:
        fnum = pt.next_fnum()
    fig = pt.figure(fnum=fnum, pnum=pnum, docla=True)
    num, channels, height, width = all_weights.shape
    if use_color is None:
        # Try to infer if use_color should be shown
        use_color = channels == 3

    stacked_img = make_conv_weight_image(all_weights, limit)
    # ax = fig.add_subplot(111)
    if len(stacked_img.shape) == 3 and stacked_img.shape[-1] == 1:
        stacked_img = stacked_img.reshape(stacked_img.shape[:-1])
    pt.imshow(stacked_img)
    return fig


def make_conv_weight_image(all_weights, limit=144):
    """just makes the image ndarray of the weights"""
    import vtool as vt
    import cv2

    # Try to infer if use_color should be shown
    num, channels, height, width = all_weights.shape
    # Try to infer if use_color should be shown
    use_color = channels == 3
    # non-use_color features need to be flattened
    if not use_color:
        all_weights_ = all_weights.reshape(num * channels, height, width, 1)
    else:
        # convert from theano to cv2 BGR
        all_weights_ = utils.convert_theano_images_to_cv2_images(all_weights)
        # convert from BGR to RGB
        all_weights_ = all_weights_[..., ::-1]
        # cv2.cvtColor(all_weights_[-1], cv2.COLOR_BGR2RGB)

    # Limit all_weights_
    # num = all_weights_.shape[0]
    num, height, width, channels = all_weights_.shape
    if limit is not None and num > limit:
        all_weights_ = all_weights_[:limit]
        num = all_weights_.shape[0]

    # Convert weight values to image values
    normalize_individually = False
    if normalize_individually:
        # Normalize each feature individually
        all_max = vt.multiaxis_reduce(np.amax, all_weights_, startaxis=1)
        all_min = vt.multiaxis_reduce(np.amin, all_weights_, startaxis=1)
        all_domain = all_max - all_min
        extra_dims = (None,) * (len(all_weights_.shape) - 1)
        broadcaster = (slice(None),) + extra_dims
        all_features = (
            (all_weights_ - all_min[broadcaster]) * (255.0 / all_domain[broadcaster])
        ).astype(np.uint8)
    else:
        # Normalize jointly across all filters
        _max = all_weights_.max()
        _min = all_weights_.min()
        _domain = _max - _min
        all_features = ((all_weights_ - _min) * (255.0 / _domain)).astype(np.uint8)

    # import scipy.misc
    # resize feature, give them a border, and stack them together
    new_height, new_width = max(32, height), max(32, width)
    nbp_ = 1  # num border pixels
    _resized_features = np.array(
        [
            cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
            for img in all_features
        ]
    )
    resized_features = _resized_features.reshape(num, new_height, new_width, channels)
    border_shape = (num, new_height + (nbp_ * 2), new_width + (nbp_ * 2), channels)
    bordered_features = np.zeros(border_shape, dtype=resized_features.dtype)
    bordered_features[:, nbp_:-nbp_, nbp_:-nbp_, :] = resized_features
    # img_list = bordered_features
    stacked_img = vt.stack_square_images(bordered_features)
    return stacked_img


def output_confusion_matrix(X_test, results_path, test_results, model, **kwargs):
    """currently hacky implementation, fix it later"""
    loss, accu_test, prob_list, auglbl_list, pred_list, conf_list = test_results
    # Output confusion matrix
    mapping_fn = None
    if model is not None:
        mapping_fn = getattr(model, 'label_order_mapping', None)
    # TODO: THIS NEEDS TO BE FIXED
    label_list = list(range(kwargs.get('output_dims')))
    # Encode labels if avaialble
    # encoder = kwargs.get('encoder', None)
    encoder = getattr(model, 'encoder', None)
    if encoder is not None:
        label_list = encoder.inverse_transform(label_list)
    # Make confusion matrix (pass X to write out failed cases)
    show_confusion_matrix(
        auglbl_list, pred_list, label_list, results_path, mapping_fn, X_test
    )


def save_confusion_matrix(
    results_path, correct_y, predict_y, category_list, mapping_fn=None, data_x=None
):
    from wbia import plottool as pt

    fig = show_confusion_matrix(
        correct_y, predict_y, category_list, mapping_fn=mapping_fn, data_x=data_x
    )
    output_fpath = join(results_path, 'confusion.png')
    pt.save_figure(fig, fpath=output_fpath)
    return output_fpath


def show_confusion_matrix(
    correct_y, predict_y, category_list, results_path, mapping_fn=None, data_x=None
):
    """
    Given the correct and predict labels, show the confusion matrix

    Args:
        correct_y (list of int): the list of correct labels
        predict_y (list of int): the list of predict assigned labels
        category_list (list of str): the category list of all categories

    Displays:
        matplotlib: graph of the confusion matrix

    Returns:
        None

    TODO FIXME and simplify
    """
    import matplotlib.pyplot as plt
    import cv2

    confused_examples = join(results_path, 'confused')
    if data_x is not None:
        if exists(confused_examples):
            ut.remove_dirs(confused_examples, quiet=True)
        ut.ensuredir(confused_examples)
    size = len(category_list)

    if mapping_fn is None:
        # Identity
        category_mapping = {key: index for index, key in enumerate(category_list)}
        category_list_ = category_list
    else:
        category_mapping = mapping_fn(category_list)
        assert all(
            [category in category_mapping.keys() for category in category_list]
        ), 'Not all categories are mapped'
        values = list(category_mapping.values())
        assert len(list(set(values))) == len(
            values
        ), 'Mapped categories have a duplicate assignment'
        assert 0 in values, 'Mapped categories must have a 0 index'
        temp = list(category_mapping.iteritems())
        temp = sorted(temp, key=itemgetter(1))
        category_list_ = [t[0] for t in temp]

    confidences = np.zeros((size, size))
    counters = {}
    for index, (correct, predict) in enumerate(zip(correct_y, predict_y)):
        # Ensure type
        correct = int(correct)
        predict = int(predict)
        # Get the "text" label
        example_correct_label = category_list[correct]
        example_predict_label = category_list[predict]
        # Perform any mapping that needs to be done
        correct_ = category_mapping[example_correct_label]
        predict_ = category_mapping[example_predict_label]
        # Add to the confidence matrix
        confidences[correct_][predict_] += 1
        if data_x is not None and correct_ != predict_:
            example = data_x[index]
            example_name = '%s^SEEN_INCORRECTLY_AS^%s' % (
                example_correct_label,
                example_predict_label,
            )
            if example_name not in counters.keys():
                counters[example_name] = 0
            counter = counters[example_name]
            counters[example_name] += 1
            example_name = '%s^%d.png' % (example_name, counter)
            example_path = join(confused_examples, example_name)
            # TODO: make write confused examples function
            cv2.imwrite(example_path, example)

    row_sums = np.sum(confidences, axis=1)
    norm_conf = (confidences.T / row_sums).T

    fig = plt.figure(1)
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(np.array(norm_conf), cmap=plt.cm.jet, interpolation='nearest')

    for x in range(size):
        for y in range(size):
            ax.annotate(
                str(int(confidences[x][y])),
                xy=(y, x),
                horizontalalignment='center',
                verticalalignment='center',
            )

    cb = fig.colorbar(res)  # NOQA
    plt.xticks(np.arange(size), category_list_[0:size], rotation=90)
    plt.yticks(np.arange(size), category_list_[0:size])
    margin_small = 0.1
    margin_large = 0.9
    plt.subplots_adjust(
        left=margin_small, right=margin_large, bottom=margin_small, top=margin_large
    )
    plt.xlabel('Predicted')
    plt.ylabel('Correct')
    return fig


if __name__ == '__main__':
    """
    CommandLine:
        python -m wbia_cnn.draw_net
        python -m wbia_cnn.draw_net --allexamples
        python -m wbia_cnn.draw_net --allexamples --noface --nosrc
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
