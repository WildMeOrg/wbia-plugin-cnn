#!/usr/bin/env python
"""
constructs the Theano optimization and trains a learning model,
optionally by initializing the network with pre-trained weights.
"""
from __future__ import absolute_import, division, print_function
from ibeis_cnn import utils
from ibeis_cnn import models
from ibeis_cnn import ibsplugin

import time
import theano
import numpy as np
import cPickle as pickle
from lasagne import layers
from sklearn import preprocessing

import utool as ut
import six
from os.path import join, abspath, dirname


def train(data_file, labels_file, model, weights_file, results_path,
          pretrained_weights_file=None, pretrained_kwargs=False, **kwargs):
    """
    Driver function

    Args:
        data_file (?):
        labels_file (?):
        model (?):
        weights_file (?):
        pretrained_weights_file (None):
        pretrained_kwargs (bool):
    """

    ######################################################################################

    # Load the data
    print('\n[data] loading data...')
    print('data_file = %r' % (data_file,))
    print('labels_file = %r' % (labels_file,))
    data, labels = utils.load(data_file, labels_file)

    # Ensure results dir
    weights_path = dirname(abspath(weights_file))
    ut.ensuredir(weights_path)
    ut.ensuredir(results_path)
    if pretrained_weights_file is not None:
        pretrained_weights_path = dirname(abspath(pretrained_weights_file))
        ut.ensuredir(pretrained_weights_path)

    # Training parameters defaults
    utils._update(kwargs, 'center',         True)
    utils._update(kwargs, 'encode',         True)
    utils._update(kwargs, 'learning_rate',  0.03)
    utils._update(kwargs, 'momentum',       0.9)
    utils._update(kwargs, 'batch_size',     128)
    utils._update(kwargs, 'patience',       15)
    utils._update(kwargs, 'test',           5)  # Test every X epochs
    utils._update(kwargs, 'max_epochs',     kwargs.get('patience') * 10)
    utils._update(kwargs, 'regularization', None)
    utils._update(kwargs, 'test_time_augmentation',  False)
    utils._update(kwargs, 'output_dims',    None)
    utils._update(kwargs, 'show_features',  True)

    # Automatically figure out how many classes
    if kwargs.get('output_dims') is None:
        kwargs['output_dims'] = len(list(np.unique(labels)))

    # print('[load] adding channels...')
    # data = utils.add_channels(data)
    print('[train]     data.shape = %r' % (data.shape,))
    print('[train]     data.dtype = %r' % (data.dtype,))
    print('[train]     labels.shape = %r' % (labels.shape,))
    print('[train]     labels.dtype = %r' % (labels.dtype,))

    labelhist = {key: len(val) for key, val in six.iteritems(ut.group_items(labels, labels))}
    print('label stats = \n' + ut.dict_str(labelhist))
    print('train kwargs = \n' + (ut.dict_str(kwargs)))

    # Encoding labels
    kwargs['encoder'] = None
    if kwargs.get('encode', False):
        kwargs['encoder'] = preprocessing.LabelEncoder()
        kwargs['encoder'].fit(labels)

    # utils.show_image_from_data(data)

    # Split the dataset into training and validation
    print('[train] creating train, validation datasaets...')
    dataset = utils.train_test_split(data, labels, eval_size=0.2)
    X_train, y_train, X_valid, y_valid = dataset
    dataset = utils.train_test_split(X_train, y_train, eval_size=0.1)
    X_train, y_train, X_test, y_test = dataset

    # Build and print the model
    print('\n[model] building model...')
    input_cases, input_channels, input_height, input_width = data.shape
    output_layer = model.build_model(
        kwargs.get('batch_size'), input_width, input_height,
        input_channels, kwargs.get('output_dims'))
    utils.print_layer_info(output_layer)

    # Create the Theano primitives
    print('[model] creating Theano primitives...')
    learning_rate_theano = theano.shared(utils.float32(kwargs.get('learning_rate')))
    all_iters = utils.create_iter_funcs(learning_rate_theano, output_layer, **kwargs)
    train_iter, valid_iter, test_iter = all_iters

    # Load the pretrained model if specified
    if pretrained_weights_file is not None:
        print('[model] loading pretrained weights from %s' % (pretrained_weights_file))
        with open(pretrained_weights_file, 'rb') as pfile:
            kwargs_ = pickle.load(pfile)
            pretrained_weights = kwargs_.get('best_weights', None)
            layers.set_all_param_values(output_layer, pretrained_weights)
            if pretrained_kwargs:
                kwargs = kwargs_

    # Center the data by subtracting the mean (AFTER KWARGS UPDATE)
    if kwargs.get('center'):
        print('[train] applying data centering...')
        utils._update(kwargs, 'center_mean', np.mean(X_train, axis=0))
        # utils._update(kwargs, 'center_std', np.std(X_train, axis=0))
        utils._update(kwargs, 'center_std', 255.0)
    else:
        utils._update(kwargs, 'center_mean', 0.0)
        utils._update(kwargs, 'center_std', 255.0)

    # Begin training the neural network
    vals = (utils.get_current_time(), kwargs.get('learning_rate'), )
    print('\n[train] starting training at %s with learning rate %.9f' % vals)
    utils.print_header_columns()

    utils._update(kwargs, 'best_weights',        None)
    utils._update(kwargs, 'best_valid_accuracy', 0.0)
    utils._update(kwargs, 'best_test_accuracy',  0.0)
    utils._update(kwargs, 'best_train_loss',     np.inf)
    utils._update(kwargs, 'best_valid_loss',     np.inf)
    utils._update(kwargs, 'best_valid_accuracy', 0.0)
    utils._update(kwargs, 'best_test_accuracy',  0.0)
    try:
        epoch = 0
        epoch_marker = epoch
        while True:
            try:
                # Start timer
                t0 = time.time()

                # compute the loss over all training and validation batches
                augment_fn = getattr(model, 'augment', None)
                avg_train_loss = utils.forward_train(X_train, y_train, train_iter, rand=True,
                                                     augment=augment_fn, **kwargs)
                if kwargs.get('test_time_augmentation', False):
                    avg_valid_data = utils.forward_valid(X_valid, y_valid, valid_iter,
                                                         augment=augment_fn, **kwargs)
                else:
                    avg_valid_data = utils.forward_valid(X_valid, y_valid, valid_iter, **kwargs)
                avg_valid_loss, avg_valid_accuracy = avg_valid_data

                # If the training loss is nan, the training has diverged
                if np.isnan(avg_train_loss):
                    print('\n[train] training diverged\n')
                    break

                # Is this model the best we've ever seen?
                best_found = avg_valid_loss < kwargs.get('best_valid_loss')
                if best_found:
                    kwargs['best_epoch'] = epoch
                    epoch_marker = epoch
                    kwargs['best_weights'] = layers.get_all_param_values(output_layer)

                # compute the loss over all testing batches
                request_test = kwargs.get('test') is not None and epoch % kwargs.get('test') == 0
                if best_found or request_test:
                    avg_test_accuracy = utils.forward_test(X_test, y_test, test_iter,
                                                           results_path, **kwargs)
                else:
                    avg_test_accuracy = None

                # Output the layer 1 features
                if kwargs.get('show_features'):
                    utils.show_convolutional_features(output_layer, results_path, target=0)

                # Running tab for what the best model
                if avg_train_loss < kwargs.get('best_train_loss'):
                    kwargs['best_train_loss'] = avg_train_loss
                if avg_valid_loss < kwargs.get('best_valid_loss'):
                    kwargs['best_valid_loss'] = avg_valid_loss
                if avg_valid_accuracy > kwargs.get('best_valid_accuracy'):
                    kwargs['best_valid_accuracy'] = avg_valid_accuracy
                if avg_test_accuracy > kwargs.get('best_test_accuracy'):
                    kwargs['best_test_accuracy'] = avg_test_accuracy

                # Learning rate schedule update
                if epoch >= epoch_marker + kwargs.get('patience'):
                    epoch_marker = epoch
                    utils.set_learning_rate(learning_rate_theano, model.learning_rate_update)

                # End timer
                t1 = time.time()

                # Increment the epoch
                epoch += 1
                utils.print_epoch_info(avg_train_loss, avg_valid_loss,
                                       avg_valid_accuracy, avg_test_accuracy,
                                       epoch, t1 - t0, **kwargs)

                # Break on max epochs
                if epoch >= kwargs.get('max_epochs'):
                    print('\n[train] maximum number of epochs reached\n')
                    break
            except KeyboardInterrupt:
                # We have caught the Keyboard Interrupt, figure out what resolution mode
                print('\n[train] Caught CRTL+C')
                resolution = ''
                while not (resolution.isdigit() and int(resolution) in [1, 2, 3]):
                    print('\n[train] What do you want to do?')
                    print('[train]     1 - Shock weights')
                    print('[train]     2 - Save best weights')
                    print('[train]     3 - Stop network training')
                    resolution = raw_input('[train] Resolution: ')
                resolution = int(resolution)
                # We have a resolution
                if resolution == 1:
                    # Shock the weights of the network
                    utils.shock_network(output_layer)
                    epoch_marker = epoch
                    utils.set_learning_rate(learning_rate_theano, model.learning_rate_shock)
                elif resolution == 2:
                    # Save the weights of the network
                    utils.save_model(kwargs, weights_file)
                else:
                    # Terminate the network training
                    raise KeyboardInterrupt
    except KeyboardInterrupt:
        print('\n\n[train] ...stopping network training\n')

    # Save the best network
    utils.save_model(kwargs, weights_file)


#@ibeis.register_plugin()
def train_identification_pz():
    r"""

    CommandLine:
        python -m ibeis_cnn.train --test-train_identification_pz

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis_cnn.train import *  # NOQA
        >>> train_identification_pz()
    """
    print('get_identification_decision_training_data')
    import ibeis
    ibs = ibeis.opendb('NNP_Master3')
    base_size = 64
    #max_examples = 1001
    max_examples = None
    data_file, labels_file = ibsplugin.get_identify_training_fpaths(ibs, base_size=base_size, max_examples=max_examples)

    model = models.IdentificationModel()
    config = dict(
        batch_size=8,
        learning_rate=.003,
    )
    nets_dir = ut.unixjoin(ibs.get_cachedir(), 'nets')
    ut.ensuredir(nets_dir)
    weights_file = join(nets_dir, 'ibeis_cnn_weights.pickle')
    train(data_file, labels_file, model, weights_file, **config)
    #X = k


def train_pz():
    r"""
    CommandLine:
        python -m ibeis_cnn.train --test-train_pz

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis_cnn.train import *  # NOQA
        >>> result = train_pz()
        >>> print(result)
    """
    project_name            = 'plains'
    model                   = models.PZ_Model()
    config                  = {
        'patience':   15,
        'max_epochs': 100,
        'test_time_augmentation': True,
        # 'regularization': 0.001,
    }
    root                    = abspath(join('..', 'data'))
    train_data_file         = join(root, 'numpy', project_name, 'X.npy')
    train_labels_file       = join(root, 'numpy', project_name, 'y.npy')
    results_path            = join(root, 'results', project_name)
    weights_file            = join(root, 'nets', project_name, 'ibeis_cnn_weights.pickle')
    pretrained_weights_file = join(root, 'nets', project_name, 'pretrained_weights.pickle')  # NOQA

    train(train_data_file, train_labels_file, model, weights_file, results_path, **config)
    #train(train_data_file, train_labels_file, weights_file, pretrained_weights_file)


def train_pz_girm():
    r"""
    CommandLine:
        python -m ibeis_cnn.train --test-train_pz_girm

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis_cnn.train import *  # NOQA
        >>> result = train_pz_girm()
        >>> print(result)
    """
    project_name            = 'viewpoint'
    model                   = models.PZ_GIRM_Model()
    config                  = {
        'patience':   15,
        'max_epochs': 300,
        'test_time_augmentation': True,
        # 'regularization': 0.001,
    }
    root                    = abspath(join('..', 'data'))
    train_data_file         = join(root, 'numpy', project_name, 'X.npy')
    train_labels_file       = join(root, 'numpy', project_name, 'y.npy')
    results_path            = join(root, 'results', project_name)
    weights_file            = join(root, 'nets', project_name, 'ibeis_cnn_weights.pickle')
    pretrained_weights_file = join(root, 'nets', project_name, 'pretrained_weights.pickle')  # NOQA

    train(train_data_file, train_labels_file, model, weights_file, results_path, **config)
    #train(train_data_file, train_labels_file, weights_file, pretrained_weights_file)


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis_cnn.train
        python -m ibeis_cnn.train --allexamples
        python -m ibeis_cnn.train --allexamples --noface --nosrc

    CommandLine:
        cd %CODE_DIR%/ibies_cnn/code
        cd $CODE_DIR/ibies_cnn/code
        code
        cd ibeis_cnn/code
        python train.py

    PythonPrereqs:
        pip install theano
        pip install git+https://github.com/Lasagne/Lasagne.git
        pip install git+git://github.com/lisa-lab/pylearn2.git
        #pip install lasagne
        #pip install pylearn2
        git clone git://github.com/lisa-lab/pylearn2.git
        git clone https://github.com/Lasagne/Lasagne.git
        cd pylearn2
        python setup.py develop
        cd ..
        cd Lesagne
        git checkout 8758ac1434175159e5c1f30123041799c2b6098a
        python setup.py develop
    """
    #train_pz()
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
