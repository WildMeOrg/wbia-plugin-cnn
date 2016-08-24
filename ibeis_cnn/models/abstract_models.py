# -*- coding: utf-8 -*-
"""
Directory structure of training

The network directory is the root of the structure and is typically in
_ibeis_cache/nets for ibeis databases. Otherwise it it custom defined (like in
.cache/ibeis_cnn/training for mnist tests)

# era=(group of epochs)

----------------
|-- netdir <training_dpath>
----------------

Datasets contain ingested data packed into a single file for quick loading.
Data can be presplit into testing /  learning / validation sets.  Metadata is
always a dictionary where keys specify columns and each item corresponds a row
of data. Non-corresponding metadata is currently not supported, but should
probably be located in a manifest.json file.

# TODO: what is the same data has tasks that use different labels?
# need to incorporate that structure.

The model directory must keep track of several things:
    * The network architecture (which may depend on the dataset being used)
        - input / output shape
        - network layers
    * The state of learning
        - epoch/era number
        - learning rate
        - regularization rate
    * diagnostic information
        - graphs of loss / error rates
        - images of convolutional weights
        - other visualizations

The trained model keeps track of the trained weights and is now independant of
the dataset. Finalized weights should be copied to and loaded from here.

----------------
|   |--  <training_dpath>
|   |   |-- dataset_{dataset_id} *
|   |   |   |-- full
|   |   |   |   |-- {dataset_id}_data.pkl
|   |   |   |   |-- {dataset_id}_labels.pkl
|   |   |   |   |-- {dataset_id}_labels_{task1}.pkl?
|   |   |   |   |-- {dataset_id}_labels_{task2}.pkl?
|   |   |   |   |-- {dataset_id}_metadata.pkl
|   |   |   |-- splits
|   |   |   |   |-- {split_id}_{num} *
|   |   |   |   |   |-- {dataset_id}_{split_id}_data.pkl
|   |   |   |   |   |-- {dataset_id}_{split_id}_labels.pkl
|   |   |   |   |   |-- {dataset_id}_{split_id}_metadata.pkl
|   |   |   |-- models
|   |   |   |   |-- arch_{archid} *
|   |   |   |   |   |-- best_results
|   |   |   |   |   |   |-- model_state.pkl
|   |   |   |   |   |-- checkpoints
|   |   |   |   |   |   |-- {history_id} *
|   |   |   |   |   |   |    |-- model_history.pkl
|   |   |   |   |   |   |    |-- model_state.pkl
|   |   |   |   |   |-- progress
|   |   |   |   |   |   |-- <latest>
|   |   |   |   |   |-- diagnostics
|   |   |   |   |   |   |-- {history_id} *
|   |   |   |   |   |   |   |-- <files>
|   |   |-- trained_models
|   |   |   |-- arch_{archid} *
----------------
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import six
import numpy as np
import utool as ut
from os.path import join, exists, dirname, basename, split, splitext
from six.moves import cPickle as pickle  # NOQA
import warnings
from ibeis_cnn import net_strs
from ibeis_cnn import draw_net
from ibeis_cnn.models import _model_legacy
print, rrr, profile = ut.inject2(__name__, '[ibeis_cnn.abstract_models]')


VERBOSE_CNN = ut.get_module_verbosity_flags('cnn')[0] or ut.VERBOSE

# Delayed imports
lasange = None
T = None
theano = None


def delayed_import():
    global lasagne
    global theano
    global T
    import ibeis_cnn.__LASAGNE__ as lasagne
    import ibeis_cnn.__THEANO__ as theano
    from ibeis_cnn.__THEANO__ import tensor as T  # NOQA
    # import ibeis_cnn.__LASAGNE__
    # import ibeis_cnn.__THEANO__
    # from ibeis_cnn.__THEANO__ import tensor
    # lasagne = ibeis_cnn.__LASAGNE__
    # theano = ibeis_cnn.__THEANO__
    # T = tensor


def imwrite_wrapper(show_func):
    r"""
    helper to convert show funcs into imwrite funcs
    Automatically creates filenames if not specified

    DEPRICATE
    """
    def imwrite_func(model, dpath=None, fname=None, dpi=180, asdiagnostic=True,
                     ascheckpoint=None, verbose=1, **kwargs):
        #import plottool as pt
        # Resolve path to save the image
        if dpath is None:
            if ascheckpoint is True:
                dpath = model._get_model_dpath(checkpoint_tag=model.hist_id)
            elif asdiagnostic is True:
                dpath = model.get_epoch_diagnostic_dpath()
            else:
                dpath = model.model_dpath
        # Make the image
        fig = show_func(model, **kwargs)
        # Save the image
        fpath = join(dpath, fname)
        fig.savefig(fpath, dpi=dpi)
        #output_fpath = pt.save_figure(fig=fig, dpath=dpath, dpi=dpi, verbose=verbose)
        return fpath
    return imwrite_func


@ut.reloadable_class
class LearnState(ut.DictLike):
    """
    Keeps track of parameters that can be changed during theano execution
    """
    def __init__(self, learning_rate, momentum, weight_decay):
        self._keys = [
            'momentum',
            'weight_decay',
            'learning_rate',
        ]
        self._shared_state = {key: None for key in self._keys}
        self._isinit = False
        # Set special properties
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay

    # --- special properties ---
    momentum = property(
        fget=lambda self: self.getitem('momentum'),
        fset=lambda self, val: self.setitem('momentum', val))

    learning_rate = property(
        fget=lambda self: self.getitem('learning_rate'),
        fset=lambda self, val: self.setitem('learning_rate', val))

    weight_decay = property(
        fget=lambda self: self.getitem('weight_decay'),
        fset=lambda self, val: self.setitem('weight_decay', val))

    @property
    def shared(self):
        if self._isinit:
            return self._shared_state
        else:
            raise AssertionError('Learning has not been initialized')

    def init(self):
        if not self._isinit:
            self._isinit = True
            # Reset variables with shared theano state
            _preinit_state = self._shared_state.copy()
            for key in self.keys():
                self._shared_state[key] = None
            for key in self.keys():
                self[key] = _preinit_state[key]

    def keys(self):
        return self._keys

    def getitem(self, key):
        _shared = self._shared_state[key]
        if self._isinit:
            value = None if _shared is None else _shared.get_value()
        else:
            value = _shared
        return value

    def setitem(self, key, value):
        if self._isinit:
            import ibeis_cnn.__THEANO__ as theano
            print('[model] setting %s to %.9r' % (key, value))
            _shared = self._shared_state[key]
            if value is None and _shared is not None:
                raise ValueError('Cannot set an initialized shared variable to None.')
            elif _shared is None and value is not None:
                self._shared_state[key] = theano.shared(
                    np.cast['float32'](value), name=key)
            elif _shared is not None:
                _shared.set_value(np.cast['float32'](value))
        else:
            self._shared_state[key] = value


@ut.reloadable_class
class _ModelFitter(object):
    """
    CommandLine:
        python -m ibeis_cnn _ModelFitter.fit:0
    """
    def _init_fit_vars(model, kwargs):
        model.current_era = None
        # an era is a group of epochs
        model.era_history = []
        # Training state
        model.requested_headers = ['learn_loss', 'valid_loss', 'learnval_rat']
        model.data_params = None
        # Stores current result
        model.best_results = {
            'epoch'      : None,
            'learn_loss' : np.inf,
            'valid_loss' : np.inf,
            'weights'    : None
        }
        # Dynamic configuration that may change with time
        model.learn_state = LearnState(
            learning_rate=kwargs.pop('learning_rate', .005),
            momentum=kwargs.pop('momentum', .9),
            weight_decay=kwargs.pop('weight_decay', None),
        )
        # Static configuration indicating hyper-parameters
        # (these will influence how the model learns)
        model.hyperparams = {
            'whiten_on': False,
            'augment_on': False,
            'era_size': 100,  # epochs per era
            'max_epochs': None,
            'rate_schedule': .8,
        }
        # Static configuration indicating training preferences
        # (these will not influence the model learning)
        model.monitor_config = {
            'monitor_updates': False,
            'checkpoint_freq': 200,
            'slowdump_freq': 10,
            'monitor': ut.get_argflag('--monitor'),
            'showprog': True,
        }
        # This will by a dynamic dict that will span the life of a training
        # session
        model._fit_session = None

    def fit(model, X_train, y_train, X_valid=None, y_valid=None, valid_idx=None):
        r"""
        Trains the network with backprop.

        CommandLine:
            python -m ibeis_cnn _ModelFitter.fit --name=dropout
            python -m ibeis_cnn _ModelFitter.fit --name=bnorm
            python -m ibeis_cnn _ModelFitter.fit --name=incep

        Example1:
            >>> from ibeis_cnn.models import mnist
            >>> name = ut.get_argval('--name', default='bnorm')
            >>> model, dataset = mnist.testdata_mnist(name=name, dropout=.5)
            >>> model.initialize_architecture()
            >>> model.print_layer_info()
            >>> model.print_model_info_str()
            >>> X_train, y_train = dataset.subset('train')
            >>> model.fit(X_train, y_train)
        """
        from ibeis_cnn import utils
        print('\n[train] --- TRAINING LOOP ---')
        model._validate_input(X_train, y_train)

        X_learn, y_learn, X_valid, y_valid = model._ensure_learnval_split(
            X_train, y_train, X_valid, y_valid, valid_idx)

        model.ensure_data_params(X_learn, y_learn)
        print('Learn y histogram: ' + ut.repr2(ut.dict_hist(y_learn)))
        print('Valid y histogram: ' + ut.repr2(ut.dict_hist(y_valid)))

        model._new_fit_session()

        epoch = model.best_results['epoch']
        if epoch is None:
            epoch = 0
            print('Initializng training at epoch=%r' % (epoch,))
        else:
            print('Resuming training at epoch=%r' % (epoch,))
        # Begin training the neural network
        print('model.batch_size = %r' % (model.batch_size,))
        print('model.hyperparams = %s' % (ut.repr4(model.hyperparams),))
        print('learn_state = %s' % ut.repr4(model.learn_state.asdict()))

        # create theano symbolic expressions that define the network
        theano_backprop = model.build_backprop_func()
        theano_forward = model.build_forward_func()

        # number of non-best iterations after, that triggers a best save
        # This prevents strings of best-saves one after another
        save_after_best_wait_epochs = model.hyperparams['era_size'] * 2
        save_after_best_countdown = None

        printcol_info = utils.get_printcolinfo(model.requested_headers)

        model._new_era(X_learn, y_learn, X_valid, y_valid)
        utils.print_header_columns(printcol_info)

        tt = ut.Timer(verbose=False)
        while True:
            try:
                # ---------------------------------------
                # Execute backwards and forward passes
                tt.tic()
                learn_info = model._epoch_learn_step(theano_backprop, X_learn,
                                                     y_learn)
                if learn_info.get('diverged'):
                    break
                valid_info = model._epoch_validate_step(theano_forward,
                                                        X_valid, y_valid)

                # ---------------------------------------
                # Summarize the epoch
                epoch_info = {
                    'epoch': epoch,
                }
                epoch_info.update(**learn_info)
                epoch_info.update(**valid_info)
                epoch_info['duration'] = tt.toc()
                epoch_info['learnval_rat'] = (
                    epoch_info['learn_loss'] / epoch_info['valid_loss'])

                # ---------------------------------------
                # Record this epoch in history
                model._record_epoch(epoch_info)

                # ---------------------------------------
                # Check how we are learning
                if epoch_info['valid_loss'] < model.best_results['valid_loss']:
                    model.best_results['weights'] = model.get_all_param_values()
                    model.best_results['epoch'] = epoch_info['epoch']
                    for key in model.requested_headers:
                        model.best_results[key] = epoch_info[key]
                    save_after_best_countdown = save_after_best_wait_epochs

                # Check frequencies and countdowns
                checkpoint_flag = utils.checkfreq(
                    model.monitor_config['checkpoint_freq'], epoch)
                if save_after_best_countdown is not None:
                    if save_after_best_countdown == 0:
                        ## Callbacks on best found
                        save_after_best_countdown = None
                        checkpoint_flag = True
                    else:
                        save_after_best_countdown -= 1

                # ---------------------------------------
                # Output Diagnostics

                # Print the epoch
                utils.print_epoch_info(model, printcol_info, epoch_info)

                # Output any diagnostics
                if checkpoint_flag:
                    # FIXME: just move it to the second location
                    model.checkpoint_save_model_info()
                    model.checkpoint_save_model_state()
                    model.save_model_info()
                    model.save_model_state()

                if model.monitor_config['monitor']:
                    model._dump_epoch_monitor()

                if model.monitor_config['monitor']:
                    if utils.checkfreq(model.monitor_config['slowdump_freq'], epoch):
                        model._dump_slow_monitor()

                # Check if the era is done
                max_era_size = model._fit_session['max_era_size']
                if model.current_era['size'] >= max_era_size:
                    # Decay learning rate
                    era = model.total_eras
                    rate_schedule = model.hyperparams['rate_schedule']
                    rate_schedule = ut.ensure_iterable(rate_schedule)
                    frac = rate_schedule[min(era, len(rate_schedule) - 1)]
                    learn_state = model.learn_state
                    learn_state.learning_rate = (
                        learn_state.learning_rate * frac)
                    # Increase number of epochs in the next era
                    max_era_size = np.ceil(max_era_size / (frac ** 2))
                    model._fit_session['max_era_size'] = max_era_size
                    # Start a new era
                    model._new_era(X_learn, y_learn, X_valid, y_valid)
                    utils.print_header_columns(printcol_info)

                # Break on max epochs
                if model.hyperparams['max_epochs'] is not None:
                    if epoch >= model.hyperparams['max_epochs']:
                        print('\n[train] maximum number of epochs reached\n')
                        break
                # Increment the epoch
                epoch += 1

            except KeyboardInterrupt:
                print('\n[train] Caught CRTL+C')
                resolution = ''
                while True:
                    print('\n[train] What do you want to do?')
                    print('[train]     0 - Continue')
                    print('[train]     1 - Shock weights')
                    print('[train]     2 - Save best weights')
                    print('[train]     3 - View session directory')
                    print('[train]     4 - Embed into IPython')
                    print('[train]     5 - Draw current weights')
                    print('[train]     6 - Show training state')
                    print('[train]     q - Stop network training')
                    resolution = str(input('[train] Resolution: '))
                    # We have a resolution
                    if resolution == 'q':
                        print('quit training...')
                    if resolution == '0':
                        print('resuming training...')
                    elif resolution == '1':
                        # Shock the weights of the network
                        utils.shock_network(model.output_layer)
                        learn_state = model.learn_state
                        learn_state.learning_rate = learn_state.learning_rate * 2
                        #epoch_marker = epoch
                        utils.print_header_columns(printcol_info)
                    elif resolution == '2':
                        # Save the weights of the network
                        model.checkpoint_save_model_info()
                        model.checkpoint_save_model_state()
                        model.save_model_info()
                        model.save_model_state()
                    elif resolution == '3':
                        session_dpath = model._fit_session['session_dpath']
                        ut.view_directory(session_dpath)
                    elif resolution == '4':
                        ut.embed()
                    elif resolution == '5':
                        output_fpath_list = model.imwrite_weights(index=0)
                        for fpath in output_fpath_list:
                            ut.startfile(fpath)
                    elif resolution == '6':
                        model.print_state_str()
                    else:
                        continue
                    # Handled the resolution
                    break
        # Save the best network
        model.checkpoint_save_model_state()
        model.save_model_state()

    def _ensure_learnval_split(model, X_train, y_train, X_valid=None,
                               y_valid=None, valid_idx=None):
        if X_valid is not None:
            assert valid_idx is None, 'Cant specify both valid_idx and X_valid'
            # When X_valid is given assume X_train is actually X_learn
            X_learn = X_train
            y_learn = y_train
        else:
            if valid_idx is None:
                # Split training set into a learning / validation set
                from ibeis_cnn.dataset import stratified_shuffle_split
                train_idx, valid_idx = stratified_shuffle_split(
                    y_train, fractions=[.7, .3], rng=432321)
                #import sklearn.cross_validation
                #xvalkw = dict(n_folds=2, shuffle=True, random_state=43432)
                #skf = sklearn.cross_validation.StratifiedKFold(y_train, **xvalkw)
                #train_idx, valid_idx = list(skf)[0]
            elif valid_idx is None and X_valid is None:
                train_idx = ut.index_complement(valid_idx, len(X_train))
            else:
                assert False, 'impossible state'
            # Set to learn network weights
            X_learn = X_train.take(train_idx, axis=0)
            y_learn = y_train.take(train_idx, axis=0)
            # Set to crossvalidate hyperparamters
            X_valid = X_train.take(valid_idx, axis=0)
            y_valid = y_train.take(valid_idx, axis=0)
        # print('\n[train] --- MODEL INFO ---')
        # model.print_architecture_str()
        # model.print_layer_info()
        return X_learn, y_learn, X_valid, y_valid

    def ensure_data_params(model, X_learn, y_learn):
        # TODO: move to dataset. This is independant of the model.
        if model.hyperparams['whiten_on']:
            # Center the data by subtracting the mean
            if model.data_params is None:
                model.data_params = {}
                print('computing center mean/std. (hacks std=1)')
                X_ = X_learn.astype(np.float32)
                if ut.is_int(X_learn):
                    ut.assert_inbounds(X_learn, 0, 255, eq=True,
                                       verbose=ut.VERBOSE)
                    X_ = X_ / 255
                ut.assert_inbounds(X_, 0.0, 1.0, eq=True,
                                   verbose=ut.VERBOSE)
                # Ensure that the mean is computed on 0-1 normalized data
                model.data_params['center_mean'] = np.mean(X_, axis=0)
                model.data_params['center_std'] = 1.0

            # Hack to preconvert mean / std to 0-1 for old models
            model._fix_center_mean_std()
        else:
            model.data_params = None

        if getattr(model, 'encoder', None) is None:
            if hasattr(model, 'initialize_encoder'):
                model.initialize_encoder(y_learn)

    def _new_fit_session(model):
        """
        Starts a model training session
        """
        model._fit_session = {
            'start_time': ut.get_timestamp(),
            'max_era_size': model.hyperparams['era_size'],
            'era_epoch_num': 0,
        }
        ut.ensuredir(model.arch_dpath)
        # Write feed-forward arch info to arch root
        ut.writeto(join(model.arch_dpath, 'arch_info.json'),
                   model.make_arch_json(with_noise=False), verbose=True)

        if model.monitor_config['monitor']:
            # Create a directory for this training session with a timestamp
            session_dname = 'fit_session_' + model._fit_session['start_time']
            session_dpath = join(model.saved_session_dpath, session_dname)
            session_dpath = ut.get_nonconflicting_path(session_dpath, offset=1,
                                                       suffix='_conflict%d')
            ut.symlink(session_dpath, join(model.arch_dpath, 'latest_session'))

            ut.ensuredir(session_dpath)
            model._fit_session['session_dpath'] = session_dpath

            loss_progress_dir = join(session_dpath, 'history')
            weights_progress_dir = join(session_dpath, 'weights')
            dream_progress_dir = join(session_dpath, 'dream')

            history_text_fpath = join(session_dpath, 'era_history.txt')
            back_archinfo_fpath = join(session_dpath, 'fit_arch_info.json')
            back_archimg_fpath = join(session_dpath, 'fit_arch_graph.jpg')

            # Write backprop arch info to arch root
            back_archinfo_json = model.make_arch_json(with_noise=True)
            ut.writeto(back_archinfo_fpath, back_archinfo_json, verbose=True)

            # Write arch graph to root
            try:
                model.imwrite_arch(fpath=back_archimg_fpath)
                model._overwrite_latest_image(back_archimg_fpath, 'arch_graph')
            except Exception as ex:
                ut.printex(ex, iswarning=True)

            progress_dirs = {
                'dream': dream_progress_dir,
                'loss': loss_progress_dir,
                'weights': weights_progress_dir,
            }

            model._fit_session.update(**{
                'progress_dirs': progress_dirs,
                'history_text_fpath': history_text_fpath,
            })

            for dpath in progress_dirs.values():
                ut.ensuredir(dpath)

            if ut.get_argflag('--vd'):
                ut.vd(session_dpath)

            # Write initial states of the weights
            try:
                fpath = model.imwrite_weights(
                    dpath=weights_progress_dir,
                    fname='weights_' + model.hist_id + '.png',
                    fnum=2, verbose=0)
                model._overwrite_latest_image(fpath, 'weights')
            except Exception as ex:
                ut.printex(ex, iswarning=True)

    def _overwrite_latest_image(model, fpath, new_name):
        """
        copies the new image to a path to be overwritten so new updates are
        shown
        """
        import shutil
        dpath, fname = split(fpath)
        ext = splitext(fpath)[1]
        session_dpath = model._fit_session['session_dpath']
        # Copy latest image to the session dir and the main arch dir
        shutil.copy(fpath, join(session_dpath, 'latest_' + new_name + ext))
        shutil.copy(fpath, join(model.arch_dpath, 'latest_' + new_name + ext))

    def _dump_slow_monitor(model):
        progress_dirs = model._fit_session['progress_dirs']
        try:
            # Save class dreams
            _fn = ut.get_method_func(model.show_class_dream)
            fpath = imwrite_wrapper(_fn)(
                model,
                dpath=progress_dirs['dream'],
                fname='dream_' + model.hist_id + '.png',
                fnum=4, verbose=0)
            model._overwrite_latest_image(fpath, 'class_dream')
        except Exception as ex:
            ut.printex(ex, iswarning=True)

        try:
            # Save weights images
            fpath = model.imwrite_weights(dpath=progress_dirs['weights'],
                                          fname='weights_' + model.hist_id + '.png',
                                          fnum=2, verbose=0)
            model._overwrite_latest_image(fpath, 'weights')
        except Exception as ex:
            ut.printex(ex, iswarning=True)

    def _dump_epoch_monitor(model):
        progress_dirs = model._fit_session['progress_dirs']

        # Save text info
        text_fpath = model._fit_session['history_text_fpath']
        history_text = ut.list_str(model.era_history, newlines=True)
        ut.write_to(text_fpath, history_text, verbose=False)

        # Save loss graphs
        try:
            fpath = imwrite_wrapper(ut.get_method_func(model.show_era_loss_history))(
                model,
                dpath=progress_dirs['loss'],
                fname='loss_' + model.hist_id + '.png',
                fnum=1, verbose=0)
            model._overwrite_latest_image(fpath, 'loss')
        except Exception as ex:
            ut.printex(ex, iswarning=True)

        # Save weight updates
        try:
            _fn = ut.get_method_func(model.show_era_update_mag_history)
            fpath = imwrite_wrapper(_fn)(
                model,
                dpath=progress_dirs['loss'],
                fname='update_mag_' + model.hist_id + '.png',
                fnum=3, verbose=0)
            model._overwrite_latest_image(fpath, 'update_mag')
        except Exception as ex:
            ut.printex(ex, iswarning=True)

    def _new_era(model, X_learn, y_learn, X_valid, y_valid):
        """
        Used to denote a change in hyperparameters during training.
        """
        # TODO: fix the training data hashid stuff
        y_hashid = ut.hashstr_arr(y_learn, 'y', alphabet=ut.ALPHABET_27)

        learn_hashid =  str(model.arch_id) + '_' + y_hashid
        # learn_hashid =  str(model.arch_tag) + '_' + y_hashid
        if model.current_era is not None and len(model.current_era['epoch_list']) == 0:
            print('Not starting new era (old one hasnt begun yet')
        else:
            _new_era = {
                'learn_hashid': learn_hashid,
                'arch_hashid': model.get_arch_hashid(),
                'arch_id': model.arch_id,
                'num_learn': len(y_learn),
                'num_valid': len(y_valid),
                'valid_loss_list': [],
                'learn_loss_list': [],
                'epoch_list': [],
                'learn_state': model.learn_state.asdict(),
                'size': 0,
            }
            num_eras = len(model.era_history)
            print('starting new era %d' % (num_eras,))
            #if model.current_era is not None:
            model.current_era = _new_era
            model.era_history.append(model.current_era)

    def _record_epoch(model, epoch_info):
        """
        Records an epoch in an era.
        """
        # each key/val in an epoch_info dict corresponds to a key/val_list in
        # an era dict.
        model.current_era['size'] += 1
        for key in epoch_info:
            key_ = key + '_list'
            if key_ not in model.current_era:
                model.current_era[key_] = []
            model.current_era[key_].append(epoch_info[key])

    def _epoch_learn_step(model, theano_backprop, X_learn, y_learn):
        """
        Backwards propogate -- Run learning set through the backwards pass
        """
        # from ibeis_cnn import batch_processing as batch

        learn_outputs = model.process_batch(theano_backprop, X_learn, y_learn,
                                            shuffle=True,
                                            augment_on=model.hyperparams['augment_on'],
                                            buffered=True)

        # compute the loss over all learning batches
        learn_info = {}
        learn_info['learn_loss'] = learn_outputs['loss'].mean()

        if 'loss_regularized' in learn_outputs:
            # Regularization information
            learn_info['learn_loss_regularized'] = (
                learn_outputs['loss_regularized'].mean())
            #if 'valid_acc' in model.requested_headers:
            #    learn_info['test_acc']  = learn_outputs['accuracy']
            regularization_amount = (
                learn_outputs['loss_regularized'] - learn_outputs['loss'])
            regularization_ratio = (
                regularization_amount / learn_outputs['loss'])
            regularization_percent = (
                regularization_amount / learn_outputs['loss_regularized'])

            learn_info['regularization_percent'] = regularization_percent
            learn_info['regularization_ratio'] = regularization_ratio

        param_update_mags = {}
        for key, val in learn_outputs.items():
            if key.startswith('param_update_magnitude_'):
                key_ = key.replace('param_update_magnitude_', '')
                param_update_mags[key_] = (val.mean(), val.std())
        if param_update_mags:
            learn_info['param_update_mags'] = param_update_mags

        # If the training loss is nan, the training has diverged
        if np.isnan(learn_info['learn_loss']):
            #import utool
            #utool.embed()
            print('\n[train] train loss is Nan. training diverged\n')
            print('learn_outputs = %r' % (learn_outputs,))
            print('\n[train] train loss is Nan. training diverged\n')
            """
            from ibeis_cnn import draw_net
            draw_net.imwrite_theano_symbolic_graph(theano_backprop)
            """
            # imwrite_theano_symbolic_graph(thean_expr):
            learn_info['diverged'] = True
        return learn_info

    def _epoch_validate_step(model, theano_forward, X_valid, y_valid):
        """
        Forwards propogate -- Run validation set through the forwards pass
        """
        valid_outputs = model.process_batch(theano_forward, X_valid, y_valid)
        valid_info = {}
        valid_info['valid_loss'] = valid_outputs['loss_determ'].mean()
        valid_info['valid_loss_std'] = valid_outputs['loss_determ'].std()
        if 'valid_acc' in model.requested_headers:
            valid_info['valid_acc'] = valid_outputs['accuracy'].mean()
        return valid_info


@ut.reloadable_class
class _BatchUtility(object):

    @classmethod
    def expand_data_indicies(cls, label_idx, data_per_label=1):
        """
        when data_per_label > 1, gives the corresponding data indicies for the
        data indicies
        """
        expanded_idx = [label_idx * data_per_label + count
                        for count in range(data_per_label)]
        data_idx = np.vstack(expanded_idx).T.flatten()
        return data_idx

    @classmethod
    def shuffle_input(cls, X, y, data_per_label=1, rng=None):
        rng = ut.ensure_rng(rng)
        num_labels = X.shape[0] // data_per_label
        label_idx = ut.random_indexes(num_labels, rng=rng)
        data_idx = cls.expand_data_indicies(label_idx, data_per_label)
        X = X.take(data_idx, axis=0)
        X = np.ascontiguousarray(X)
        if y is not None:
            y = y.take(label_idx, axis=0)
            y = np.ascontiguousarray(y)
        return X, y

    @classmethod
    def slice_batch(cls, X, y, batch_size, batch_index, data_per_label,
                    wraparound=False):
        start_x = batch_index * batch_size
        end_x = (batch_index + 1) * batch_size
        # Take full batch of images and take the fraction of labels if
        # data_per_label > 1
        x_sl = slice(start_x, end_x)
        y_sl = slice(start_x // data_per_label, end_x // data_per_label)
        Xb = X[x_sl]
        yb = y if y is None else y[y_sl]
        if wraparound:
            # Append extra data to ensure the batch size is full
            if Xb.shape[0] != batch_size:
                extra = batch_size - Xb.shape[0]
                Xb_wrap = X[slice(0, extra)]
                Xb = np.concatenate([Xb, Xb_wrap], axis=0)
                if yb is not None:
                    yb_wrap = y[slice(0, extra // data_per_label)]
                    yb = np.concatenate([yb, yb_wrap], axis=0)
        return Xb, yb

    def _pad_labels(model, yb):
        """
        # TODO: FIX data_per_label_input ISSUES
        # most models will do the padding implicitly
        # in the layer architecture
        """
        pad_size = len(yb) * (model.data_per_label_input - 1)
        yb_buffer = -np.ones(pad_size, dtype=yb.dtype)
        yb = np.hstack((yb, yb_buffer))
        return yb

    def prepare_data(model, X):
        is_int = ut.is_int(X)
        is_cv2 = model.X_is_cv2_native
        whiten_on = model.hyperparams['whiten_on']
        Xb, _ = model._prepare_batch(X, None, is_int=is_int, is_cv2=is_cv2,
                                     whiten_on=whiten_on)
        return Xb

    def _stack_outputs(model, theano_fn, output_list):
        """
        Combines outputs across batches and returns them in a dictionary keyed
        by the theano variable output name.
        """
        import vtool as vt
        output_vars = [outexpr.variable for outexpr in theano_fn.outputs]
        output_names = [str(var) if var.name is None else var.name
                        for var in output_vars]
        unstacked_output_gen = [[bop[count] for bop in output_list]
                                for count, name in enumerate(output_names)]
        stacked_output_list  = [
            vt.safe_cat(_output_unstacked, axis=0)
            for _output_unstacked in unstacked_output_gen
        ]
        outputs = dict(zip(output_names, stacked_output_list))
        return outputs

    def _unwrap_outputs(model, outputs, X):
        # batch iteration may wrap-around returned data.
        # slice off the padding
        num_inputs = X.shape[0] / model.data_per_label_input
        num_outputs = num_inputs * model.data_per_label_output
        for key in outputs.keys():
            outputs[key] = outputs[key][0:num_outputs]
        return outputs


@ut.reloadable_class
class _ModelBatch(_BatchUtility):

    def _init_batch_vars(model, kwargs):
        model.pad_labels = False
        model.X_is_cv2_native = True

    def process_batch(model, theano_fn, X, y=None, buffered=False,
                      unwrap=False, shuffle=False, augment_on=False):
        """ Execute a theano function on batches of X and y """
        # Break data into generated batches
        # TODO: sliced batches when there is no shuffling
        # Create an iterator to generate batches of data
        batch_iter = model.batch_iterator(X, y, shuffle=shuffle,
                                          augment_on=augment_on)
        if buffered:
            batch_iter = ut.buffered_generator(batch_iter)
        if model.monitor_config['showprog']:
            num_batches = (X.shape[0] + model.batch_size - 1) // model.batch_size
            batch_iter = ut.ProgressIter(
                batch_iter, nTotal=num_batches, lbl=theano_fn.name, freq=10,
                bs=True, adjust=True)

        # Execute the function with either known or unknown y-targets
        output_list = []
        if y is None:
            aug_yb_list = None
            for Xb, yb in batch_iter:
                batch_output = theano_fn(Xb)
                output_list.append(batch_output)
        else:
            aug_yb_list = []
            for Xb, yb in batch_iter:
                batch_output = theano_fn(Xb, yb)
                output_list.append(batch_output)
                aug_yb_list.append(yb)

        # Combine results of batches into one big result
        outputs = model._stack_outputs(theano_fn, output_list)
        if y is not None:
            # Hack in special outputs
            auglbl_list = np.hstack(aug_yb_list)
            outputs['auglbl_list'] = auglbl_list
        if unwrap:
            # slice of batch induced padding
            outputs = model._unwrap_outputs(outputs, X)
        return outputs

    @profile
    def batch_iterator(model, X, y, shuffle=False, augment_on=False):
        """
        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis_cnn.models.abstract_models import *  # NOQA
            >>> from ibeis_cnn import models
            >>> model = models.DummyModel(batch_size=16)
            >>> X, y = model.make_random_testdata(num=37, cv2_format=True)
            >>> model.ensure_data_params(X, y)
            >>> result_list = [(Xb, Yb) for Xb, Yb in model.batch_iterator(X, y)]
            >>> Xb, yb = result_list[0]
            >>> assert np.all(X[0, :, :, 0] == Xb[0, 0, :, :])
            >>> result = ut.depth_profile(result_list, compress_consecutive=True)
            >>> print(result)
            (7, [(16, 1, 4, 4), 16])

        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis_cnn.models.abstract_models import *  # NOQA
            >>> from ibeis_cnn import models
            >>> model = models.DummyModel(batch_size=16)
            >>> X, y = model.make_random_testdata(num=37, cv2_format=False, asint=True)
            >>> model.X_is_cv2_native = False
            >>> model.ensure_data_params(X, y)
            >>> result_list = [(Xb, Yb) for Xb, Yb in model.batch_iterator(X, y)]
            >>> Xb, yb = result_list[0]
            >>> assert np.all(np.isclose(X[0] / 255, Xb[0]))
            >>> result = depth
            >>> print(result)
        """
        # need to be careful with batchsizes if directly specified to theano
        batch_size = model.batch_size
        data_per_label = model.data_per_label_input
        wraparound = model.input_shape[0] is not None
        model._validate_labels(X, y)

        num_batches = (X.shape[0] + batch_size - 1) // batch_size

        if shuffle:
            X, y = model.shuffle_input(X, y, data_per_label)

        is_int = ut.is_int(X)
        is_cv2 = model.X_is_cv2_native
        whiten_on = model.data_params is not None

        # Slice and preprocess data in batch
        for batch_index in range(num_batches):
            # Take a slice from the data
            Xb, yb = model.slice_batch(X, y, batch_size, batch_index,
                                       data_per_label, wraparound)
            # Prepare data for the GPU
            Xb, yb = model._prepare_batch(Xb, yb, is_int=is_int, is_cv2=is_cv2,
                                          augment_on=augment_on,
                                          whiten_on=whiten_on)
            yield Xb, yb

    def _prepare_batch(model, Xb, yb, is_int=True, is_cv2=True,
                       augment_on=False, whiten_on=False):
        Xb = Xb.astype(np.float32, copy=True)
        yb = None if yb is None else yb.astype(np.int32, copy=True)
        if is_int:
            # Rescale the batch data to the range 0 to 1
            Xb = Xb / 255.0
        if augment_on:
            Xb, yb = model.augment(Xb, yb)
        if whiten_on:
            mean = model.data_params['center_mean']
            std  = model.data_params['center_std']
            assert np.all(mean <= 1.0)
            assert np.all(std <= 1.0)
            np.subtract(Xb, mean, out=Xb)
            np.divide(Xb, std, out=Xb)
            #Xb = (Xb - mean) / (std)
        if is_cv2:
            # Convert from cv2 to lasange format
            Xb = Xb.transpose((0, 3, 1, 2))
        if yb is not None:
            # if encoder is not None:
            #     # Apply an encoding if applicable
            #     yb = encoder.transform(yb).astype(np.int32)
            if model.data_per_label_input > 1 and model.pad_labels:
                # Pad data for siamese networks
                yb = model._pad_labels(yb)
        return Xb, yb


class _ModelPredicter(object):

    def _predict(model, X_test):
        """
        Returns all prediction outputs of the network in a dictionary.
        """
        # from ibeis_cnn import batch_processing as batch
        if ut.VERBOSE:
            print('\n[train] --- MODEL INFO ---')
            model.print_architecture_str()
            model.print_layer_info()
        # create theano symbolic expressions that define the network
        theano_predict = model.build_predict_func()
        # Begin testing with the neural network
        print('\n[test] predict with batch size %0.1f' % (
            model.batch_size))
        test_outputs = model.process_batch(theano_predict, X_test, unwrap=True)
        return test_outputs

    def predict_proba(model, X_test):
        test_outputs = model._predict(X_test)
        y_proba = test_outputs['network_output_determ']
        return y_proba

    def predict_proba_Xb(model, Xb):
        """ Accepts prepared inputs """
        theano_predict = model.build_predict_func()
        batch_output = theano_predict(Xb)
        output_names = [
            str(outexpr.variable)
            if outexpr.variable.name is None else
            outexpr.variable.name
            for outexpr in theano_predict.outputs
        ]
        test_outputs = dict(zip(output_names, batch_output))
        y_proba = test_outputs['network_output_determ']
        return y_proba

    def predict(model, X_test):
        test_outputs = model._predict(X_test)
        y_predict = test_outputs['predictions']
        return y_predict


@ut.reloadable_class
class _ModelVisualization(object):
    """
    """

    def show_arch(model, fnum=None, fullinfo=True, **kwargs):
        import plottool as pt
        layers = model.get_all_layers(**kwargs)
        draw_net.show_arch_nx_graph(layers, fnum=fnum, fullinfo=fullinfo)
        pt.set_figtitle(model.arch_id)

    def show_class_dream(model, fnum=None, **kwargs):
        """
        CommandLine:
            python -m ibeis_cnn.models.abstract_models show_class_dream --show

        Example:
            >>> # DISABLE_DOCTEST
            >>> # Assumes mnist is trained
            >>> from ibeis_cnn.draw_net import *  # NOQA
            >>> from ibeis_cnn.models import mnist
            >>> model, dataset = mnist.testdata_mnist(dropout=.5)
            >>> model.initialize_architecture()
            >>> model.load_model_state()
            >>> ut.quit_if_noshow()
            >>> import plottool as pt
            >>> #pt.qt4ensure()
            >>> model.show_class_dream()
            >>> ut.show_if_requested()
        """
        import plottool as pt
        kw = dict(init='randn', niters=200, update_rate=.05, weight_decay=1e-4)
        kw.update(**kwargs)
        target_labels = list(range(model.output_dims))
        dream = getattr(model, '_dream', None)
        if dream is None:
            dream = draw_net.Dream(model, **kw)
            model._dream = dream
        images = dream.make_class_images(target_labels)
        pnum_ = pt.make_pnum_nextgen(nSubplots=len(target_labels))
        fnum = pt.ensure_fnum(fnum)
        fig = pt.figure(fnum=fnum)
        for target_label, img in zip(target_labels, images):
            pt.imshow(img, fnum=fnum, pnum=pnum_(), title='target=%r' % (target_label,))
        return fig

    def dump_class_dream(model, fpath):
        """
        initial =
        """
        dpath = model._fit_session['session_dpath']
        dpath = join(dpath, 'dreamvid')
        dataset = ut.search_stack_for_localvar('dataset')
        X_test, y_test = dataset.subset('valid')

        import ibeis_cnn.__THEANO__ as theano
        from ibeis_cnn.__THEANO__ import tensor as T  # NOQA
        import utool as ut
        num = 64
        Xb = model.prepare_data(X_test[0:num])
        yb = y_test[0:num]
        shared_images = theano.shared(Xb.astype(np.float32))
        dream = model._dream
        step_fn = dream._make_objective(shared_images, yb)
        out = dream._postprocess_class_image(shared_images, yb,
                                             was_scalar=True)
        import vtool as vt
        count = 0
        for _ in range(100):
            out = dream._postprocess_class_image(shared_images, yb,
                                                 was_scalar=True)
            vt.imwrite(join(dpath, 'out%d.jpg' % (count,)), out)
            count += 1
            for _ in ut.ProgIter(range(10)):
                step_fn()

    def show_era_update_mag_history(model, fnum=None):
        """
        CommandLine:
            python -m ibeis_cnn --tf _ModelVisualization.show_era_update_mag_history --show

        Example:
            >>> # DISABLE_DOCTEST
            >>> from ibeis_cnn.models.abstract_models import *  # NOQA
            >>> model = testdata_model_with_history()
            >>> fnum = None
            >>> model.show_era_update_mag_history(fnum)
            >>> ut.show_if_requested()
        """
        import plottool as pt
        fnum = pt.ensure_fnum(fnum)

        layer_info = model.get_all_layer_info()
        param_groups = [[p['name'] for p in info['param_infos']
                         if 'trainable' in p['tags']] for info in layer_info]
        param_groups = ut.compress(param_groups, param_groups)

        fig = pt.figure(fnum=fnum, pnum=(1, 1, 1), doclf=True, docla=True)
        next_pnum = pt.make_pnum_nextgen(nSubplots=1 + len(param_groups))

        if len(model.era_history) > 0:
            for param_keys in param_groups:
                model.show_weight_updates(param_keys=param_keys, fnum=fnum, pnum=next_pnum())

        # Show learning rate
        labels = None
        if len(model.era_history) > 0:
            #era = model.era_history[0]
            labels = {'rate': 'learning rate'}
        ydatas = [
            {
                'rate': np.array([era['learn_state']['learning_rate']] * len(era['epoch_list'])),
            }
            for era in model.era_history
        ]
        model._show_era_measure(ydatas, labels=labels, ylabel='learning rate',
                                yscale='linear', fnum=fnum, pnum=next_pnum())

        # TODO: dont do this if off
        #model.show_weight_updates(fnum=fnum, pnum=next_pnum())
        # TODO: add confusion
        pt.set_figtitle('Weight Update History: ' + model.hist_id + '\n' + model.arch_id)
        return fig

    def show_era_loss_history(model, fnum=None):
        r"""
        Args:
            fnum (int):  figure number(default = None)

        Returns:
            mpl.Figure: fig

        CommandLine:
            python -m ibeis_cnn --tf _ModelVisualization.show_era_loss_history --show

        Example:
            >>> # DISABLE_DOCTEST
            >>> from ibeis_cnn.models.abstract_models import *  # NOQA
            >>> model = testdata_model_with_history()
            >>> fnum = None
            >>> model.show_era_loss_history(fnum)
            >>> ut.show_if_requested()
        """
        import plottool as pt
        fnum = pt.ensure_fnum(fnum)
        fig = pt.figure(fnum=fnum, pnum=(1, 1, 1), doclf=True, docla=True)
        next_pnum = pt.make_pnum_nextgen(nRows=2, nCols=2)
        if len(model.era_history) > 0:
            model.show_era_loss(fnum=fnum, pnum=next_pnum(), yscale='log')
            model.show_era_loss(fnum=fnum, pnum=next_pnum(), yscale='linear')
            model.show_era_acc(fnum=fnum, pnum=next_pnum(), yscale='linear')
            model.show_era_lossratio(fnum=fnum, pnum=next_pnum())

        # TODO: dont do this if off
        #model.show_weight_updates(fnum=fnum, pnum=next_pnum())
        pt.set_figtitle('Era Loss History: ' + model.hist_id + '\n' + model.arch_id)
        return fig

    def show_era_lossratio(model, **kwargs):
        labels = None
        if len(model.era_history) > 0:
            labels = {'ratio': 'learn/valid'}
        ydatas = [
            {
                'ratio': np.divide(era['learn_loss_list'],
                                   era['valid_loss_list'])
            }
            for era in model.era_history
        ]
        return model._show_era_measure(ydatas, labels, ylabel='learn/valid', yscale='linear', **kwargs)

    def show_era_acc(model, **kwargs):
        styles = {'valid': '-o'}
        labels = None
        if len(model.era_history) > 0:
            era = model.era_history[0]
            lastacc = era['valid_acc_list'][-1]
            labels = {
                'valid': 'valid acc ' + str(era['num_valid']) + ' ' + ut.repr2(lastacc * 100, precision=2) + '%',
            }
        ydatas = [
            {
                'valid': era_['valid_acc_list'],
            }
            for era_ in model.era_history
        ]
        fig = model._show_era_measure(ydatas, labels,  styles, ylabel='accuracy',
                                       **kwargs)
        #import plottool as pt
        #ax = pt.gca()
        #ymin, ymax = ax.get_ylim()
        #pt.gca().set_ylim((ymin, 100))
        return fig

    def show_era_loss(model, **kwargs):
        styles = {'learn': '-x', 'valid': '-o'}
        labels = None
        if len(model.era_history) > 0:
            labels = {
                'learn': 'learn ' + str(model.era_history[0]['num_learn']),
                'valid': 'valid ' + str(model.era_history[0]['num_valid']),
            }
        ydatas = [
            {
                'learn': era['learn_loss_list'],
                'valid': era['valid_loss_list'],
            }
            for era in model.era_history
        ]
        return model._show_era_measure(ydatas, labels,  styles, ylabel='loss',
                                       **kwargs)

    def show_weight_updates(model, param_keys=None, **kwargs):
        # has_mag_updates = False
        import plottool as pt
        xdatas = []
        ydatas = []
        yspreads = []
        labels = None
        colors = None

        if len(model.era_history) > 0:
            era = model.era_history[-1]
            if 'param_update_mags_list' in era:
                if param_keys is None:
                    param_keys = list(set(ut.flatten([
                        dict_.keys() for dict_ in era['param_update_mags_list']
                    ])))
                labels = dict(zip(param_keys, param_keys))
                colors = dict(zip(param_keys, pt.distinct_colors(len(param_keys))))

        # colors = {}
        for index, era in enumerate(model.era_history):
            if 'param_update_mags_list' not in era:
                continue
            xdata = era['epoch_list']
            if index == 0:
                xdata = xdata[1:]
            xdatas.append(xdata)
            update_mags_list = era['param_update_mags_list']
            # Transpose keys and positions
            #param_keys = list(set(ut.flatten([
            #    dict_.keys() for dict_ in update_mags_list
            #])))
            param_val_list = [
                [list_[param] for list_ in update_mags_list]
                for param in param_keys
            ]
            if index == 0:
                param_val_list = [list_[1:] for list_ in param_val_list]
            ydata = {}
            yspread = {}
            for key, val in zip(param_keys, param_val_list):
                ydata[key] = ut.get_list_column(val, 0)
                yspread[key] = ut.get_list_column(val, 1)
            ydatas.append(ydata)
            yspreads.append(yspread)
        return model._show_era_measure(ydatas, labels=labels, colors=colors,
                                       xdatas=xdatas,
                                       ylabel='update magnitude',
                                       yspreads=yspreads,
                                       yscale='linear', **kwargs)

    def _show_era_measure(model, ydatas, labels=None, styles=None, xdatas=None,
                          xlabel='epoch', ylabel='', yspreads=None,
                          colors=None, fnum=None, pnum=(1, 1, 1),
                          yscale='log'):

        # print('Show Era Measure ylabel = %r' % (ylabel,))
        import plottool as pt
        num_eras = len(model.era_history)

        if labels is None:
            pass

        if styles is None:
            styles = ut.ddict(lambda: '-o')

        if xdatas is None:
            xdatas = [era['epoch_list'] for era in model.era_history]

        if colors is None:
            colors = pt.distinct_colors(num_eras)

        prev_xdata = []
        prev_ydata = {}
        prev_yspread = {}

        fnum = pt.ensure_fnum(fnum)
        fig = pt.figure(fnum=fnum, pnum=pnum)
        ax = pt.gca()

        can_plot = True

        for index in range(num_eras):
            if len(xdatas) <= index:
                can_plot = False
                break
            xdata = xdatas[index]
            ydata = ydatas[index]

            if isinstance(colors, list):
                color_ = colors[index]
            else:
                color_ = colors

            # Draw lines between eras
            if len(prev_xdata) > 0:
                for key in ydata.keys():
                    color = color_[key] if isinstance(color_, dict) else color_
                    xdata_ = ut.flatten([prev_xdata, xdata[:1]])
                    ydata_ = ut.flatten([prev_ydata[key], ydata[key][:1]])
                    if yspreads is not None:
                        std_ = ut.flatten([prev_yspread[key], yspreads[index][key][:1]])
                        std_ = np.array(std_)
                        ymax = np.array(ydata_) + std_
                        ymin = np.array(ydata_) - std_
                        ax.fill_between(xdata_, ymin, ymax, alpha=.12, color=color)
                    pt.plot(xdata_, ydata_,
                            '--', color=color, yscale=yscale)

            # Draw lines inside era
            for key in ydata.keys():
                kw = {}
                # The last epoch gets the label
                if index == num_eras - 1:
                    kw = {'label': labels[key]}
                ydata_ = ydata[key]
                color = color_[key] if isinstance(color_, dict) else color_
                if yspreads is not None:
                    std = np.array(yspreads[index][key])
                    ymax = np.array(ydata_) + std
                    ymin = np.array(ydata_) - std
                    ax.fill_between(xdata, ymin, ymax, alpha=.2, color=color)
                    prev_yspread[key] = std[-1:]
                pt.plot(xdata, ydata_, styles[key], color=color,
                        yscale=yscale, **kw)
                prev_ydata[key] = ydata_[-1:]
            prev_xdata = xdata[-1:]

        # append_phantom_legend_label
        pt.set_xlabel(xlabel)
        pt.set_ylabel(ylabel)
        if num_eras > 0 and can_plot:
            pt.legend()
        pt.dark_background()
        return fig

    def show_regularization_stuff(model, fnum=None, pnum=(1, 1, 1)):
        import plottool as pt
        fnum = pt.ensure_fnum(fnum)
        fig = pt.figure(fnum=fnum, pnum=pnum)
        num_eras = len(model.era_history)
        for index, era in enumerate(model.era_history):
            epochs = era['epoch_list']
            if 'update_mags_list' not in era:
                continue
            update_mags_list = era['update_mags_list']
            # Transpose keys and positions
            param_keys = list(set(ut.flatten([
                dict_.keys()
                for dict_ in update_mags_list
            ])))
            param_val_list = [
                [list_[param] for list_ in update_mags_list]
                for param in param_keys
            ]
            colors = pt.distinct_colors(len(param_val_list))
            for key, val, color in zip(param_keys, param_val_list, colors):
                update_mag_mean = ut.get_list_column(val, 0)
                update_mag_std = ut.get_list_column(val, 1)
                if index == len(model.era_history) - 1:
                    pt.interval_line_plot(epochs, update_mag_mean,
                                          update_mag_std, marker='x',
                                          linestyle='-', color=color,
                                          label=key)
                else:
                    pt.interval_line_plot(epochs, update_mag_mean,
                                          update_mag_std, marker='x',
                                          linestyle='-', color=color)
            pass
        if num_eras > 0:
            pt.legend()

        pt.dark_background()
        return fig

    def show_weights_image(model, index=0, *args, **kwargs):
        import plottool as pt
        network_layers = model.get_all_layers()
        cnn_layers = [layer_ for layer_ in network_layers
                      if hasattr(layer_, 'W')]
        layer = cnn_layers[index]
        all_weights = layer.W.get_value()
        layername = net_strs.make_layer_str(layer)
        fig = draw_net.show_convolutional_weights(all_weights, **kwargs)
        figtitle = layername + '\n' + model.arch_id + '\n' + model.hist_id
        pt.set_figtitle(figtitle, subtitle='shape=%r, sum=%.4f, l2=%.4f' %
                        (all_weights.shape, all_weights.sum(),
                         (all_weights ** 2).sum()))
        return fig

    # --- IMAGE WRITE

    imwrite_weights = imwrite_wrapper(show_weights_image)
    #imwrite_arch = imwrite_wrapper(show_arch)

    def render_arch(model):
        import plottool as pt
        with pt.RenderingContext() as render:
            model.show_arch(fnum=render.fig.number)
        return render.image

    def imwrite_arch(model, fpath=None):
        r"""
        Args:
            fpath (str):  file path string(default = None)

        CommandLine:
            python -m ibeis_cnn.models.abstract_models imwrite_arch --show

        Example:
            >>> # DISABLE_DOCTEST
            >>> from ibeis_cnn.models.abstract_models import *  # NOQA
            >>> from ibeis_cnn.models.mnist import MNISTModel
            >>> model = MNISTModel(batch_size=128, data_shape=(24, 24, 1),
            >>>                    output_dims=10, batch_norm=False, name='mnist')
            >>> model.initialize_architecture()
            >>> fapth = model.imwrite_arch()
            >>> ut.quit_if_noshow()
            >>> ut.startfile(fapth)
        """
        # FIXME
        import vtool as vt
        if fpath is None:
            fpath = './' + model.arch_id + '.jpg'
        image = model.render_arch()
        vt.imwrite(fpath, image)
        return fpath


@ut.reloadable_class
class _ModelStrings(object):
    """
    """

    def get_state_str(model, other_override_reprs={}):
        era_history_str = ut.list_str(
            [ut.dict_str(era, truncate=True, sorted_=True)
             for era in model.era_history], strvals=True)

        override_reprs = {
            'best_results': ut.repr3(model.best_results),
            #'best_weights': ut.truncate_str(str(model.best_weights)),
            'data_params': ut.repr2(model.data_params, truncate=True),
            'learn_state': ut.repr3(model.learn_state),
            'era_history': era_history_str,
        }
        override_reprs.update(other_override_reprs)
        keys = list(set(model.__dict__.keys()) - set(override_reprs.keys()))
        for key in keys:
            if ut.is_func_or_method(model.__dict__[key]):
                # rrr support
                continue
            override_reprs[key] = repr(model.__dict__[key])

        state_str = ut.dict_str(override_reprs, sorted_=True, strvals=True)
        return state_str

    def make_arch_json(model, with_noise=False):
        """
        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis_cnn.models.abstract_models import *  # NOQA
            >>> from ibeis_cnn.models.mnist import MNISTModel
            >>> model = MNISTModel(batch_size=128, data_shape=(24, 24, 1),
            >>>                    output_dims=10, batch_norm=False)
            >>> model.initialize_architecture()
            >>> json_str = model.make_arch_json()
            >>> print(json_str)
        """
        layer_list = model.get_all_layers(with_noise=with_noise)
        layer_json_list = [net_strs.make_layer_json_dict(layer)
                           for layer in layer_list]
        json_dict = ut.odict()
        json_dict['arch_hashid'] = model.get_arch_hashid()
        json_dict['layers'] = layer_json_list
        #json_str1 = ut.to_json(json_dict, pretty=False)
        # prettier json
        json_str = ut.repr2(json_dict, nl=2)
        json_str = str(json_str.replace('\'', '"').replace('(', '[').replace(')', ']'))
        return json_str

    def get_architecture_str(model, sep='_', with_noise=False):
        r"""
        with_noise is a boolean that specifies if layers that doesnt
        affect the flow of information in the determenistic setting are to be
        included. IE get rid of dropout.

        CommandLine:
            python -m ibeis_cnn.models.abstract_models _ModelStrings.get_architecture_str:0

        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis_cnn.models.abstract_models import *  # NOQA
            >>> from ibeis_cnn.models.mnist import MNISTModel
            >>> model = MNISTModel(batch_size=128, data_shape=(24, 24, 1),
            >>>                    output_dims=10, batch_norm=True)
            >>> model.initialize_architecture()
            >>> result = model.get_architecture_str(sep=ut.NEWLINE, with_noise=False)
            >>> print(result)
            InputLayer(name=I0,shape=(128, 1, 24, 24))
            Conv2DDNNLayer(name=C1,num_filters=32,stride=(1, 1),nonlinearity=rectify)
            MaxPool2DDNNLayer(name=P1,stride=(2, 2))
            Conv2DDNNLayer(name=C2,num_filters=32,stride=(1, 1),nonlinearity=rectify)
            MaxPool2DDNNLayer(name=P2,stride=(2, 2))
            DenseLayer(name=F3,num_units=256,nonlinearity=rectify)
            DenseLayer(name=O4,num_units=10,nonlinearity=softmax)
        """
        if model.output_layer is None:
            return ''
        layer_list = model.get_all_layers(with_noise=with_noise)
        layer_str_list = [net_strs.make_layer_str(layer)
                          for layer in layer_list]
        architecture_str = sep.join(layer_str_list)
        return architecture_str

    def get_layer_info_str(model):
        """
        CommandLine:
            python -m ibeis_cnn.models.abstract_models _ModelStrings.get_layer_info_str:0

        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis_cnn.models.abstract_models import *  # NOQA
            >>> from ibeis_cnn.models.mnist import MNISTModel
            >>> model = MNISTModel(batch_size=128, data_shape=(24, 24, 1),
            >>>                    output_dims=10)
            >>> model.initialize_architecture()
            >>> result = model.get_layer_info_str()
            >>> print(result)
            Network Structure:
             index  Name  Layer               Outputs      Bytes OutShape           Params
             0      I0    InputLayer              576    294,912 (128, 1, 24, 24)   []
             1      C1    Conv2DDNNLayer       12,800  6,556,928 (128, 32, 20, 20)  [C1.W(32,1,5,5, {t,r}), C1.b(32, {t})]
             2      P1    MaxPool2DDNNLayer     3,200  1,638,400 (128, 32, 10, 10)  []
             3      C2    Conv2DDNNLayer        1,152    692,352 (128, 32, 6, 6)    [C2.W(32,32,5,5, {t,r}), C2.b(32, {t})]
             4      P2    MaxPool2DDNNLayer       288    147,456 (128, 32, 3, 3)    []
             5      D2    DropoutLayer            288    147,456 (128, 32, 3, 3)    []
             6      F3    DenseLayer              256    427,008 (128, 256)         [F3.W(288,256, {t,r}), F3.b(256, {t})]
             7      D3    DropoutLayer            256    131,072 (128, 256)         []
             8      O4    DenseLayer               10     15,400 (128, 10)          [O4.W(256,10, {t,r}), O4.b(10, {t})]
            ...this model has 103,018 learnable parameters
            ...this model will use 10,050,984 bytes = 9.59 MB
        """
        return net_strs.get_layer_info_str(model.get_all_layers())

    @property
    def total_epochs(model):
        return sum([len(era['epoch_list']) for era in model.era_history])

    @property
    def total_eras(model):
        return len(model.era_history)

    # --- PRINTING

    def print_state_str(model, **kwargs):
        print(model.get_state_str(**kwargs))

    def print_layer_info(model):
        net_strs.print_layer_info(model.get_all_layers())

    def print_architecture_str(model, sep='\n  '):
        architecture_str = model.get_architecture_str(sep=sep)
        if architecture_str is None or architecture_str == '':
            architecture_str = 'UNDEFINED'
        print('\nArchitecture:' + sep + architecture_str)

    def print_model_info_str(model):
        print('\n---- Arch Str')
        model.print_architecture_str(sep='\n')
        print('\n---- Layer Info')
        model.print_layer_info()
        print('\n---- Arch HashID')
        print('arch_hashid=%r' % (model.get_arch_hashid()),)
        print('----')


@ut.reloadable_class
class _ModelIDs(object):

    def _init_id_vars(model, kwargs):
        model.name = kwargs.pop('name', None)
        #if model.name is None:
        #    model.name = ut.get_classname(model.__class__, local=True)

    def __nice__(model):
        parts = [model.get_arch_nice(), model.get_history_nice()]
        if model.name is not None:
            parts = [model.name] + parts
        return '(' + ' '.join(parts) + ')'

    @property
    def hash_id(model):
        arch_hashid = model.get_arch_hashid()
        history_hashid = model.get_history_hashid()
        hashid = ut.hashstr27(history_hashid + arch_hashid)
        return hashid

    @property
    def arch_id(model):
        """
        CommandLine:
            python -m ibeis_cnn.models.abstract_models _ModelIDs.arch_id:0

        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis_cnn.models.abstract_models import *  # NOQA
            >>> from ibeis_cnn.models.mnist import MNISTModel
            >>> model = MNISTModel(batch_size=128, data_shape=(24, 24, 1),
            >>>                    output_dims=10, name='bnorm')
            >>> model.initialize_architecture()
            >>> result = str(model.arch_id)
            >>> print(result)
        """
        if model.name is not None:
            parts = ['arch', model.name, model.get_arch_nice(), model.get_arch_hashid()]
        else:
            parts = ['arch', model.get_arch_nice(), model.get_arch_hashid()]
        arch_id = '_'.join(parts)
        return arch_id

    @property
    def hist_id(model):
        r"""
        CommandLine:
            python -m ibeis_cnn.models.abstract_models --test-_ModelIDs.hist_id:0

        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis_cnn.models.abstract_models import *  # NOQA
            >>> model = testdata_model_with_history()
            >>> result = str(history_hashid)
            >>> print(result)
            eras003_epochs0012_eeeejtddhhhkoaim
        """
        hashid = model.get_history_hashid()
        nice = model.get_history_nice()
        history_id = nice + hashid
        return history_id

    def get_arch_hashid(model):
        """
        Returns a hash identifying the architecture of the determenistic net.
        This does not involve any dropout or noise layers, nor does the
        initialization of the weights matter.
        """
        arch_str = model.get_architecture_str(with_noise=False)
        arch_hashid = ut.hashstr27(arch_str, hashlen=8)
        return arch_hashid

    def get_history_hashid(model):
        r"""
        Builds a hashid that uniquely identifies the architecture and the
        training procedure this model has gone through to produce the current
        architecture weights.
        """
        era_hash_list = [ut.hashstr27(ut.repr2(era))
                         for era in model.era_history]
        era_hash_str = ''.join(era_hash_list)
        history_hashid = ut.hashstr27(era_hash_str, hashlen=8)
        return history_hashid

    def get_arch_nice(model):
        """
        Makes a string that shows the number of input units, output units,
        hidden units, parameters, and model depth.

        CommandLine:
            python -m ibeis_cnn.models.abstract_models get_arch_nice --show

        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis_cnn.models.abstract_models import *  # NOQA
            >>> from ibeis_cnn.models.mnist import MNISTModel
            >>> model = MNISTModel(batch_size=128, data_shape=(24, 24, 1),
            >>>                    output_dims=10)
            >>> model.initialize_architecture()
            >>> result = str(model.get_arch_nice())
            >>> print(result)
            o10_d4_c107
        """
        if model.output_layer is None:
            return 'NoArch'
        else:
            weighted_layers = model.get_all_layers(with_noise=False, with_weightless=False)
            info_list = [net_strs.get_layer_info(layer) for layer in weighted_layers]
            # The number of units depends if you look at things via input or output
            # does a convolutional layer have its outputs or the outputs of the pooling layer?
            #num_units1 = sum([info['num_outputs'] for info in info_list])
            nhidden = sum([info['num_inputs'] for info in info_list[1:]])
            #num_units2 += net_strs.get_layer_info(model.output_layer)['num_outputs']
            depth = len(weighted_layers)
            nparam = sum([info['num_params'] for info in info_list])
            nin = np.prod(model.data_shape)
            nout = model.output_dims
            fmtdict = dict(
                nin=nin,
                nout=nout,
                depth=depth,
                nhidden=nhidden,
                nparam=nparam,
                #logmcomplex=int(np.round(np.log10(nhidden + nparam)))
                mcomplex=int(np.round((nhidden + nparam) / 1000))
            )
            #nice = 'i{nin}_o{nout}_d{depth}_h{nhidden}_p{nparam}'.format(**fmtdict)
            # Use a complexity measure
            #nice = 'i{nin}_o{nout}_d{depth}_c{mcomplex}'.format(**fmtdict)
            nice = 'o{nout}_d{depth}_c{mcomplex}'.format(**fmtdict)
            return nice

    def get_history_nice(model):
        if model.total_epochs == 0:
            nice = 'NoHist'
        else:
            nice = 'era%03d_epoch%04d' % (model.total_eras, model.total_epochs)
        return nice


@ut.reloadable_class
class _ModelIO(object):

    def _init_io_vars(model, kwargs):
        model.dataset_dpath = kwargs.pop('dataset_dpath', '.')
        model.training_dpath = kwargs.pop('training_dpath', '.')

    def print_structure(model):
        """

        CommandLine:
            python -m ibeis_cnn.models.abstract_models print_structure --show

        Example:
            >>> from ibeis_cnn.ingest_data import *  # NOQA
            >>> dataset = grab_mnist_category_dataset()
            >>> dataset.print_dir_structure()
            >>> # ----
            >>> from ibeis_cnn.models.mnist import MNISTModel
            >>> model = MNISTModel(batch_size=128, data_shape=(24, 24, 1),
            >>>                    output_dims=10, dataset_dpath=dataset.dataset_dpath)
            >>> model.print_structure()

        """
        print(model.model_dpath)
        print(model.arch_dpath)
        print(model.best_dpath)
        print(model.saved_session_dpath)
        print(model.checkpoint_dpath)
        print(model.diagnostic_dpath)
        print(model.trained_model_dpath)
        print(model.trained_arch_dpath)

    @property
    def trained_model_dpath(model):
        return join(model.training_dpath, 'trained_models')

    @property
    def trained_arch_dpath(model):
        return join(model.trained_model_dpath, model.arch_id)

    @property
    def model_dpath(model):
        return join(model.dataset_dpath, 'models')

    @property
    def arch_dpath(model):
        return join(model.model_dpath, model.arch_id)

    @property
    def best_dpath(model):
        return join(model.arch_dpath, 'best')

    @property
    def checkpoint_dpath(model):
        return join(model.arch_dpath, 'checkpoints')

    @property
    def saved_session_dpath(model):
        return join(model.arch_dpath, 'saved_sessions')

    @property
    def diagnostic_dpath(model):
        return join(model.arch_dpath, 'diagnostics')

    def get_epoch_diagnostic_dpath(model, epoch=None):
        import utool as ut
        diagnostic_dpath = model.diagnostic_dpath
        ut.ensuredir(diagnostic_dpath)
        epoch_dpath = ut.unixjoin(diagnostic_dpath, model.hist_id)
        ut.ensuredir(epoch_dpath)
        return epoch_dpath

    def list_saved_checkpoints(model):
        dpath = model._get_model_dpath(None, True)
        checkpoint_dirs = sorted(ut.glob(dpath, '*', fullpath=False))
        return checkpoint_dirs

    def _get_model_dpath(model, dpath, checkpoint_tag):
        dpath = model.arch_dpath if dpath is None else dpath
        if checkpoint_tag is not None:
            # checkpoint dir requested
            dpath = join(dpath, 'checkpoints')
            if checkpoint_tag is not True:
                # specific checkpoint requested
                dpath = join(dpath, checkpoint_tag)
        return dpath

    def _get_model_file_fpath(model, default_fname, fpath, dpath, fname,
                              checkpoint_tag):
        if fpath is None:
            fname = default_fname if fname is None else fname
            dpath = model._get_model_dpath(dpath, checkpoint_tag)
            fpath = join(dpath, fname)
        else:
            assert checkpoint_tag is None, 'fpath overrides all other settings'
            assert dpath is None, 'fpath overrides all other settings'
            assert fname is None, 'fpath overrides all other settings'
        return fpath

    def resolve_fuzzy_checkpoint_pattern(model, checkpoint_pattern,
                                         extern_dpath=None):
        r"""
        tries to find a matching checkpoint so you dont have to type a full
        hash
        """
        dpath = model._get_model_dpath(extern_dpath, checkpoint_pattern)
        if exists(dpath):
            checkpoint_tag = checkpoint_pattern
        else:
            checkpoint_dpath = dirname(dpath)
            checkpoint_globpat = '*' + checkpoint_pattern + '*'
            matching_dpaths = ut.glob(checkpoint_dpath, checkpoint_globpat)
            if len(matching_dpaths) == 0:
                raise RuntimeError(
                    'Could not resolve checkpoint_pattern=%r. No Matches' %
                    (checkpoint_pattern,))
            elif len(matching_dpaths) > 1:
                raise RuntimeError(
                    ('Could not resolve checkpoint_pattern=%r. '
                        'matching_dpaths=%r. Too many matches') %
                    (checkpoint_pattern, matching_dpaths))
            else:
                checkpoint_tag = basename(matching_dpaths[0])
                print('Resolved checkpoint pattern to checkpoint_tag=%r' %
                        (checkpoint_tag,))
        return checkpoint_tag

    def has_saved_state(model, checkpoint_tag=None):
        """
        Check if there are any saved model states matching the checkpoing tag.
        """
        fpath = model.get_model_state_fpath(checkpoint_tag=checkpoint_tag)
        if checkpoint_tag is not None:
            ut.assertpath(fpath)
        return ut.checkpath(fpath)

    def get_model_state_fpath(model, fpath=None, dpath=None, fname=None,
                              checkpoint_tag=None):
        default_fname = 'model_state_arch_%s.pkl' % (
            model.get_arch_hashid())
        model_state_fpath = model._get_model_file_fpath(default_fname, fpath,
                                                        dpath, fname,
                                                        checkpoint_tag)
        return model_state_fpath

    def get_model_info_fpath(model, fpath=None, dpath=None, fname=None,
                             checkpoint_tag=None):
        default_fname = 'model_info_arch_%s.pkl' % (
            model.get_arch_hashid())
        model_state_fpath = model._get_model_file_fpath(default_fname, fpath,
                                                        dpath, fname,
                                                        checkpoint_tag)
        return model_state_fpath

    def checkpoint_save_model_state(model):
        fpath = model.get_model_state_fpath(checkpoint_tag=model.hist_id)
        ut.ensuredir(dirname(fpath))
        model.save_model_state(fpath=fpath)
        return fpath

    def checkpoint_save_model_info(model):
        fpath = model.get_model_info_fpath(checkpoint_tag=model.hist_id)
        ut.ensuredir(dirname(fpath))
        model.save_model_info(fpath=fpath)

    def save_model_state(model, **kwargs):
        """ saves current model state """
        current_weights = model.get_all_param_values()
        model_state = {
            'best_results': model.best_results,
            'data_params': model.data_params,
            'current_weights': current_weights,

            'input_shape':  model.input_shape,
            'data_shape': model.data_shape,
            'batch_size': model.data_shape,
            'output_dims':  model.output_dims,

            'era_history':  model.era_history,
            # 'arch_tag': model.arch_tag,
        }
        model_state_fpath = model.get_model_state_fpath(**kwargs)
        print('saving model state')
        #print('saving model state to: %s' % (model_state_fpath,))
        ut.save_cPkl(model_state_fpath, model_state, verbose=False)
        print('finished saving')

    def save_model_info(model, **kwargs):
        """ save model information (history and results but no weights) """
        model_info = {
            'best_results': model.best_results,
            'input_shape':  model.input_shape,
            'output_dims':  model.output_dims,
            'era_history':  model.era_history,
        }
        model_info_fpath = model.get_model_state_fpath(**kwargs)
        print('saving model info')
        #print('saving model info to: %s' % (model_info_fpath,))
        ut.save_cPkl(model_info_fpath, model_info, verbose=False)
        print('finished saving')

    def load_model_state(model, **kwargs):
        """
        kwargs = {}
        TODO: resolve load_model_state and load_extern_weights into a single
            function that is less magic in what it does and more
            straightforward

        Example:
            >>> # Assumes mnist is trained
            >>> from ibeis_cnn.models.abstract_models import  *  # NOQA
            >>> from ibeis_cnn.models import mnist
            >>> model, dataset = mnist.testdata_mnist()
            >>> model.initialize_architecture()
            >>> model.load_model_state()
        """
        model_state_fpath = model.get_model_state_fpath(**kwargs)
        print('[model] loading model state from: %s' % (model_state_fpath,))
        model_state = ut.load_cPkl(model_state_fpath)
        if model.__class__.__name__ != 'BaseModel':
            assert model_state['input_shape'][1:] == model.input_shape[1:], (
                'architecture disagreement')
            assert model_state['output_dims'] == model.output_dims, (
                'architecture disagreement')
            if 'preproc_kw' in model_state:
                model.data_params = model_state['preproc_kw']
                model._fix_center_mean_std()
            else:
                model.data_params = model_state['data_params']
        else:
            # HACK TO LOAD ABSTRACT MODEL FOR DIAGNOSITIC REASONS
            print("WARNING LOADING ABSTRACT MODEL")
        model.best_results = model_state['best_results']
        model.input_shape  = model_state['input_shape']
        model.output_dims  = model_state['output_dims']
        model.era_history  = model_state.get('era_history', [None])
        if model.__class__.__name__ != 'BaseModel':
            # hack for abstract model
            # model.output_layer is not None
            model.set_all_param_values(model.best_results['weights'])

    def load_extern_weights(model, **kwargs):
        """ load weights from another model """
        model_state_fpath = model.get_model_state_fpath(**kwargs)
        print('[model] loading extern weights from: %s' % (model_state_fpath,))
        model_state = ut.load_cPkl(model_state_fpath)
        if VERBOSE_CNN:
            print('External Model State:')
            print(ut.dict_str(model_state, truncate=True))
        # check compatibility with this architecture
        assert model_state['input_shape'][1:] == model.input_shape[1:], (
            'architecture disagreement')
        assert model_state['output_dims'] == model.output_dims, (
            'architecture disagreement')
        # Just set the weights, no other training state variables
        model.set_all_param_values(model_state['best_weights'])
        # also need to make sure the same preprocessing is used
        # TODO make this a layer?
        if 'preproc_kw' in model_state:
            model.data_params = model_state['preproc_kw']
            model._fix_center_mean_std()
        else:
            model.data_params = model_state['data_params']
        model.era_history = model_state['era_history']


class _ModelUtility(object):

    def set_all_param_values(model, weights_list):
        import ibeis_cnn.__LASAGNE__ as lasagne
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', '.*topo.*')
            lasagne.layers.set_all_param_values(
                model.output_layer, weights_list)

    def get_all_param_values(model):
        import ibeis_cnn.__LASAGNE__ as lasagne
        weights_list = lasagne.layers.get_all_param_values(model.output_layer)
        return weights_list

    def get_all_params(model, **tags):
        import ibeis_cnn.__LASAGNE__ as lasagne
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', '.*topo.*')
            parameters = lasagne.layers.get_all_params(
                model.output_layer, **tags)
            return parameters

    def get_all_layer_info(model):
        return [net_strs.get_layer_info(layer) for layer in model.get_all_layers()]

    def get_all_layers(model, with_noise=True, with_weightless=True):
        import ibeis_cnn.__LASAGNE__ as lasagne
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', '.*topo.*')
            warnings.filterwarnings('ignore', '.*layer.get_all_layers.*')
            assert model.output_layer is not None, 'need to initialize'
            layer_list_ = lasagne.layers.get_all_layers(model.output_layer)
        layer_list = layer_list_
        if not with_noise:
            # Remove dropout / gaussian noise layers
            layer_list = [
                layer for layer in layer_list_
                if layer.__class__.__name__ not in lasagne.layers.noise.__all__
            ]
        if not with_weightless:
            # Remove layers without weights
            layer_list = [layer for layer in layer_list if hasattr(layer, 'W')]
        return layer_list

    @property
    def layers_(model):
        """ for compatibility with nolearn visualizations """
        return model.get_all_layers()

    def get_output_layer(model):
        if model.output_layer is not None:
            return model.output_layer
        else:
            return None

    def _validate_data(model, X_train):
        """ Check to make sure data agrees with model input """
        input_layer = model.get_all_layers()[0]
        expected_item_shape = ut.take(input_layer.shape[1:], [1, 2, 0])
        expected_item_shape = tuple(expected_item_shape)
        given_item_shape = X_train.shape[1:]
        if given_item_shape != expected_item_shape:
            raise ValueError(
                'inconsistent item shape: ' +
                ('expected_item_shape = %r, ' % (expected_item_shape,)) +
                ('given_item_shape = %r' % (given_item_shape,))
            )

    def _validate_labels(model, X, y):
        if y is not None:
            assert X.shape[0] == (y.shape[0] * model.data_per_label_input), (
                'bad data / label alignment')

    def _validate_input(model, X, y=None):
        model._validate_data(X)
        model._validate_labels(X, y)

    def make_random_testdata(model, num=37, rng=0, cv2_format=False, asint=False):
        print('made random testdata')
        rng = ut.ensure_rng(rng)
        num_labels = num
        num_data   = num * model.data_per_label_input
        X = rng.rand(num_data, *model.data_shape)
        y = rng.rand(num_labels) * (model.output_dims + 1)
        X = (X * 100).astype(np.int) / 100
        if asint:
            X = (X * 255).astype(np.uint8)
        else:
            X = X.astype(np.float32)
        y = y.astype(np.int32)
        if not cv2_format:
            X = X.transpose((0, 3, 1, 2))
        return X, y


class _ModelBackend(object):
    """
    Functions that build and compile theano exepressions
    """

    def _init_compile_vars(model, kwargs):
        model._theano_exprs = ut.ddict(lambda: None)
        model._theano_backprop = None
        model._theano_forward = None
        model._theano_predict = None
        model._mode = None

    @property
    def theano_mode(model):
        # http://deeplearning.net/software/theano/library/compile/
        # function.html#theano.compile.function.function
        # http://deeplearning.net/software/theano/tutorial/modes.html
        # theano.compile.FAST_COMPILE / FAST_RUN
        return model._mode

    @theano_mode.setter
    def theano_mode(model, mode):
        import ibeis_cnn.__THEANO__ as theano
        if mode is None:
            pass
        elif mode == 'FAST_COMPILE':
            mode = theano.compile.FAST_COMPILE
        elif mode == 'FAST_RUN':
            mode = theano.compile.FAST_RUN
        else:
            raise ValueError('Unknown mode=%r' % (mode,))
        return mode

    def build(model):
        print('[model] --- BUILDING SYMBOLIC THEANO FUNCTIONS ---')
        model.build_backprop_func()
        model.build_forward_func()
        model.build_predict_func()
        print('[model] --- FINISHED BUILD ---')

    def build_predict_func(model):
        if model._theano_predict is None:
            import ibeis_cnn.__THEANO__ as theano
            from ibeis_cnn.__THEANO__ import tensor as T  # NOQA
            print('[model.build] request_predict')

            netout_exprs = model._get_network_output()
            network_output_determ = netout_exprs['network_output_determ']
            unlabeled_outputs = model._get_unlabeled_outputs()

            X, X_batch = model._get_batch_input_exprs()
            y, y_batch = model._get_batch_output_exprs()
            theano_predict = theano.function(
                inputs=[theano.In(X_batch)],
                outputs=[network_output_determ] + unlabeled_outputs,
                givens={X: X_batch},
                updates=None,
                mode=model.theano_mode,
                name=':predict'
            )
            model._theano_predict = theano_predict
        return model._theano_predict

    def build_forward_func(model):
        if model._theano_forward is None:
            import ibeis_cnn.__THEANO__ as theano
            from ibeis_cnn.__THEANO__ import tensor as T  # NOQA

            X, X_batch = model._get_batch_input_exprs()
            y, y_batch = model._get_batch_output_exprs()
            labeled_outputs = model._get_labeled_outputs()
            unlabeled_outputs = model._get_unlabeled_outputs()

            loss_exprs = model._get_loss_exprs()
            loss_determ = loss_exprs['loss_determ']

            print('[model.build] request_forward')
            theano_forward = theano.function(
                inputs=[theano.In(X_batch), theano.In(y_batch)],
                outputs=[loss_determ] + labeled_outputs + unlabeled_outputs,
                givens={X: X_batch, y: y_batch},
                updates=None,
                mode=model.theano_mode,
                name=':feedforward'
            )
            model._theano_forward = theano_forward
        return model._theano_forward

    def build_backprop_func(model):
        if model._theano_backprop is None:
            print('[model.build] request_backprop')
            import ibeis_cnn.__THEANO__ as theano
            from ibeis_cnn.__THEANO__ import tensor as T  # NOQA

            model.learn_state.init()

            X, X_batch = model._get_batch_input_exprs()
            y, y_batch = model._get_batch_output_exprs()
            labeled_outputs = model._get_labeled_outputs()

            # Build backprop losses
            loss_exprs = model._get_loss_exprs()
            loss             = loss_exprs['loss']
            loss_regularized = loss_exprs['loss_regularized']

            if loss_regularized is not None:
                backprop_loss_ = loss_regularized
            else:
                backprop_loss_ = loss

            backprop_losses = []
            if loss_regularized is not None:
                backprop_losses.append(loss_regularized)
            backprop_losses.append(loss)

            # Updates network parameters based on the training loss
            parameters = model.get_all_params(trainable=True)

            updates = model._make_updates(parameters, backprop_loss_)
            monitor_outputs = model._make_monitor_outputs(parameters, updates)

            theano_backprop = theano.function(
                inputs=[theano.In(X_batch), theano.In(y_batch)],
                outputs=(backprop_losses + labeled_outputs + monitor_outputs),
                givens={X: X_batch, y: y_batch},
                updates=updates,
                mode=model.theano_mode,
                name=':backprop',
            )
            model._theano_backprop = theano_backprop
        return model._theano_backprop

    def _get_batch_input_exprs(model):
        from ibeis_cnn.__THEANO__ import tensor as T  # NOQA
        if model._theano_exprs['batch_input'] is None:
            #if input_type is None:
            input_type = T.tensor4
            X = input_type('x')
            X_batch = input_type('x_batch')
            model._theano_exprs['batch_input'] = X, X_batch
        return model._theano_exprs['batch_input']

    def _get_batch_output_exprs(model):
        from ibeis_cnn.__THEANO__ import tensor as T  # NOQA
        if model._theano_exprs['batch_output'] is None:
            #if output_type is None:
            output_type = T.ivector
            y = output_type('y')
            y_batch = output_type('y_batch')
            model._theano_exprs['batch_output'] = y, y_batch
        return model._theano_exprs['batch_output']

    def _get_loss_exprs(model):
        r"""
        Requires that a custom loss function is defined in the inherited class
        """
        if model._theano_exprs['loss'] is None:
            import ibeis_cnn.__LASAGNE__ as lasagne
            with warnings.catch_warnings():
                X, X_batch = model._get_batch_input_exprs()
                y, y_batch = model._get_batch_output_exprs()

                netout_exprs = model._get_network_output()
                network_output_learn = netout_exprs['network_output_learn']
                network_output_determ = netout_exprs['network_output_determ']

                print('Building symbolic loss function')
                losses = model.loss_function(network_output_learn, y_batch)
                loss = lasagne.objectives.aggregate(losses, mode='mean')
                loss.name = 'loss'

                print('Building symbolic loss function (determenistic)')
                losses_determ = model.loss_function(network_output_determ, y_batch)
                # TODO Weight classes
                loss_determ = lasagne.objectives.aggregate(losses_determ,
                                                           weights=None,
                                                           mode='mean')
                loss_determ.name = 'loss_determ'

                # Regularize
                # TODO: L2 should be one of many available options for
                # regularization
                L2 = lasagne.regularization.regularize_network_params(
                    model.output_layer, lasagne.regularization.l2)
                weight_decay = model.learn_state['weight_decay']
                if weight_decay is not None or weight_decay == 0:
                    regularization_term = weight_decay * L2
                    regularization_term.name = 'regularization_term'
                    #L2 = lasagne.regularization.l2(model.output_layer)
                    loss_regularized = loss + regularization_term
                    loss_regularized.name = 'loss_regularized'
                else:
                    loss_regularized = None

                loss_exprs = {
                    'loss': loss,
                    'loss_determ': loss_determ,
                    'loss_regularized': loss_regularized,
                }
            model._theano_exprs['loss'] = loss_exprs
        return model._theano_exprs['loss']

    def _make_updates(model, parameters, backprop_loss_):
        import ibeis_cnn.__LASAGNE__ as lasagne
        import ibeis_cnn.__THEANO__ as theano
        from ibeis_cnn.__THEANO__ import tensor as T  # NOQA

        grads = theano.grad(backprop_loss_, parameters, add_names=True)

        shared_learning_rate = model.learn_state.shared['learning_rate']
        momentum = model.learn_state['momentum']

        updates = lasagne.updates.nesterov_momentum(
            loss_or_grads=grads,
            params=parameters,
            learning_rate=shared_learning_rate,
            momentum=momentum
            # add_names=True  # TODO; commit to lasange
        )

        # workaround for pylearn2 bug documented in
        # https://github.com/Lasagne/Lasagne/issues/728
        for param, update in updates.items():
            if param.broadcastable != update.broadcastable:
                updates[param] = T.patternbroadcast(update, param.broadcastable)
        return updates

    def _get_network_output(model):
        if model._theano_exprs['netout'] is None:
            import ibeis_cnn.__LASAGNE__ as lasagne
            X, X_batch = model._get_batch_input_exprs()

            network_output_learn = lasagne.layers.get_output(
                model.output_layer, X_batch)
            network_output_learn.name = 'network_output_learn'

            network_output_determ = lasagne.layers.get_output(
                model.output_layer, X_batch, deterministic=True)
            network_output_determ.name = 'network_output_determ'

            netout_exprs = {
                'network_output_learn': network_output_learn,
                'network_output_determ': network_output_determ,
            }
            model._theano_exprs['netout'] = netout_exprs
        return model._theano_exprs['netout']

    def _get_unlabeled_outputs(model):
        if model._theano_exprs['unlabeled_out'] is None:
            netout_exprs = model._get_network_output()
            network_output_determ = netout_exprs['network_output_determ']
            out = model.custom_unlabeled_outputs(network_output_determ)
            model._theano_exprs['unlabeled_out'] = out
        return model._theano_exprs['unlabeled_out']

    def _get_labeled_outputs(model):
        if model._theano_exprs['labeled_out'] is None:
            netout_exprs = model._get_network_output()
            network_output_determ = netout_exprs['network_output_determ']
            y, y_batch = model._get_batch_output_exprs()
            model._theano_exprs['labeled_out'] = model.custom_labeled_outputs(
                network_output_determ, y_batch)
        return model._theano_exprs['labeled_out']

    def _make_monitor_outputs(model, parameters, updates):
        """
        Builds parameters to monitor the magnitude of updates durning learning
        """
        from ibeis_cnn.__THEANO__ import tensor as T  # NOQA
        # Build outputs to babysit training
        monitor_outputs = []
        if model.monitor_config['monitor_updates']:  # and False:
            for param in parameters:
                # The vector each param was udpated with
                # (one vector per channel)
                param_update_vec = updates[param] - param
                param_update_vec.name = 'param_update_vector_' + param.name
                flat_shape = (param_update_vec.shape[0],
                              T.prod(param_update_vec.shape[1:]))
                flat_param_update_vec = param_update_vec.reshape(flat_shape)
                param_update_mag = (flat_param_update_vec ** 2).sum(-1)
                param_update_mag.name = 'param_update_magnitude_' + param.name
                monitor_outputs.append(param_update_mag)
        return monitor_outputs

    def custom_unlabeled_outputs(model, network_output):
        """
        override in inherited subclass to enable custom symbolic expressions
        based on the network output alone
        """
        raise NotImplementedError('need override')
        return []

    def custom_labeled_outputs(model, network_output, y_batch):
        """
        override in inherited subclass to enable custom symbolic expressions
        based on the network output and the labels
        """
        raise NotImplementedError('need override')
        return []


@ut.reloadable_class
class BaseModel(_model_legacy._ModelLegacy, _ModelVisualization, _ModelIO,
                _ModelStrings, _ModelIDs, _ModelBackend, _ModelFitter,
                _ModelPredicter, _ModelBatch, _ModelUtility, ut.NiceRepr):
    """
    Abstract model providing functionality for all other models to derive from
    """
    def __init__(model, **kwargs):
        """
        Guess on Shapes:
            input_shape (tuple): in Theano format (b, c, h, w)
            data_shape (tuple):  in  Numpy format (b, h, w, c)
        """
        kwargs = kwargs.copy()
        if kwargs.pop('verbose_compile', True):
            import logging
            compile_logger = logging.getLogger('theano.compile')
            compile_logger.setLevel(-10)
        model._init_io_vars(kwargs)
        model._init_id_vars(kwargs)
        model._init_shape_vars(kwargs)
        model._init_compile_vars(kwargs)
        model._init_fit_vars(kwargs)
        model._init_batch_vars(kwargs)
        model.output_layer = None
        assert len(kwargs) == 0, (
            'Model was given unused keywords=%r' % (list(kwargs.keys())))

    def _init_shape_vars(model, kwargs):
        input_shape = kwargs.pop('input_shape', None)
        batch_size = kwargs.pop('batch_size', None)
        data_shape = kwargs.pop('data_shape', None)
        output_dims = kwargs.pop('output_dims', None)

        if input_shape is None and data_shape is None:
            report_error(
                'Must specify either input_shape or data_shape')
        elif input_shape is None:
            input_shape = (batch_size, data_shape[2], data_shape[0],
                           data_shape[1])
        elif data_shape is None and batch_size is None:
            data_shape = (input_shape[2], input_shape[3], input_shape[1])
            batch_size = input_shape[0]
        else:
            report_error(
                'Dont specify batch_size or data_shape with input_shape')

        model.output_dims = output_dims
        model.input_shape = input_shape
        model.data_shape = data_shape
        model.batch_size = batch_size
        # bad name, says that this network will take
        # 2*N images in a batch and N labels that map to
        # two images a piece
        model.data_per_label_input  = 1  # state of network input
        model.data_per_label_output = 1  # state of network output

    # --- OTHER
    @property
    def input_batchsize(model):
        return model.input_shape[0]

    @property
    def input_channels(model):
        return model.input_shape[1]

    @property
    def input_height(model):
        return model.input_shape[2]

    @property
    def input_width(model):
        return model.input_shape[3]

    # --- INITIALIZATION

    def initialize_architecture(model):
        raise NotImplementedError('reimplement')

    def augment(model, Xb, yb):
        raise NotImplementedError('data augmentation not implemented')

    def loss_function(model, network_output, truth):
        raise NotImplementedError('need to implement a loss function')

    def reinit_weights(model, W=None):
        """
        initailizes weights after the architecture has been defined.
        """
        import ibeis_cnn.__LASAGNE__ as lasagne
        if W is None:
            W = 'orthogonal'
        if isinstance(W, six.string_types):
            if W == 'orthogonal':
                W = lasagne.init.Orthogonal()
        print('Reinitializing all weights to %r' % (W,))
        weights_list = model.get_all_params(regularizable=True, trainable=True)
        #print(weights_list)
        for weights in weights_list:
            #print(weights)
            shape = weights.get_value().shape
            new_values = W.sample(shape)
            weights.set_value(new_values)

    # ---- UTILITY


@ut.reloadable_class
class AbstractCategoricalModel(BaseModel):
    """ base model for catagory classifiers """

    def __init__(model, **kwargs):
        # BaseModel.__init__(model, **kwargs)
        super(AbstractCategoricalModel, model).__init__(**kwargs)
        model.encoder = None
        # categorical models have a concept of accuracy
        #model.requested_headers += ['valid_acc', 'test_acc']
        model.requested_headers += ['valid_acc']

    def initialize_encoder(model, labels):
        print('[model] encoding labels')
        from sklearn import preprocessing
        model.encoder = preprocessing.LabelEncoder()
        model.encoder.fit(labels)
        model.output_dims = len(list(np.unique(labels)))
        print('[model] model.output_dims = %r' % (model.output_dims,))

    def loss_function(model, network_output, truth):
        # https://en.wikipedia.org/wiki/Loss_functions_for_classification
        from ibeis_cnn.__THEANO__ import tensor as T  # NOQA
        # categorical cross-entropy between predictions and targets
        # L_i = -\sum_{j} t_{i,j} \log{p_{i, j}}
        return T.nnet.categorical_crossentropy(network_output, truth)

    def custom_unlabeled_outputs(model, network_output):
        from ibeis_cnn.__THEANO__ import tensor as T  # NOQA
        # Network outputs define category probabilities
        probabilities = network_output
        predictions = T.argmax(probabilities, axis=1)
        predictions.name = 'predictions'
        confidences = probabilities.max(axis=1)
        confidences.name = 'confidences'
        unlabeled_outputs = [predictions, confidences]
        return unlabeled_outputs

    def custom_labeled_outputs(model, network_output, y_batch):
        from ibeis_cnn.__THEANO__ import tensor as T  # NOQA
        probabilities = network_output
        predictions = T.argmax(probabilities, axis=1)
        predictions.name = 'tmp_predictions'
        accuracy = T.mean(T.eq(predictions, y_batch))
        accuracy.name = 'accuracy'
        labeled_outputs = [accuracy]
        return labeled_outputs


def report_error(msg):
    if False:
        raise ValueError(msg)
    else:
        print('WARNING:' + msg)


def evaluate_layer_list(network_layers_def, verbose=None):
    r"""
    compiles a sequence of partial functions into a network
    """
    if verbose is None:
        verbose = VERBOSE_CNN
    total = len(network_layers_def)
    network_layers = []
    if verbose:
        print('Evaluting List of %d Layers' % (total,))
    layer_fn_iter = iter(network_layers_def)
    #if True:
    #with warnings.catch_warnings():
    #    warnings.filterwarnings(
    #        'ignore', '.*The uniform initializer no longer uses Glorot.*')
    try:
        with ut.Indenter(' ' * 4, enabled=verbose):
            next_args = tuple()
            for count, layer_fn in enumerate(layer_fn_iter, start=1):
                if verbose:
                    print('Evaluating layer %d/%d (%s) ' %
                          (count, total, ut.get_funcname(layer_fn), ))
                with ut.Timer(verbose=False) as tt:
                    layer = layer_fn(*next_args)
                next_args = (layer,)
                network_layers.append(layer)
                if verbose:
                    print('  * took %.4fs' % (tt.toc(),))
                    print('  * layer = %r' % (layer,))
                    if hasattr(layer, 'input_shape'):
                        print('  * layer.input_shape = %r' % (
                            layer.input_shape,))
                    if hasattr(layer, 'shape'):
                        print('  * layer.shape = %r' % (
                            layer.shape,))
                    print('  * layer.output_shape = %r' % (
                        layer.output_shape,))
    except Exception as ex:
        keys = ['layer_fn', 'layer_fn.func', 'layer_fn.args',
                'layer_fn.keywords', 'layer', 'count']
        ut.printex(ex,
                   ('Error buildling layers.\n'
                    'layer.name=%r') % (layer),
                   keys=keys)
        raise
    return network_layers


def testdata_model_with_history():
    model = BaseModel()
    # make a dummy history
    X_train, y_train = [1, 2, 3], [0, 0, 1]
    rng = np.random.RandomState(0)
    def dummy_epoch_dict(num):
        from scipy.special import expit
        mean_loss = 1 / np.exp(num / 10)
        frac = (num / total_epochs)
        def rand():
            return rng.rand() / 10
        epoch_info = {
            'epoch': num,
            # 'loss': 1 / np.exp(num / 10) + rng.rand() / 100,
            'learn_loss': mean_loss + rand(),
            'learn_loss_regularized': (mean_loss + np.exp(rand() * num + rand())),
            'valid_loss': mean_loss - rand(),
            'valid_acc': expit(-6 + 12 * np.clip(frac + rand(), 0, 1)),
            'param_update_mags': {
                'C0': (rng.normal() ** 2, rng.rand()),
                'F1': (rng.normal() ** 2, rng.rand()),
            }
        }
        return epoch_info
    def dummy_get_all_layer_info(model):
        return [{'param_infos': [{'name': 'C0', 'tags': ['trainable']}]},
                {'param_infos': [{'name': 'F1', 'tags': ['trainable']}]}]
    ut.inject_func_as_method(model, dummy_get_all_layer_info, 'get_all_layer_info', allow_override=True)
    epoch = 0
    eralens = [4, 4, 4]
    total_epochs = sum(eralens)
    for era_length in eralens:
        model._new_era(X_train, y_train, X_train, y_train)
        for _ in range(era_length):
            model._record_epoch(dummy_epoch_dict(num=epoch))
            epoch += 1
    #model._record_epoch({'epoch': 1, 'valid_loss': .8, 'learn_loss': .9})
    #model._record_epoch({'epoch': 2, 'valid_loss': .5, 'learn_loss': .7})
    #model._record_epoch({'epoch': 3, 'valid_loss': .3, 'learn_loss': .6})
    #model._record_epoch({'epoch': 4, 'valid_loss': .2, 'learn_loss': .3})
    #model._record_epoch({'epoch': 5, 'valid_loss': .1, 'learn_loss': .2})
    return model


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis_cnn.abstract_models
        python -m ibeis_cnn.abstract_models --allexamples
        python -m ibeis_cnn.abstract_models --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
