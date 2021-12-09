# -*- coding: utf-8 -*-
"""
Directory structure of training

The network directory is the root of the structure and is typically in
_ibeis_cache/nets for ibeis databases. Otherwise it it custom defined (like in
.cache/wbia_cnn/training for mnist tests)

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
import logging
import six
import numpy as np
import utool as ut
import sys
from os.path import join, exists, dirname, basename, split, splitext
from six.moves import cPickle as pickle  # NOQA
import warnings
import sklearn
from wbia_cnn import net_strs
from wbia_cnn import draw_net
from wbia_cnn.models import _model_legacy

print, rrr, profile = ut.inject2(__name__)
logger = logging.getLogger()


VERBOSE_CNN = ut.get_module_verbosity_flags('cnn')[0] or ut.VERBOSE

# Delayed imports
lasagne = None
T = None
theano = None


def delayed_import():
    global lasagne
    global theano
    global T
    import lasagne
    import theano
    from theano import tensor as T  # NOQA


# def testdata_model_with_history():
#     model = BaseModel()
#     # make a dummy history
#     X_train, y_train = [1, 2, 3], [0, 0, 1]
#     rng = np.random.RandomState(0)

#     def dummy_epoch_dict(num):
#         epoch_info = {
#             'epoch': num,
#             'loss': 1 / np.exp(num / 10) + rng.rand() / 100,
#             'train_loss': 1 / np.exp(num / 10) + rng.rand() / 100,
#             'train_loss_regularized': (
#                 1 / np.exp(num / 10) + np.exp(rng.rand() * num) + rng.rand() / 100
#             ),
#             'valid_loss': 1 / np.exp(num / 10) - rng.rand() / 100,
#             'param_update_mags': {
#                 'C0': (rng.normal() ** 2, rng.rand()),
#                 'F1': (rng.normal() ** 2, rng.rand()),
#             },
#         }
#         return epoch_info

#     count = 0
#     for era_length in [4, 4, 4]:
#         alias_key = 'dummy_alias_key'
#         model.start_new_era(X_train, y_train, X_train, y_train, alias_key)
#         for count in range(count, count + era_length):
#             model.record_epoch(dummy_epoch_dict(count))
#     # model.record_epoch({'epoch': 1, 'valid_loss': .8, 'train_loss': .9})
#     # model.record_epoch({'epoch': 2, 'valid_loss': .5, 'train_loss': .7})
#     # model.record_epoch({'epoch': 3, 'valid_loss': .3, 'train_loss': .6})
#     # model.record_epoch({'epoch': 4, 'valid_loss': .2, 'train_loss': .3})
#     # model.record_epoch({'epoch': 5, 'valid_loss': .1, 'train_loss': .2})
#     return model


if 'theano' in sys.modules:
    delayed_import()


@ut.reloadable_class
class History(ut.NiceRepr):
    """
    Manages bookkeeping for training history
    """

    def __init__(history):
        # an era is a group of epochs
        history.era_list = []
        history.epoch_list = []
        # Marks the start of the era
        history._start_epoch = 0

    def __len__(history):
        return history.total_epochs

    def __nice__(history):
        return history.get_history_nice()

    # def __iter__(history):
    #    for epochs in history.grouped_epochs():
    #        yield epochs

    @classmethod
    def from_oldstyle(cls, era_history):
        history = cls()
        for era_num, era in enumerate(era_history):
            epoch_info_list = era['epoch_info_list']
            era = ut.delete_dict_keys(era.copy(), ['epoch_info_list', 'size'])
            # Append new information
            era['era_num'] = era_num
            for epoch in epoch_info_list:
                epoch = epoch.copy()
                epoch['era_num'] = era_num
                if 'epoch' in epoch:
                    epoch['epoch_num'] = epoch['epoch']
                    del epoch['epoch']
                history.epoch_list.append(epoch)
            history.era_list.append(era)
        history._start_epoch = len(history.epoch_list)
        return history

    @property
    def total_epochs(history):
        return len(history.epoch_list)

    @property
    def total_eras(history):
        return len(history.era_list)

    @property
    def hist_id(history):
        r"""
        CommandLine:
            python -m wbia_cnn.models.abstract_models --test-History.hist_id:0

        Example:
            >>> # ENABLE_DOCTEST
            >>> from wbia_cnn.models.abstract_models import *  # NOQA
            >>> model = testdata_model_with_history()
            >>> history = model.history
            >>> result = str(model.history.hist_id)
            >>> print(result)
            epoch0003_era012_zamwoidy
        """
        hashid = history.get_history_hashid()
        nice = history.get_history_nice()
        history_id = nice + '_' + hashid
        return history_id

    @property
    def current_era_size(history):
        return history.total_epochs - history._start_epoch

    def get_history_hashid(history):
        r"""
        Builds a hashid that uniquely identifies the architecture and the
        training procedure this model has gone through to produce the current
        architecture weights.
        """
        era_hash_list = [ut.hashstr27(ut.repr2(era)) for era in history.era_list]
        # epoch_hash_list = [ut.hashstr27(ut.repr2(epoch)) for epoch in history.epoch_list]
        # epoch_hash_str = ''.join(epoch_hash_list)
        era_hash_str = ''.join(era_hash_list)
        era_hash_str += str(history.total_epochs)
        history_hashid = ut.hashstr27(era_hash_str, hashlen=8)
        return history_hashid

    def get_history_nice(history):
        if history.total_epochs == 0:
            nice = 'NoHist'
        else:
            nice = 'epoch%04d_era%03d' % (history.total_eras, history.total_epochs)
        return nice

    def grouped_epochs(history):
        era_num = ut.take_column(history.epoch_list, 'era_num')
        unique, groupxs = ut.group_indices(era_num)
        grouped_epochs = ut.apply_grouping(history.epoch_list, groupxs)
        return grouped_epochs

    def grouped_epochsT(history):
        for epochs in history.grouped_epochs():
            yield ut.dict_stack2(epochs)

    def record_epoch(history, epoch_info):
        epoch_info['epoch_num'] = len(history.epoch_list)
        history.epoch_list.append(epoch_info)

    def _new_era(history, model, X_learn, y_learn, X_valid, y_valid):
        """
        Used to denote a change in hyperparameters during training.
        """
        y_hashid = ut.hashstr_arr(y_learn, 'y', alphabet=ut.ALPHABET_27)

        learn_hashid = str(model.arch_id) + '_' + y_hashid
        if history.total_epochs > 0 and history.current_era_size == 0:
            logger.info('Not starting new era (previous era has no epochs)')
        else:
            _new_era = {
                'size': 0,
                'learn_hashid': learn_hashid,
                'arch_hashid': model.get_arch_hashid(),
                'arch_id': model.arch_id,
                'num_learn': len(y_learn),
                'num_valid': len(y_valid),
                'learn_state': model.learn_state.asdict(),
            }
            num_eras = history.total_eras
            logger.info('starting new era %d' % (num_eras,))
            model.current_era = _new_era
            history.era_list.append(_new_era)
            history._start_epoch = history.total_epochs

    def _record_epoch(history, epoch_info):
        """
        Records an epoch in an era.
        """
        # each key/val in epoch_info dict corresponds to a key/val_list in an
        # era dict.
        # history.current_era['size'] += 1
        # history.current_era['epoch_info_list'].append(epoch_info)
        epoch_info['era_num'] = history.total_eras
        history.epoch_list.append(epoch_info)

    def rewind_to(history, epoch_num):
        target_epoch = history.epoch_list[epoch_num]
        era_num = target_epoch['era_num']
        history.epoch_list = history.epoch_list[: epoch_num + 1]
        history.era_list = history.era_list[: era_num + 1]
        history._start_epoch = history.total_epochs

    def to_json(history):
        return ut.to_json(history.__dict__)


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
        fset=lambda self, val: self.setitem('momentum', val),
    )

    learning_rate = property(
        fget=lambda self: self.getitem('learning_rate'),
        fset=lambda self, val: self.setitem('learning_rate', val),
    )

    weight_decay = property(
        fget=lambda self: self.getitem('weight_decay'),
        fset=lambda self, val: self.setitem('weight_decay', val),
    )

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
            import theano

            logger.info('[model] setting %s to %.9r' % (key, value))
            _shared = self._shared_state[key]
            if value is None and _shared is not None:
                raise ValueError('Cannot set an initialized shared variable to None.')
            elif _shared is None and value is not None:
                self._shared_state[key] = theano.shared(
                    np.cast['float32'](value), name=key
                )
            elif _shared is not None:
                _shared.set_value(np.cast['float32'](value))
        else:
            self._shared_state[key] = value


@ut.reloadable_class
class _ModelFitter(object):
    """
    CommandLine:
        python -m wbia_cnn _ModelFitter.fit:0
    """

    def _init_fit_vars(model, kwargs):
        model._rng = ut.ensure_rng(0)
        model.history = History()
        # Training state
        model.requested_headers = ['learn_loss', 'valid_loss', 'learnval_rat']
        model.data_params = None
        # Stores current result
        model.best_results = {
            'epoch_num': None,
            'learn_loss': np.inf,
            'valid_loss': np.inf,
            'weights': None,
        }

        # TODO: some sort of behavior config Things that dont influence
        # training, but do impact performance / memory usage.
        model._behavior = {
            'buffered': False,
        }
        # Static configuration indicating training preferences
        # (these will not influence the model learning)
        model.monitor_config = {
            'monitor': ut.get_argflag('--monitor'),
            'monitor_updates': False,
            'checkpoint_freq': 50,
            'case_dump_freq': 25,
            'weight_dump_freq': 5,
            'showprog': True,
        }
        ut.update_existing(model.monitor_config, kwargs)
        ut.delete_dict_keys(kwargs, model.monitor_config.keys())
        # Static configuration indicating hyper-parameters
        # (these will influence how the model learns)
        # Some of these values (ie learning state) may be dynamic durring
        # training. The dynamic version should be used in this context. This
        # dictionary is always static in a fit session and only indicates the
        # initial state of these variables.
        model.hyperparams = {
            'label_encode_on': True,
            'whiten_on': False,
            'augment_on': False,
            'augment_on_validate': False,
            'augment_weights': False,
            'augment_delay': 0,
            # 'augment_delay': 2,
            'era_size': 10,  # epochs per era
            'era_clean': False,
            'max_epochs': None,
            'rate_schedule': 0.9,
            'stopping_patience': 100,
            # 'class_weight': None,
            'class_weight': 'balanced',
            'random_seed': None,
            'learning_rate': 0.005,
            'momentum': 0.9,
            'weight_decay': 0.0,
        }
        ut.update_existing(model.hyperparams, kwargs)
        ut.delete_dict_keys(kwargs, model.hyperparams.keys())
        # Dynamic configuration that may change with time
        model.learn_state = LearnState(
            learning_rate=model.hyperparams['learning_rate'],
            momentum=model.hyperparams['momentum'],
            weight_decay=model.hyperparams['weight_decay'],
        )
        # This will by a dynamic dict that will span the life of a training
        # session
        model._fit_session = None

    def _default_input_weights(model, X, y, w=None):
        if w is None:
            # Hack, assuming a classification task
            if 'class_to_weight' in model.data_params:
                class_to_weight = model.data_params['class_to_weight']
                w = class_to_weight.take(y).astype(np.float32)
            else:
                # logger.info('no class weights')
                w = np.ones(y.shape).astype(np.float32)
        return w

    def fit(
        model,
        X_train,
        y_train,
        X_valid=None,
        y_valid=None,
        valid_idx=None,
        X_test=None,
        y_test=None,
        verbose=True,
        **kwargs
    ):
        r"""
        Trains the network with backprop.

        CommandLine:
            python -m wbia_cnn _ModelFitter.fit --name=bnorm --vd --monitor
            python -m wbia_cnn _ModelFitter.fit --name=dropout
            python -m wbia_cnn _ModelFitter.fit --name=incep

        Example1:
            >>> from wbia_cnn.models import mnist
            >>> model, dataset = mnist.testdata_mnist(defaultname='bnorm', dropout=.5)
            >>> model.init_arch()
            >>> model.print_layer_info()
            >>> model.print_model_info_str()
            >>> X_train, y_train = dataset.subset('train')
            >>> model.fit(X_train, y_train)
        """
        from wbia_cnn import utils

        logger.info('\n[train] --- TRAINING LOOP ---')
        ut.update_existing(model.hyperparams, kwargs)
        ut.delete_dict_keys(kwargs, model.hyperparams.keys())
        ut.update_existing(model.monitor_config, kwargs)
        ut.delete_dict_keys(kwargs, model.monitor_config.keys())
        ut.update_existing(model._behavior, kwargs)
        ut.delete_dict_keys(kwargs, model._behavior.keys())
        assert len(kwargs) == 0, 'unhandled kwargs=%r' % (kwargs,)

        try:
            model._validate_input(X_train, y_train)
        except Exception:
            logger.info('[WARNING] Input validation failed...')

        X_learn, y_learn, X_valid, y_valid = model._ensure_learnval_split(
            X_train, y_train, X_valid, y_valid, valid_idx
        )

        model.ensure_data_params(X_learn, y_learn)

        has_encoder = getattr(model, 'encoder', None) is not None
        learn_hist = model.encoder.inverse_transform(y_learn) if has_encoder else y_learn
        valid_hist = model.encoder.inverse_transform(y_valid) if has_encoder else y_valid
        if learn_hist.shape[-1] == 1:
            logger.info('Learn y histogram: ' + ut.repr2(ut.dict_hist(learn_hist)))
        if valid_hist.shape[-1] == 1:
            logger.info('Valid y histogram: ' + ut.repr2(ut.dict_hist(valid_hist)))

        # FIXME: make class weights more ellegant and customizable
        w_learn = model._default_input_weights(X_learn, y_learn)
        w_valid = model._default_input_weights(X_valid, y_valid)

        model._new_fit_session()

        epoch = model.best_results['epoch_num']
        if epoch is None:
            epoch = 0
            logger.info('Initializng training at epoch=%r' % (epoch,))
        else:
            logger.info('Resuming training at epoch=%r' % (epoch,))
        # Begin training the neural network
        logger.info('model.monitor_config = %s' % (ut.repr4(model.monitor_config),))
        logger.info('model.batch_size = %r' % (model.batch_size,))
        logger.info('model.hyperparams = %s' % (ut.repr4(model.hyperparams),))
        logger.info('learn_state = %s' % ut.repr4(model.learn_state.asdict()))
        logger.info('model.arch_id = %r' % (model.arch_id,))

        # create theano symbolic expressions that define the network
        theano_backprop = model.build_backprop_func()
        theano_forward = model.build_forward_func()

        # number of non-best iterations after, that triggers a best save
        # This prevents strings of best-saves one after another
        countdown_defaults = {
            'checkpoint': model.hyperparams['era_size'] * 2,
            'stop': model.hyperparams['stopping_patience'],
        }
        countdowns = {key: None for key in countdown_defaults.keys()}

        def check_countdown(key):
            if countdowns[key] is not None:
                if countdowns[key] > 0:
                    countdowns[key] -= 1
                else:
                    countdowns[key] = countdown_defaults[key]
                    return True

        model.history._new_era(model, X_train, y_train, X_train, y_train)
        printcol_info = utils.get_printcolinfo(model.requested_headers)
        utils.print_header_columns(printcol_info)
        tt = ut.Timer(verbose=False)

        # ---------------------------------------
        # EPOCH 0: Execute backwards and forward passes
        tt.tic()
        learn_info = model._epoch_validate_learn(
            theano_forward, X_learn, y_learn, w_learn
        )
        valid_info = model._epoch_validate(theano_forward, X_valid, y_valid, w_valid)

        # ---------------------------------------
        # EPOCH 0: Summarize the epoch
        epoch_info = {'epoch_num': epoch}
        epoch_info.update(**learn_info)
        epoch_info.update(**valid_info)
        epoch_info['duration'] = tt.toc()
        epoch_info['learn_state'] = model.learn_state.asdict()
        epoch_info['learnval_rat'] = epoch_info['learn_loss'] / epoch_info['valid_loss']

        # ---------------------------------------
        # EPOCH 0: Check how we are learning
        # Cache best results
        model.best_results['weights'] = model.get_all_param_values()
        model.best_results['epoch_num'] = epoch_info['epoch_num']
        if 'valid_precision' in epoch_info:
            model.best_results['valid_precision'] = epoch_info['valid_precision']
            model.best_results['valid_recall'] = epoch_info['valid_recall']
            model.best_results['valid_fscore'] = epoch_info['valid_fscore']
            model.best_results['valid_support'] = epoch_info['valid_support']
        for key in model.requested_headers:
            model.best_results[key] = epoch_info[key]

        # ---------------------------------------
        # EPOCH 0: Record this epoch in history and print info
        # model.history._record_epoch(epoch_info)
        utils.print_epoch_info(model, printcol_info, epoch_info)
        epoch += 1

        while True:
            try:
                # ---------------------------------------
                # Execute backwards and forward passes
                tt.tic()
                learn_info = model._epoch_learn(
                    theano_backprop, X_learn, y_learn, w_learn, epoch
                )
                if learn_info.get('diverged'):
                    break
                valid_info = model._epoch_validate(
                    theano_forward, X_valid, y_valid, w_valid
                )

                # ---------------------------------------
                # Summarize the epoch
                epoch_info = {'epoch_num': epoch}
                epoch_info.update(**learn_info)
                epoch_info.update(**valid_info)
                epoch_info['duration'] = tt.toc()
                epoch_info['learn_state'] = model.learn_state.asdict()
                epoch_info['learnval_rat'] = (
                    epoch_info['learn_loss'] / epoch_info['valid_loss']
                )

                # ---------------------------------------
                # Record this epoch in history
                model.history._record_epoch(epoch_info)

                # ---------------------------------------
                # Check how we are learning
                if epoch_info['valid_loss'] < model.best_results['valid_loss']:
                    # Found a better model. Reset countdowns.
                    for key in countdowns.keys():
                        countdowns[key] = countdown_defaults[key]
                    # Cache best results
                    model.best_results['weights'] = model.get_all_param_values()
                    model.best_results['epoch_num'] = epoch_info['epoch_num']
                    if 'valid_precision' in epoch_info:
                        model.best_results['valid_precision'] = epoch_info[
                            'valid_precision'
                        ]
                        model.best_results['valid_recall'] = epoch_info['valid_recall']
                        model.best_results['valid_fscore'] = epoch_info['valid_fscore']
                        model.best_results['valid_support'] = epoch_info['valid_support']
                    if 'learn_precision' in epoch_info:
                        model.best_results['learn_precision'] = epoch_info[
                            'learn_precision'
                        ]
                        model.best_results['learn_recall'] = epoch_info['learn_recall']
                        model.best_results['learn_fscore'] = epoch_info['learn_fscore']
                        model.best_results['learn_support'] = epoch_info['learn_support']
                    for key in model.requested_headers:
                        model.best_results[key] = epoch_info[key]

                # Check frequencies and countdowns
                checkpoint_flag = utils.checkfreq(
                    model.monitor_config['checkpoint_freq'], epoch
                )

                if check_countdown('checkpoint'):
                    countdowns['checkpoint'] = None
                    checkpoint_flag = True

                # ---------------------------------------
                # Output Diagnostics

                # Print the epoch
                utils.print_epoch_info(model, printcol_info, epoch_info)

                # Output any diagnostics
                if checkpoint_flag:
                    # FIXME: just move it to the second location
                    if model.monitor_config['monitor']:
                        model._dump_best_monitor()
                    model.checkpoint_save_model_info()
                    model.checkpoint_save_model_state()
                    model.save_model_info()
                    model.save_model_state()

                if model.monitor_config['monitor']:
                    model._dump_epoch_monitor()
                    # if epoch > 10:
                    # TODO: can dump case info every epoch
                    # But we want to dump the images less often
                    # Make function to just grab the failure case info
                    # and another function to visualize it.

                if model.monitor_config['monitor']:
                    if utils.checkfreq(model.monitor_config['weight_dump_freq'], epoch):
                        model._dump_weight_monitor()
                    if utils.checkfreq(model.monitor_config['case_dump_freq'], epoch):
                        model._dump_case_monitor(X_learn, y_learn, X_valid, y_valid)

                if check_countdown('stop'):
                    logger.info('Early stopping')
                    break

                # Check if the era is done
                max_era_size = model._fit_session['max_era_size']
                if model.history.current_era_size >= max_era_size:
                    # Decay learning rate
                    era = model.history.total_eras
                    rate_schedule = model.hyperparams['rate_schedule']
                    rate_schedule = ut.ensure_iterable(rate_schedule)
                    frac = rate_schedule[min(era, len(rate_schedule) - 1)]
                    model.learn_state.learning_rate = (
                        model.learn_state.learning_rate * frac
                    )
                    # Increase number of epochs in the next era
                    max_era_size = np.ceil(max_era_size / (frac ** 2))
                    model._fit_session['max_era_size'] = max_era_size
                    # Start a new era
                    model.history._new_era(model, X_train, y_train, X_train, y_train)

                    if model.hyperparams.get('era_clean', False):
                        y_learn = model._epoch_clean(
                            theano_forward, X_learn, y_learn, w_learn
                        )
                        y_valid = model._epoch_clean(
                            theano_forward, X_valid, y_valid, w_valid
                        )

                    utils.print_header_columns(printcol_info)

                # Break on max epochs
                if model.hyperparams['max_epochs'] is not None:
                    if epoch >= model.hyperparams['max_epochs']:
                        logger.info('\n[train] maximum number of epochs reached\n')
                        break
                # Increment the epoch
                epoch += 1

            except KeyboardInterrupt:
                logger.info('\n[train] Caught CRTL+C')
                logger.info('model.arch_id = %r' % (model.arch_id,))
                logger.info('learn_state = %s' % ut.repr4(model.learn_state.asdict()))
                from six.moves import input

                actions = ut.odict(
                    [
                        ('resume', (['0', 'r'], 'resume training')),
                        ('view', (['v', 'view'], 'view session directory')),
                        ('ipy', (['ipy', 'ipython', 'cmd'], 'embed into IPython')),
                        ('print', (['p', 'print'], 'print model state')),
                        ('shock', (['shock'], 'shock the network')),
                        ('save', (['s', 'save'], 'save best weights')),
                        ('quit', (['q', 'exit', 'quit'], 'quit')),
                    ]
                )
                while True:
                    # prompt
                    msg_list = [
                        'enter %s to %s'
                        % (ut.conj_phrase(ut.lmap(repr, map(str, tup[0])), 'or'), tup[1])
                        for key, tup in actions.items()
                    ]
                    msg = ut.indentjoin(msg_list, '\n | * ')
                    msg = ''.join([' +-----------', msg, '\n L-----------\n'])
                    logger.info(msg)
                    #
                    ans = str(input()).strip()

                    # We have a resolution
                    if ans in actions['quit'][0]:
                        logger.info('quit training...')
                        return
                    elif ans in actions['resume'][0]:
                        break
                    elif ans in actions['ipy'][0]:
                        ut.embed()
                    elif ans in actions['save'][0]:
                        # Save the weights of the network
                        model.checkpoint_save_model_info()
                        model.checkpoint_save_model_state()
                        model.save_model_info()
                        model.save_model_state()
                    elif ans in actions['print'][0]:
                        model.print_state_str()
                    elif ans in actions['shock'][0]:
                        utils.shock_network(model.output_layer)
                        model.learn_state.learning_rate = (
                            model.learn_state.learning_rate * 2
                        )
                    elif ans in actions['view'][0]:
                        session_dpath = model._fit_session['session_dpath']
                        ut.view_directory(session_dpath)
                    else:
                        continue
                    # Handled the resolution
                    logger.info('resuming training...')
                    break
            except (IndexError, ValueError, Exception) as ex:
                ut.printex(ex, 'Error Occurred Embedding to enable debugging', tb=True)
                errorstate = {'is_fixed': False}
                # is_fixed = False
                import utool

                utool.embed()
                if not errorstate['is_fixed']:
                    raise
        # Save the best network
        model.checkpoint_save_model_state()
        model.save_model_state()

        # Set model to best weights
        model.set_all_param_values(model.best_results['weights'])

        # # Remove history after overfitting starts
        # if 'epoch_num' not in model.best_results:
        #     model.best_results['epoch_num'] = model.best_results['epoch']
        # epoch_num = model.best_results['epoch_num']
        # model.history.rewind_to(epoch_num)

        if X_test is not None and y_test is not None:
            # TODO: dump test output in a standard way
            w_test = model._default_input_weights(X_test, y_test)
            theano_forward = model.build_forward_func()
            info = model._epoch_validate(theano_forward, X_test, y_test, w_test)
            logger.info('train info = %r' % (info,))
            model.dump_cases(X_test, y_test, 'test', dpath=model.arch_dpath)
            # model._run_test(X_test, y_test)

    # def _run_test(model, X_test, y_test):
    #    # Perform a test on the fitted model
    #    test_outptuts = model._predict(X_test)
    #    y_pred = test_outptuts['predictions']
    #    logger.info(model.name)
    #    report = sklearn.metrics.classification_report(
    #        y_true=y_test, y_pred=y_pred,
    #    )
    #    logger.info(report)
    #    pass

    def _ensure_learnval_split(
        model, X_train, y_train, X_valid=None, y_valid=None, valid_idx=None
    ):
        if X_valid is not None:
            assert valid_idx is None, 'Cant specify both valid_idx and X_valid'
            # When X_valid is given assume X_train is actually X_learn
            X_learn = X_train
            y_learn = y_train
        else:
            if valid_idx is None:
                # Split training set into a learning / validation set
                from wbia_cnn.dataset import stratified_shuffle_split

                train_idx, valid_idx = stratified_shuffle_split(
                    y_train, fractions=[0.7, 0.3], rng=432321
                )
                # import sklearn.cross_validation
                # xvalkw = dict(n_folds=2, shuffle=True, random_state=43432)
                # skf = sklearn.cross_validation.StratifiedKFold(y_train, **xvalkw)
                # train_idx, valid_idx = list(skf)[0]
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
        # logger.info('\n[train] --- MODEL INFO ---')
        # model.print_arch_str()
        # model.print_layer_info()
        return X_learn, y_learn, X_valid, y_valid

    def ensure_data_params(model, X_learn, y_learn):
        if model.data_params is None:
            model.data_params = {}

        # TODO: move to dataset. This is independant of the model.
        if model.hyperparams['whiten_on']:
            # Center the data by subtracting the mean
            if 'center_mean' not in model.data_params:
                logger.info('computing center mean/std. (hacks std=1)')
                X_ = X_learn.astype(np.float32)
                try:
                    if ut.is_int(X_learn):
                        ut.assert_inbounds(X_learn, 0, 255, eq=True, verbose=ut.VERBOSE)
                        X_ = X_ / 255
                    ut.assert_inbounds(X_, 0.0, 1.0, eq=True, verbose=ut.VERBOSE)
                except ValueError:
                    logger.info('[WARNING] Input bounds check failed...')
                # Ensure that the mean is computed on 0-1 normalized data
                model.data_params['center_mean'] = np.mean(X_, axis=0)
                model.data_params['center_std'] = 1.0

            # Hack to preconvert mean / std to 0-1 for old models
            model._fix_center_mean_std()
        else:
            ut.delete_dict_keys(model.data_params, ['center_mean', 'center_std'])

        if model.hyperparams['class_weight'] == 'balanced':
            logger.info('Balancing class weights')
            import sklearn.utils

            unique_classes = np.array(sorted(ut.unique(y_learn)))
            class_to_weight = sklearn.utils.compute_class_weight(
                'balanced',
                classes=unique_classes,
                y=y_learn,
            )
            model.data_params['class_to_weight'] = class_to_weight
        else:
            ut.delete_dict_keys(model.data_params, ['class_to_weight'])

        if model.hyperparams['label_encode_on']:
            if getattr(model, 'encoder', None) is None:
                if hasattr(model, 'init_encoder'):
                    model.init_encoder(y_learn)

    def _rename_old_sessions(model):
        import re

        dpath_list = ut.glob(model.saved_session_dpath, '*', with_files=False)
        for dpath in dpath_list:
            if True or not re.match('^.*_nEpochs_[0-9]*$', dpath):
                report_fpath = join(dpath, 'era_history.json')
                if exists(report_fpath):
                    report_dict = ut.load_data(report_fpath)
                    # TODO: try to read from history report
                    nEpochs = report_dict['nEpochs']
                else:
                    nEpochs = len(ut.glob(join(dpath, 'history'), 'loss_*'))
                # Add suffix to session to indicate what happened?
                # Maybe this should be done via symlink?
                dpath_new = dpath + '_nEpochs_%04d' % (nEpochs,)
                ut.move(dpath, dpath_new)

        model.saved_session_dpath
        pass

    def _new_fit_session(model):
        """
        Starts a model training session
        """
        logger.info('Starting new fit session')
        model._fit_session = {
            'start_time': ut.get_timestamp(),
            'max_era_size': model.hyperparams['era_size'],
            'era_epoch_num': 0,
        }
        # TODO: ensure this somewhere else?
        model._rng = ut.ensure_rng(model.hyperparams['random_seed'])

        if model.monitor_config['monitor']:
            ut.ensuredir(model.arch_dpath)

            # Rename old sessions to distinguish this one
            # TODO: put a lock file on any existing sessions
            model._rename_old_sessions()

            # Create a directory for this training session with a timestamp
            session_dname = 'fit_session_' + model._fit_session['start_time']
            session_dpath = join(model.saved_session_dpath, session_dname)
            session_dpath = ut.get_nonconflicting_path(
                session_dpath, offset=1, suffix='_conflict%d'
            )

            prog_dirs = {
                'dream': join(session_dpath, 'dream'),
                'loss': join(session_dpath, 'history'),
                'weights': join(session_dpath, 'weights'),
            }

            model._fit_session.update(
                **{
                    'prog_dirs': prog_dirs,
                }
            )

            # for dpath in prog_dirs.values():
            #    ut.ensuredir(dpath)

            ut.ensuredir(session_dpath)
            model._fit_session['session_dpath'] = session_dpath

            if ut.get_argflag('--vd'):
                # Open session in file explorer
                ut.view_directory(session_dpath)

            # Make a symlink to the latest session
            session_link = join(model.arch_dpath, 'latest_session')
            ut.symlink(session_dpath, session_link, overwrite=True)

            # Write backprop arch info to arch root
            back_archinfo_fpath = join(session_dpath, 'arch_info_fit.json')
            back_archinfo_json = model.make_arch_json(with_noise=True)
            ut.writeto(back_archinfo_fpath, back_archinfo_json, verbose=True)

            # Write feed-forward arch info to arch root
            pred_archinfo_fpath = join(session_dpath, 'arch_info_predict.json')
            pred_archinfo_json = model.make_arch_json(with_noise=False)
            ut.writeto(pred_archinfo_fpath, pred_archinfo_json, verbose=False)

            # Write arch graph to root
            try:
                back_archimg_fpath = join(session_dpath, 'arch_graph_fit.jpg')
                model.imwrite_arch(fpath=back_archimg_fpath, fullinfo=False)
                model._overwrite_latest_image(back_archimg_fpath, 'arch_graph')
            except Exception as ex:
                ut.printex(ex, iswarning=True)

            # Write initial states of the weights
            try:
                ut.ensuredir(prog_dirs['weights'])
                fig = model.show_weights_image(fnum=2)
                fpath = join(
                    prog_dirs['weights'], 'weights_' + model.history.hist_id + '.png'
                )
                fig.savefig(fpath)
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
        if session_dpath != dirname(fpath):
            # Copy latest image to the session dir if it isn't there
            shutil.copy(fpath, join(session_dpath, 'latest_' + new_name + ext))
        # Copy latest image to the main arch dir
        shutil.copy(fpath, join(model.arch_dpath, 'latest_' + new_name + ext))

    def _dump_case_monitor(model, X_learn, y_learn, X_valid, y_valid):
        prog_dirs = model._fit_session['prog_dirs']

        try:
            model.dump_cases(X_learn, y_learn, 'learn')
        except Exception:
            logger.info('WARNING: DUMP CASES HAS FAILED')
            pass
        try:
            model.dump_cases(X_valid, y_valid, 'valid')
        except Exception:
            logger.info('WARNING: DUMP CASES HAS FAILED')
            pass
        if False:
            try:
                # Save class dreams
                ut.ensuredir(prog_dirs['dream'])
                fpath = join(
                    prog_dirs['dream'], 'class_dream_' + model.history.hist_id + '.png'
                )
                fig = model.show_class_dream(fnum=4)
                fig.savefig(fpath, dpi=180)
                model._overwrite_latest_image(fpath, 'class_dream')
            except Exception as ex:
                ut.printex(ex, 'failed to dump dream', iswarning=True)

    def _dump_weight_monitor(model):
        prog_dirs = model._fit_session['prog_dirs']
        try:
            # Save weights images
            ut.ensuredir(prog_dirs['weights'])
            fpath = join(
                prog_dirs['weights'], 'weights_' + model.history.hist_id + '.png'
            )
            fig = model.show_weights_image(fnum=2)
            fig.savefig(fpath, dpi=180)
            model._overwrite_latest_image(fpath, 'weights')
        except Exception as ex:
            ut.printex(ex, 'failed to dump weights', iswarning=True)

    def get_report_json(model):
        report_dict = {}
        report_dict['best'] = ut.delete_keys(model.best_results.copy(), ['weights'])
        for key in report_dict['best'].keys():
            if hasattr(report_dict['best'][key], 'tolist'):
                report_dict['best'][key] = report_dict['best'][key].tolist()
        if len(model.history) > 0:
            report_dict['num_learn'] = model.history.era_list[-1]['num_learn']
            report_dict['num_valid'] = model.history.era_list[-1]['num_valid']
        report_dict['hyperparams'] = model.hyperparams
        report_dict['arch_hashid'] = model.get_arch_hashid()
        report_dict['model_name'] = model.name
        report_json = ut.repr2_json(report_dict, nl=2, precision=4)
        return report_json

    def _dump_best_monitor(model):
        session_dpath = model._fit_session['session_dpath']
        # Save text best info
        report_fpath = join(session_dpath, 'best_report.json')
        report_json = model.get_report_json()
        ut.write_to(report_fpath, report_json, verbose=False)

    def _dump_epoch_monitor(model):
        prog_dirs = model._fit_session['prog_dirs']
        session_dpath = model._fit_session['session_dpath']

        # Save text history info
        text_fpath = join(session_dpath, 'era_history.txt')
        history_text = model.history.to_json()
        ut.write_to(text_fpath, history_text, verbose=False)

        # Save loss graphs
        try:
            ut.ensuredir(prog_dirs['loss'])
            fpath = join(prog_dirs['loss'], 'loss_' + model.history.hist_id + '.png')
            fig = model.show_loss_history(fnum=1)
            fig.savefig(fpath, dpi=180)
            model._overwrite_latest_image(fpath, 'loss')
        except Exception as ex:
            ut.printex(ex, 'failed to dump loss', iswarning=True)
            raise

        try:
            ut.ensuredir(prog_dirs['loss'])
            fpath = join(prog_dirs['loss'], 'pr_' + model.history.hist_id + '.png')
            fig = model.show_pr_history(fnum=4)
            fig.savefig(fpath, dpi=180)
            model._overwrite_latest_image(fpath, 'pr')
        except Exception as ex:
            ut.printex(ex, 'failed to dump pr', iswarning=True)
            raise

        # Save weight updates
        try:
            ut.ensuredir(prog_dirs['loss'])
            fpath = join(
                prog_dirs['loss'], 'update_mag_' + model.history.hist_id + '.png'
            )
            fig = model.show_update_mag_history(fnum=3)
            fig.savefig(fpath, dpi=180)
            model._overwrite_latest_image(fpath, 'update_mag')
        except Exception as ex:
            ut.printex(ex, 'failed to dump update mags ', iswarning=True)

    def _epoch_learn(model, theano_backprop, X_learn, y_learn, w_learn, epoch):
        """
        Backwards propogate -- Run learning set through the backwards pass

        Ignore:
            >>> from wbia_cnn.models.abstract_models import *  # NOQA
            >>> from wbia_cnn.models import mnist
            >>> import theano
            >>> model, dataset = mnist.testdata_mnist(dropout=.5)
            >>> model.monitor_config['monitor'] = False
            >>> model.monitor_config['showprog'] = True
            >>> model._behavior['buffered'] = False
            >>> model.init_arch()
            >>> model.learn_state.init()
            >>> batch_size = 16
            >>> X_learn, y_learn = dataset.subset('test')
            >>> model.ensure_data_params(X_learn, y_learn)
            >>> class_to_weight = model.data_params['class_to_weight']
            >>> class_to_weight.take(y_learn)
            >>> w_learn = class_to_weight.take(y_learn).astype(np.float32)
            >>> model._new_fit_session()
            >>> theano_backprop = model.build_backprop_func()
        """
        buffered = model._behavior['buffered']
        augment_on = model.hyperparams.get('augment_on', True)
        if epoch <= model.hyperparams['augment_delay']:
            # Dont augment in the first few epochs so the model can start to
            # get somewhere. This will hopefully help training initialize
            # faster.
            augment_on = False

        learn_outputs = model.process_batch(
            theano_backprop,
            X_learn,
            y_learn,
            w_learn,
            shuffle=True,
            augment_on=augment_on,
            buffered=buffered,
        )

        # average loss over all learning batches
        learn_info = {}
        learn_info['learn_loss'] = learn_outputs['loss'].mean()
        learn_info['learn_loss_std'] = learn_outputs['loss'].std()

        if 'loss_reg' in learn_outputs:
            # Regularization information
            learn_info['learn_loss_reg'] = learn_outputs['loss_reg']
            reg_amount = learn_outputs['loss_reg'] - learn_outputs['loss']
            reg_ratio = reg_amount / learn_outputs['loss']
            reg_percent = reg_amount / learn_outputs['loss_reg']

            if 'accuracy' in learn_outputs:
                learn_info['learn_acc'] = learn_outputs['accuracy'].mean()
                learn_info['learn_acc_std'] = learn_outputs['accuracy'].std()
            if 'predictions' in learn_outputs:
                try:
                    p, r, f, s = sklearn.metrics.precision_recall_fscore_support(
                        y_true=learn_outputs['auglbl_list'],
                        y_pred=learn_outputs['predictions'],
                    )
                except ValueError:
                    p, r, f, s = 0.0, 0.0, 0.0, 0.0
                # report = sklearn.metrics.classification_report(
                #    y_true=learn_outputs['auglbl_list'], y_pred=learn_outputs['predictions']
                # )
                learn_info['learn_precision'] = p
                learn_info['learn_recall'] = r
                learn_info['learn_fscore'] = f
                learn_info['learn_support'] = s

            learn_info['reg_percent'] = reg_percent
            learn_info['reg_ratio'] = reg_ratio

        param_update_mags = {}
        for key, val in learn_outputs.items():
            if key.startswith('param_update_magnitude_'):
                key_ = key.replace('param_update_magnitude_', '')
                param_update_mags[key_] = (val.mean(), val.std())
        if param_update_mags:
            learn_info['param_update_mags'] = param_update_mags

        # If the training loss is nan, the training has diverged
        if np.isnan(learn_info['learn_loss']):
            logger.info('\n[train] train loss is Nan. training diverged\n')
            logger.info('learn_outputs = %r' % (learn_outputs,))
            logger.info('\n[train] train loss is Nan. training diverged\n')
            """
            from wbia_cnn import draw_net
            draw_net.imwrite_theano_symbolic_graph(theano_backprop)
            """
            # imwrite_theano_symbolic_graph(thean_expr):
            learn_info['diverged'] = True
        return learn_info

    def _epoch_validate_learn(model, theano_forward, X_learn, y_learn, w_learn):
        """
        Forwards propagate -- Run validation set through the forwards pass
        """
        augment_on = model.hyperparams.get('augment_on_validate', False)
        learn_outputs = model.process_batch(
            theano_forward, X_learn, y_learn, w_learn, augment_on=augment_on
        )
        # average loss over all learning batches
        learn_info = {}
        learn_info['learn_loss'] = learn_outputs['loss_determ'].mean()
        learn_info['learn_loss_std'] = learn_outputs['loss_determ'].std()

        if 'loss_reg' in learn_outputs:
            # Regularization information
            learn_info['learn_loss_reg'] = learn_outputs['loss_reg']
            reg_amount = learn_outputs['loss_reg'] - learn_outputs['loss']
            reg_ratio = reg_amount / learn_outputs['loss']
            reg_percent = reg_amount / learn_outputs['loss_reg']

            if 'accuracy' in learn_outputs:
                learn_info['learn_acc'] = learn_outputs['accuracy'].mean()
                learn_info['learn_acc_std'] = learn_outputs['accuracy'].std()
            if 'predictions' in learn_outputs:
                try:
                    p, r, f, s = sklearn.metrics.precision_recall_fscore_support(
                        y_true=learn_outputs['auglbl_list'],
                        y_pred=learn_outputs['predictions'],
                    )
                except ValueError:
                    p, r, f, s = 0.0, 0.0, 0.0, 0.0
                # report = sklearn.metrics.classification_report(
                #    y_true=learn_outputs['auglbl_list'], y_pred=learn_outputs['predictions']
                # )
                learn_info['learn_precision'] = p
                learn_info['learn_recall'] = r
                learn_info['learn_fscore'] = f
                learn_info['learn_support'] = s

            learn_info['reg_percent'] = reg_percent
            learn_info['reg_ratio'] = reg_ratio

        param_update_mags = {}
        for key, val in learn_outputs.items():
            if key.startswith('param_update_magnitude_'):
                key_ = key.replace('param_update_magnitude_', '')
                param_update_mags[key_] = (val.mean(), val.std())
        if param_update_mags:
            learn_info['param_update_mags'] = param_update_mags
        return learn_info

    def _epoch_validate(model, theano_forward, X_valid, y_valid, w_valid):
        """
        Forwards propagate -- Run validation set through the forwards pass
        """
        augment_on = model.hyperparams.get('augment_on_validate', False)
        valid_outputs = model.process_batch(
            theano_forward, X_valid, y_valid, w_valid, augment_on=augment_on
        )
        valid_info = {}
        valid_info['valid_loss'] = valid_outputs['loss_determ'].mean()
        valid_info['valid_loss_std'] = valid_outputs['loss_determ'].std()
        if 'valid_acc' in model.requested_headers:
            valid_info['valid_acc'] = valid_outputs['accuracy'].mean()
            valid_info['valid_acc_std'] = valid_outputs['accuracy'].std()
        if 'predictions' in valid_outputs:
            try:
                p, r, f, s = sklearn.metrics.precision_recall_fscore_support(
                    y_true=valid_outputs['auglbl_list'],
                    y_pred=valid_outputs['predictions'],
                )
            except ValueError:
                p, r, f, s = 0.0, 0.0, 0.0, 0.0
            valid_info['valid_precision'] = p
            valid_info['valid_recall'] = r
            valid_info['valid_fscore'] = f
            valid_info['valid_support'] = s
        return valid_info

    def _epoch_clean(
        model, theano_forward, X_general, y_general, w_general, conf_thresh=0.95
    ):
        """
        Forwards propogate -- Run set through the forwards pass and clean
        """
        augment_on = model.hyperparams.get('augment_on_validate', False)
        valid_outputs = model.process_batch(
            theano_forward, X_general, y_general, w_general, augment_on=augment_on
        )
        predictions = valid_outputs['predictions']
        confidences = valid_outputs['confidences']
        y_cleaned = np.array(
            [
                pred if y != pred and conf > conf_thresh else y
                for y, pred, conf in zip(y_general, predictions, confidences)
            ]
        )
        num_cleaned = len(np.nonzero(y_general != y_cleaned)[0])
        logger.info('Cleaned %d instances' % (num_cleaned,))
        return y_cleaned

    def dump_cases(model, X, y, subset_id='unknown', dpath=None):
        """
        For each class find:
            * the most-hard  failures
            * the mid-level failures
            * the critical cases (least-hard failures / most-hard successes)
            * the mid-level successes
            * the least-hard successs

        """
        import vtool as vt
        import pandas as pd

        logger.info('Dumping %s cases' % (subset_id,))
        # pd.set_option("display.max_rows", 20)
        # pd.set_option("display.precision", 2)
        # pd.set_option('expand_frame_repr', False)
        # pd.set_option('display.float_format', lambda x: '%.2f' % x)

        if dpath is None:
            dpath = model._fit_session['session_dpath']
        case_dpath = ut.ensuredir((dpath, 'cases', model.history.hist_id, subset_id))

        y_true = y
        netout = model._predict(X)
        y_conf = netout['network_output_determ']
        data_idx = np.arange(len(y))
        y_pred = y_conf.argmax(axis=1)

        if getattr(model, 'encoder', None):
            class_idxs = model.encoder.transform(model.encoder.classes_)
            class_lbls = model.encoder.classes_
        else:
            class_idxs = list(range(model.output_dims))
            class_lbls = list(range(model.output_dims))
        target_classes = ut.take(class_lbls, class_idxs)

        index = pd.Series(data_idx, name='data_idx')
        decision = pd.DataFrame(y_conf, index=index, columns=target_classes)

        easiness = np.array(ut.ziptake(decision.values, y_true))
        columns = ['pred', 'target', 'easiness']
        column_data = [y_pred, y_true, easiness]
        data = dict(zip(columns, column_data))
        df = pd.DataFrame(data, index, columns)
        df['failed'] = df['pred'] != df['target']

        def target_partition(df, target):
            df_chunk = df if target is None else df[df['target'] == target]
            df_chunk = df_chunk.take(df_chunk['easiness'].argsort()[::-1])
            return df_chunk

        def snapped_slice(size, frac, n):
            start = int(size * frac - np.ceil(n / 2))
            stop = int(size * frac + np.floor(n / 2))
            buf = 0
            if stop >= size:
                buf = size - stop - 1
            elif start < 0:
                buf = 0 - start
            stop += buf
            start += buf
            assert stop < size, 'out of bounds'
            sl = slice(start, stop)
            return sl

        for y in class_idxs:
            class_case_dpath = ut.ensuredir((case_dpath, 'class_%r' % (y,)))
            df_chunk = target_partition(df, y)
            # Find the first failure
            if any(df_chunk['failed']):
                critical_index = np.where(df_chunk['failed'])[0][0]
            else:
                critical_index = 0
            critical_frac = critical_index / len(df_chunk)
            midlevel_easy = critical_frac / 2
            midlevel_hard = critical_frac + (1 - critical_frac) / 2
            # 0 is easy, 1 is hard
            fracs = [0, midlevel_easy, critical_frac, midlevel_hard, 1.0]
            n = 4
            size = len(df_chunk)
            slices = [snapped_slice(size, frac, n) for frac in fracs]

            hard_idx = np.array(
                ut.unique(
                    ut.flatten([list(range(*sl.indices(len(df_chunk)))) for sl in slices])
                )
            )
            hard_frac = hard_idx / len(df_chunk)
            selected = df_chunk.iloc[hard_idx]

            selected['hard_idx'] = hard_idx
            selected['hard_frac'] = hard_frac

            for idx, row in selected.iterrows():
                img = X[idx]
                type_ = 'fail' if row['failed'] else 'success'
                fname = 'hardidx_%04d_hardfrac_%.2f_pred_%d_case_%s.jpg' % (
                    row['hard_idx'],
                    row['hard_frac'],
                    row['pred'],
                    type_,
                )
                fpath = join(class_case_dpath, fname)
                vt.imwrite(fpath, img)


@ut.reloadable_class
class _BatchUtility(object):
    @classmethod
    def expand_data_indicies(cls, label_idx, data_per_label=1):
        """
        when data_per_label > 1, gives the corresponding data indicies for the
        data indicies
        """
        expanded_idx = [
            label_idx * data_per_label + count for count in range(data_per_label)
        ]
        data_idx = np.vstack(expanded_idx).T.flatten()
        return data_idx

    @classmethod
    def shuffle_input(cls, X, y, w, data_per_label=1, rng=None):
        rng = ut.ensure_rng(rng)
        num_labels = X.shape[0] // data_per_label
        label_idx = ut.random_indexes(num_labels, rng=rng)
        data_idx = cls.expand_data_indicies(label_idx, data_per_label)
        X = X.take(data_idx, axis=0)
        X = np.ascontiguousarray(X)
        if y is not None:
            y = y.take(label_idx, axis=0)
            y = np.ascontiguousarray(y)
        if w is not None:
            w = w.take(label_idx, axis=0)
            w = np.ascontiguousarray(w)
        return X, y, w

    @classmethod
    def slice_batch(
        cls, X, y, w, batch_size, batch_index, data_per_label=1, wraparound=False
    ):
        start_x = batch_index * batch_size
        end_x = (batch_index + 1) * batch_size
        # Take full batch of images and take the fraction of labels if
        # data_per_label > 1
        x_sl = slice(start_x, end_x)
        y_sl = slice(start_x // data_per_label, end_x // data_per_label)
        Xb = X[x_sl]
        yb = y if y is None else y[y_sl]
        wb = w if w is None else w[y_sl]
        if wraparound:
            # Append extra data to ensure the batch size is full
            if Xb.shape[0] != batch_size:
                extra = batch_size - Xb.shape[0]
                Xb_wrap = X[slice(0, extra)]
                Xb = np.concatenate([Xb, Xb_wrap], axis=0)
                if yb is not None:
                    yb_wrap = y[slice(0, extra // data_per_label)]
                    yb = np.concatenate([yb, yb_wrap], axis=0)
                if wb is not None:
                    wb_wrap = w[slice(0, extra // data_per_label)]
                    wb = np.concatenate([wb, wb_wrap], axis=0)
        return Xb, yb, wb

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

    def _stack_outputs(model, theano_fn, output_list):
        """
        Combines outputs across batches and returns them in a dictionary keyed
        by the theano variable output name.
        """
        import vtool as vt

        output_vars = [outexpr.variable for outexpr in theano_fn.outputs]
        output_names = [str(var) if var.name is None else var.name for var in output_vars]
        unstacked_output_gen = [
            [bop[count] for bop in output_list] for count, name in enumerate(output_names)
        ]
        stacked_output_list = [
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
            outputs[key] = outputs[key][0 : int(num_outputs)]
        return outputs


@ut.reloadable_class
class _ModelBatch(_BatchUtility):
    def _init_batch_vars(model, kwargs):
        model.pad_labels = False
        model.X_is_cv2_native = True

    def process_batch(
        model,
        theano_fn,
        X,
        y=None,
        w=None,
        buffered=False,
        unwrap=False,
        shuffle=False,
        augment_on=False,
    ):
        """Execute a theano function on batches of X and y"""
        # Break data into generated batches
        # TODO: sliced batches when there is no shuffling
        # Create an iterator to generate batches of data
        batch_iter = model.batch_iterator(X, y, w, shuffle=shuffle, augment_on=augment_on)
        if buffered:
            batch_iter = ut.buffered_generator(batch_iter)
        if model.monitor_config['showprog']:
            num_batches = (X.shape[0] + model.batch_size - 1) // model.batch_size
            batch_iter = ut.ProgIter(
                batch_iter,
                nTotal=num_batches,
                lbl=theano_fn.name,
                freq=10,
                bs=True,
                adjust=True,
            )

        # Execute the function with either known or unknown y-targets
        output_list = []
        if y is None:
            aug_yb_list = None
            for Xb, yb, wb in batch_iter:
                batch_label = theano_fn(Xb)
                output_list.append(batch_label)
        else:
            aug_yb_list = []
            for Xb, yb, wb in batch_iter:
                batch_label = theano_fn(Xb, yb, wb)
                output_list.append(batch_label)
                aug_yb_list.append(yb)

        # Combine results of batches into one big result
        outputs = model._stack_outputs(theano_fn, output_list)
        if y is not None:
            # Hack in special outputs
            if isinstance(model, AbstractVectorVectorModel):
                auglbl_list = np.array(aug_yb_list)
            elif isinstance(model, AbstractVectorModel):
                auglbl_list = np.vstack(aug_yb_list)
            else:
                auglbl_list = np.hstack(aug_yb_list)
            outputs['auglbl_list'] = auglbl_list
        if unwrap:
            # slice of batch induced padding
            outputs = model._unwrap_outputs(outputs, X)
        return outputs

    @profile
    def batch_iterator(model, X, y=None, w=None, shuffle=False, augment_on=False):
        """
        Example:
            >>> # ENABLE_DOCTEST
            >>> from wbia_cnn.models.abstract_models import *  # NOQA
            >>> from wbia_cnn import models
            >>> model = models.DummyModel(batch_size=16)
            >>> X, y = model.make_random_testdata(num=37, cv2_format=True)
            >>> model.ensure_data_params(X, y)
            >>> result_list = [(Xb, Yb) for Xb, Yb, _ in model.batch_iterator(X, y)]
            >>> Xb, yb = result_list[0]
            >>> assert np.all(X[0, :, :, 0] == Xb[0, 0, :, :])
            >>> result = ut.depth_profile(result_list, compress_consecutive=True)
            >>> print(result)
            [[(16, 1, 4, 4), 16]] * 2 + [[(5, 1, 4, 4), 5]]

        Example:
            >>> # ENABLE_DOCTEST
            >>> from wbia_cnn.models.abstract_models import *  # NOQA
            >>> from wbia_cnn import models
            >>> model = models.DummyModel(batch_size=16)
            >>> X, y = model.make_random_testdata(num=37, cv2_format=False, asint=True)
            >>> model.X_is_cv2_native = False
            >>> model.ensure_data_params(X, y)
            >>> result_list = [(Xb, Yb) for Xb, Yb, _ in model.batch_iterator(X, y)]
            >>> Xb, yb = result_list[0]
            >>> assert np.all(np.isclose(X[0] / 255, Xb[0]))
            >>> result = ut.depth_profile(result_list, compress_consecutive=True)
            >>> print(result)
        """
        # need to be careful with batchsizes if directly specified to theano
        batch_size = model.batch_size
        data_per_label = model.data_per_label_input
        wraparound = model.input_shape[0] is not None
        model._validate_labels(X, y, w)

        num_batches = (X.shape[0] + batch_size - 1) // batch_size

        if shuffle:
            rng = model._rng
            X, y, w = model.shuffle_input(X, y, w, data_per_label, rng=rng)

        is_int = ut.is_int(X)
        is_cv2 = model.X_is_cv2_native
        whiten_on = model.hyperparams['whiten_on']

        # Slice and preprocess data in batch
        for batch_index in range(num_batches):
            # Take a slice from the data
            Xb_, yb_, wb_ = model.slice_batch(
                X, y, w, batch_size, batch_index, data_per_label, wraparound
            )
            # Prepare data for the GPU
            Xb, yb, wb = model._prepare_batch(
                Xb_,
                yb_,
                wb_,
                is_int=is_int,
                is_cv2=is_cv2,
                augment_on=augment_on,
                whiten_on=whiten_on,
            )
            yield Xb, yb, wb

    def _prepare_batch(
        model, Xb_, yb_, wb_, is_int=True, is_cv2=True, augment_on=False, whiten_on=False
    ):
        if augment_on:
            has_encoder = getattr(model, 'encoder', None) is not None
            yb_ = model.encoder.inverse_transform(yb_) if has_encoder else yb_
            if model.hyperparams['augment_weights']:
                Xb_, yb_, wb_ = model.augment(Xb_, yb_, wb_)
            else:
                Xb_, yb_ = model.augment(Xb_, yb_)
            yb_ = model.encoder.transform(yb_) if has_encoder else yb_
        Xb = Xb_.astype(np.float32, copy=True)
        yb = None if yb_ is None else yb_.astype(np.int32, copy=True)
        wb = None if wb_ is None else wb_.astype(np.float32, copy=False)
        if is_int:
            # Rescale the batch data to the range 0 to 1
            Xb = Xb / 255.0
        if whiten_on:
            mean = model.data_params['center_mean']
            std = model.data_params['center_std']
            # assert np.all(mean <= 1.0)
            # assert np.all(std <= 1.0)
            np.subtract(Xb, mean, out=Xb)
            np.divide(Xb, std, out=Xb)
            # Xb = (Xb - mean) / (std)
        if is_cv2 and len(Xb.shape) == 4:
            # Convert from cv2 to lasagne format
            Xb = Xb.transpose((0, 3, 1, 2))
        if yb is not None:
            # if encoder is not None:
            #     # Apply an encoding if applicable
            #     yb = encoder.transform(yb).astype(np.int32)
            if model.data_per_label_input > 1 and model.pad_labels:
                # Pad data for siamese networks
                yb = model._pad_labels(yb)
        return Xb, yb, wb

    def prepare_data(model, X, y=None, w=None):
        """convenience function for external use"""
        is_int = ut.is_int(X)
        is_cv2 = model.X_is_cv2_native
        whiten_on = model.hyperparams['whiten_on']
        Xb, yb, wb = model._prepare_batch(
            X, y, w, is_int=is_int, is_cv2=is_cv2, whiten_on=whiten_on
        )
        if y is None:
            return Xb
        elif w is None:
            return Xb, yb
        else:
            return Xb, yb, wb


class _ModelPredicter(object):
    def _predict(model, X_test):
        """
        Returns all prediction outputs of the network in a dictionary.
        """
        if ut.VERBOSE:
            logger.info('\n[train] --- MODEL INFO ---')
            model.print_arch_str()
            model.print_layer_info()
            logger.info('\n[test] predict with batch size %0.1f' % (model.batch_size))
        # create theano symbolic expressions that define the network
        theano_predict = model.build_predict_func()
        # Begin testing with the neural network
        test_outputs = model.process_batch(theano_predict, X_test, unwrap=True)
        return test_outputs

    def predict_proba(model, X_test):
        test_outputs = model._predict(X_test)
        y_proba = test_outputs['network_output_determ']
        return y_proba

    def predict_proba_Xb(model, Xb):
        """Accepts prepared inputs"""
        theano_predict = model.build_predict_func()
        batch_label = theano_predict(Xb)
        output_names = [
            str(outexpr.variable)
            if outexpr.variable.name is None
            else outexpr.variable.name
            for outexpr in theano_predict.outputs
        ]
        test_outputs = dict(zip(output_names, batch_label))
        y_proba = test_outputs['network_output_determ']
        return y_proba

    def predict(model, X_test):
        test_outputs = model._predict(X_test)
        y_predict = test_outputs['predictions']
        return y_predict


class _ModelBackend(object):
    """
    Functions that build and compile theano exepressions
    """

    def _init_compile_vars(model, kwargs):
        model._theano_exprs = ut.ddict(lambda: None)
        model._theano_backprop = None
        model._theano_forward = None
        model._theano_predict = None
        model._theano_mode = None
        # theano.compile.FAST_COMPILE
        # theano.compile.FAST_RUN

    def build(model):
        logger.info('[model] --- BUILDING SYMBOLIC THEANO FUNCTIONS ---')
        model.build_backprop_func()
        model.build_forward_func()
        model.build_predict_func()
        logger.info('[model] --- FINISHED BUILD ---')

    def build_predict_func(model):
        """Computes predictions given unlabeled data"""
        if model._theano_predict is None:
            logger.info('[model.build] request_predict')
            netout_exprs = model._get_network_output()
            network_output_determ = netout_exprs['network_output_determ']
            unlabeled_outputs = model._get_unlabeled_outputs()

            fn_inputs = model._theano_fn_inputs
            X_batch, X_given = ut.take(fn_inputs, ['X_batch', 'X_given'])

            theano_predict = theano.function(
                inputs=[theano.In(X_batch)],
                outputs=[network_output_determ] + unlabeled_outputs,
                givens={X_given: X_batch},
                updates=None,
                mode=model._theano_mode,
                name=':predict',
            )
            model._theano_predict = theano_predict
        return model._theano_predict

    def build_forward_func(model):
        """
        Computes loss, but does not learn.
        Returns diagnostic information.

        Ignore:
            >>> from wbia_cnn.models.abstract_models import *  # NOQA
            >>> from wbia_cnn.models import mnist
            >>> import theano
            >>> model, dataset = mnist.testdata_mnist(dropout=.5)
            >>> model.init_arch()
            >>> batch_size = 16
            >>> model.learn_state.init()
            >>> Xb, yb, wb = model._testdata_batch(dataset, batch_size)
            >>> loss = model._theano_exprs['loss'] = None
            >>> loss_item = model._theano_loss_exprs['loss_item']
            >>> X_in = theano.In(model._theano_fn_inputs['X_batch'])
            >>> y_in = theano.In(model._theano_fn_inputs['y_batch'])
            >>> w_in = theano.In(model._theano_fn_inputs['w_batch'])
            >>> loss_batch = loss_item.eval({X_in: Xb, y_in: yb})
        """
        if model._theano_forward is None:
            logger.info('[model.build] request_forward')
            model.learn_state.init()
            fn_inputs = model._theano_fn_inputs
            X_batch, X_given = ut.take(fn_inputs, ['X_batch', 'X_given'])
            y_batch, y_given = ut.take(fn_inputs, ['y_batch', 'y_given'])
            w_batch, w_given = ut.take(fn_inputs, ['w_batch', 'w_given'])

            labeled_outputs = model._get_labeled_outputs()
            unlabeled_outputs = model._get_unlabeled_outputs()

            loss_exprs = model._theano_loss_exprs
            loss_determ = loss_exprs['loss_determ']
            # loss_std_determ = loss_exprs['loss_std_determ']
            forward_losses = [loss_determ]

            In = theano.In
            theano_forward = theano.function(
                inputs=[In(X_batch), In(y_batch), In(w_batch)],
                outputs=forward_losses + labeled_outputs + unlabeled_outputs,
                givens={X_given: X_batch, y_given: y_batch, w_given: w_batch},
                updates=None,
                mode=model._theano_mode,
                name=':feedforward',
            )
            model._theano_forward = theano_forward
        return model._theano_forward

    def build_backprop_func(model):
        """
        Computes loss and updates model parameters.
        Returns diagnostic information.
        """
        if model._theano_backprop is None:
            logger.info('[model.build] request_backprop')
            # Must have an initialized learning state
            model.learn_state.init()

            fn_inputs = model._theano_fn_inputs
            X_batch, X_given = ut.take(fn_inputs, ['X_batch', 'X_given'])
            y_batch, y_given = ut.take(fn_inputs, ['y_batch', 'y_given'])
            w_batch, w_given = ut.take(fn_inputs, ['w_batch', 'w_given'])

            labeled_outputs = model._get_labeled_outputs()

            # Build backprop losses
            loss_exprs = model._theano_loss_exprs
            loss = loss_exprs['loss']
            # loss_std = loss_exprs['loss_std']
            loss_reg = loss_exprs['loss_reg']

            backprop_loss_ = loss_reg
            backprop_losses = [loss_reg, loss]

            # Updates network parameters based on the training loss
            parameters = model.get_all_params(trainable=True)

            updates = model._make_updates(parameters, backprop_loss_)
            monitor_outputs = model._make_monitor_outputs(parameters, updates)

            In = theano.In
            theano_backprop = theano.function(
                inputs=[In(X_batch), In(y_batch), In(w_batch)],
                outputs=(backprop_losses + labeled_outputs + monitor_outputs),
                givens={X_given: X_batch, y_given: y_batch, w_given: w_batch},
                updates=updates,
                mode=model._theano_mode,
                name=':backprop',
            )
            model._theano_backprop = theano_backprop
        return model._theano_backprop

    @property
    def _theano_fn_inputs(model):
        if model._theano_exprs['fn_inputs'] is None:
            if isinstance(model, AbstractVectorVectorModel):
                x_type = T.matrix
            else:
                x_type = T.tensor4

            if isinstance(model, AbstractVectorVectorModel):
                y_type = T.ivector
            elif isinstance(model, AbstractVectorModel):
                y_type = T.imatrix
            else:
                y_type = T.ivector

            if isinstance(model, AbstractVectorVectorModel):
                w_type = T.vector
            elif isinstance(model, AbstractVectorModel):
                w_type = T.matrix
            else:
                w_type = T.vector

            logger.info('[model] Using y_type = %r' % (y_type,))

            fn_inputs = {
                # Data
                'X_given': x_type('X_given'),
                'X_batch': x_type('X_batch'),
                # Labels
                'y_given': y_type('y_given'),
                'y_batch': y_type('y_batch'),
                # Importance
                'w_given': w_type('w_given'),
                'w_batch': w_type('w_batch'),
            }
            model._theano_exprs['fn_inputs'] = fn_inputs
        return model._theano_exprs['fn_inputs']

    def _testdata_batch(model, dataset, batch_size=16):
        data, labels = dataset.subset('test')
        model.ensure_data_params(data, labels)
        class_to_weight = model.data_params['class_to_weight']
        class_to_weight.take(labels)
        weights = class_to_weight.take(labels).astype(np.float32)
        data_per_label = model.data_per_label_input
        Xb_, yb_, wb_ = model.slice_batch(
            data, labels, weights, batch_size, data_per_label
        )
        Xb, yb, wb = model.prepare_data(Xb_, yb_, wb_)
        return Xb, yb, wb
        pass

    @property
    def _theano_loss_exprs(model):
        r"""
        Requires that a custom loss function is defined in the inherited class

        Ignore:
            >>> from wbia_cnn.models.abstract_models import *  # NOQA
            >>> from wbia_cnn.models import mnist
            >>> import theano
            >>> model, dataset = mnist.testdata_mnist(dropout=.5)
            >>> model._init_compile_vars({})  # reset state
            >>> model.init_arch()
            >>> data, labels = dataset.subset('test')
            >>> loss = model._theano_loss_exprs['loss']
            >>> loss_item = model._theano_loss_exprs['loss_item']
            >>> X_in = theano.In(model._theano_fn_inputs['X_batch'])
            >>> y_in = theano.In(model._theano_fn_inputs['y_batch'])
            >>> w_in = theano.In(model._theano_fn_inputs['w_batch'])
            >>> # Eval
            >>> input1 = {X_in: Xb, y_in: yb, w_in: wb}
            >>> input2 = {X_in: Xb, y_in: yb}
            >>> _loss = loss.eval(input1)
            >>> _loss_item = loss_item.eval(input2)
        """
        if model._theano_exprs['loss'] is None:
            with warnings.catch_warnings():
                y_batch = model._theano_fn_inputs['y_batch']
                w_batch = model._theano_fn_inputs['w_batch']

                netout_exprs = model._get_network_output()
                netout_learn = netout_exprs['network_output_learn']
                netout_determ = netout_exprs['network_output_determ']

                # In both learn and validate setting get:
                # Loss of each iterm in the batch
                # Standard deviation of the loss in the batch
                # Weighted mean of the loss in the batch

                # Loss of each example/item
                # Record loss standard deviation over batch for diagnostics

                logger.info('Building symbolic loss function')
                loss_item = model.loss_function(netout_learn, y_batch)
                loss_item.name = 'loss_item'
                # loss_std = loss_item.std()
                # loss_std.name = 'loss_std'
                loss = lasagne.objectives.aggregate(
                    loss_item, weights=w_batch, mode='mean'
                )
                loss.name = 'loss'

                logger.info('Building symbolic loss function (determenistic)')
                loss_item_determ = model.loss_function(netout_determ, y_batch)
                loss_item_determ.name = 'loss_item_determ'
                # loss_std_determ = loss_item_determ.std()
                # loss_std_determ.name = 'loss_std_determ'
                loss_determ = lasagne.objectives.aggregate(
                    loss_item_determ, weights=w_batch, mode='mean'
                )
                loss_determ.name = 'loss_determ'

                # Regularize the learning loss function

                # TODO: L2 should one of many regularization options (Lp, L1)
                L2 = lasagne.regularization.regularize_network_params(
                    model.output_layer, lasagne.regularization.l2
                )
                L2.name = 'reg_L2_param_mag'

                weight_decay = model.learn_state.shared['weight_decay']
                reg_L2_decay = weight_decay * L2
                reg_L2_decay.name = 'reg_L2_decay'
                loss_reg = loss + reg_L2_decay
                loss_reg.name = 'loss_reg'

                loss_exprs = {
                    'loss_reg': loss_reg,
                    'loss_determ': loss_determ,
                    'loss': loss,
                    # 'loss_std': loss_std,
                    # 'loss_std_determ': loss_std_determ,
                    'loss_item': loss_item,
                    'loss_item_determ': loss_item,
                }
            model._theano_exprs['loss'] = loss_exprs
        return model._theano_exprs['loss']

    def _make_updates(model, parameters, backprop_loss_):
        grads = theano.grad(backprop_loss_, parameters, add_names=True)

        shared_learning_rate = model.learn_state.shared['learning_rate']
        momentum = model.learn_state.shared['momentum']

        updates = lasagne.updates.nesterov_momentum(
            loss_or_grads=grads,
            params=parameters,
            learning_rate=shared_learning_rate,
            momentum=momentum
            # add_names=True  # TODO; commit to lasagne
        )

        # workaround for pylearn2 bug documented in
        # https://github.com/Lasagne/Lasagne/issues/728
        for param, update in updates.items():
            if param.broadcastable != update.broadcastable:
                updates[param] = T.patternbroadcast(update, param.broadcastable)
        return updates

    def _get_network_output(model):
        """
        gets the activations of the output neurons
        """
        if model._theano_exprs['netout'] is None:
            X_batch = model._theano_fn_inputs['X_batch']

            network_output_learn = lasagne.layers.get_output(model.output_layer, X_batch)
            network_output_learn.name = 'network_output_learn'

            network_output_determ = lasagne.layers.get_output(
                model.output_layer, X_batch, deterministic=True
            )
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
            y_batch = model._theano_fn_inputs['y_batch']
            model._theano_exprs['labeled_out'] = model.custom_labeled_outputs(
                network_output_determ, y_batch
            )
        return model._theano_exprs['labeled_out']

    def _make_monitor_outputs(model, parameters, updates):
        """
        Builds parameters to monitor the magnitude of updates durning learning
        """
        # Build outputs to babysit training
        monitor_outputs = []
        if model.monitor_config['monitor_updates']:  # and False:
            for param in parameters:
                # The vector each param was udpated with
                # (one vector per channel)
                param_update_vec = updates[param] - param
                param_update_vec.name = 'param_update_vector_' + param.name
                flat_shape = (
                    param_update_vec.shape[0],
                    T.prod(param_update_vec.shape[1:]),
                )
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
class _ModelVisualization(object):
    """

    CommandLine:
        python -m wbia_cnn.models.abstract_models _ModelVisualization
        python -m wbia_cnn.models.abstract_models _ModelVisualization --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia_cnn.models.abstract_models import *  # NOQA
        >>> from wbia_cnn.models import dummy
        >>> model = dummy.DummyModel(batch_size=16, autoinit=False)
        >>> #model._theano_mode = theano.compile.Mode(linker='py', optimizer='fast_compile')
        >>> #model._theano_mode = theano.compile.Mode(linker='py', optimizer='fast_compile')
        >>> model_theano_mode = theano.compile.FAST_COMPILE
        >>> model.init_arch()
        >>> X, y = model.make_random_testdata(num=27, cv2_format=True, asint=False)
        >>> model.fit(X, y, max_epochs=10, era_size=3, buffered=False)
        >>> fnum = None
        >>> from wbia import plottool as pt
        >>> pt.qt4ensure()
        >>> fnum = 1
        >>> model.show_loss_history(fnum)
        >>> #model.show_era_report(fnum)
        >>> ut.show_if_requested()
    """

    def show_arch(model, fnum=None, fullinfo=True, **kwargs):
        from wbia import plottool as pt

        layers = model.get_all_layers(**kwargs)
        draw_net.show_arch_nx_graph(layers, fnum=fnum, fullinfo=fullinfo)
        pt.set_figtitle(model.arch_id)

    def show_class_dream(model, fnum=None, **kwargs):
        """
        CommandLine:
            python -m wbia_cnn.models.abstract_models show_class_dream --show

        Example:
            >>> # DISABLE_DOCTEST
            >>> # Assumes mnist is trained
            >>> from wbia_cnn.draw_net import *  # NOQA
            >>> from wbia_cnn.models import mnist
            >>> model, dataset = mnist.testdata_mnist(dropout=.5)
            >>> model.init_arch()
            >>> model.load_model_state()
            >>> ut.quit_if_noshow()
            >>> from wbia import plottool as pt
            >>> #pt.qt4ensure()
            >>> model.show_class_dream()
            >>> ut.show_if_requested()
        """
        from wbia import plottool as pt

        kw = dict(init='randn', niters=200, update_rate=0.05, weight_decay=1e-4)
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

        import theano
        from theano import tensor as T  # NOQA
        import utool as ut

        num = 64
        Xb = model.prepare_data(X_test[0:num])
        yb = y_test[0:num]
        shared_images = theano.shared(Xb.astype(np.float32))
        dream = model._dream
        step_fn = dream._make_objective(shared_images, yb)
        out = dream._postprocess_class_image(shared_images, yb, was_scalar=True)
        import vtool as vt

        count = 0
        for _ in range(100):
            out = dream._postprocess_class_image(shared_images, yb, was_scalar=True)
            vt.imwrite(join(dpath, 'out%d.jpg' % (count,)), out)
            count += 1
            for _ in ut.ProgIter(range(10)):
                step_fn()

    def show_weights_image(model, index=0, *args, **kwargs):
        from wbia import plottool as pt

        network_layers = model.get_all_layers()
        cnn_layers = [layer_ for layer_ in network_layers if hasattr(layer_, 'W')]
        layer = cnn_layers[index]
        all_weights = layer.W.get_value()
        layername = net_strs.make_layer_str(layer)
        fig = draw_net.show_convolutional_weights(all_weights, **kwargs)
        figtitle = layername + '\n' + model.arch_id + '\n' + model.history.hist_id
        pt.set_figtitle(
            figtitle,
            subtitle='shape=%r, sum=%.4f, l2=%.4f'
            % (all_weights.shape, all_weights.sum(), (all_weights ** 2).sum()),
        )
        return fig

    def show_update_mag_history(model, fnum=None):
        """
        CommandLine:
            python -m wbia_cnn --tf _ModelVisualization.show_update_mag_history --show

        Example:
            >>> # DISABLE_DOCTEST
            >>> from wbia_cnn.models.abstract_models import *  # NOQA
            >>> model = testdata_model_with_history()
            >>> fnum = None
            >>> model.show_update_mag_history(fnum)
            >>> ut.show_if_requested()
        """
        from wbia import plottool as pt

        fnum = pt.ensure_fnum(fnum)

        layer_info = model.get_all_layer_info()
        param_groups = [
            [p['name'] for p in info['param_infos'] if 'trainable' in p['tags']]
            for info in layer_info
        ]
        param_groups = ut.compress(param_groups, param_groups)

        fig = pt.figure(fnum=fnum, pnum=(1, 1, 1), doclf=True, docla=True)
        # next_pnum = pt.make_pnum_nextgen(nSubplots=1 + len(param_groups))
        next_pnum = pt.make_pnum_nextgen(nSubplots=1)
        # if len(model.era_history) > 0:
        #    for param_keys in param_groups:
        #        model.show_weight_updates(param_keys=param_keys, fnum=fnum, pnum=next_pnum())
        # Show learning rate
        labels = None
        if len(model.history) > 0:
            labels = {'rate': 'learning rate'}
        ydatas = []
        for epochsT in model.history.grouped_epochsT():
            learn_rate_list = ut.take_column(epochsT['learn_state'], 'learning_rate')
            ydatas.append({'rate': learn_rate_list})
        model._show_era_measure(
            ydatas,
            labels=labels,
            ylabel='learning rate',
            yscale='linear',
            fnum=fnum,
            pnum=next_pnum(),
        )
        # model.show_weight_updates(fnum=fnum, pnum=next_pnum())
        # TODO: add confusion
        pt.set_figtitle(
            'Weight Update History: ' + model.history.hist_id + '\n' + model.arch_id
        )
        return fig

    def show_pr_history(model, fnum=None):
        """
        CommandLine:
            python -m wbia_cnn --tf _ModelVisualization.show_pr_history --show

        Example:
            >>> # DISABLE_DOCTEST
            >>> from wbia_cnn.models.abstract_models import *  # NOQA
            >>> model = testdata_model_with_history()
            >>> fnum = None
            >>> model.show_pr_history(fnum)
            >>> ut.show_if_requested()
        """
        from wbia import plottool as pt

        fnum = pt.ensure_fnum(fnum)
        fig = pt.figure(fnum=fnum, pnum=(1, 1, 1), doclf=True, docla=True)
        next_pnum = pt.make_pnum_nextgen(nRows=2, nCols=2)
        if len(model.history) > 0:
            model._show_era_class_pr(
                ['learn'], ['precision'], fnum=fnum, pnum=next_pnum()
            )
            model._show_era_class_pr(['learn'], ['recall'], fnum=fnum, pnum=next_pnum())
            model._show_era_class_pr(
                ['valid'], ['precision'], fnum=fnum, pnum=next_pnum()
            )
            model._show_era_class_pr(['valid'], ['recall'], fnum=fnum, pnum=next_pnum())
        pt.set_figtitle('Era PR History: ' + model.history.hist_id + '\n' + model.arch_id)
        return fig

    def show_loss_history(model, fnum=None):
        r"""
        Args:
            fnum (int):  figure number(default = None)

        Returns:
            mpl.Figure: fig

        CommandLine:
            python -m wbia_cnn _ModelVisualization.show_loss_history --show

        Example:
            >>> # DISABLE_DOCTEST
            >>> from wbia_cnn.models.abstract_models import *  # NOQA
            >>> model = testdata_model_with_history()
            >>> fnum = None
            >>> model.show_loss_history(fnum)
            >>> ut.show_if_requested()
        """
        from wbia import plottool as pt

        fnum = pt.ensure_fnum(fnum)
        fig = pt.figure(fnum=fnum, pnum=(1, 1, 1), doclf=True, docla=True)
        next_pnum = pt.make_pnum_nextgen(nRows=2, nCols=2)
        if len(model.history) > 0:
            model._show_era_loss(fnum=fnum, pnum=next_pnum(), yscale='log')
            model._show_era_loss(fnum=fnum, pnum=next_pnum(), yscale='linear')
            model._show_era_acc(fnum=fnum, pnum=next_pnum(), yscale='linear')
            model._show_era_lossratio(fnum=fnum, pnum=next_pnum())
        pt.set_figtitle(
            'Era Loss History: ' + model.history.hist_id + '\n' + model.arch_id
        )
        return fig

    def _show_era_lossratio(model, **kwargs):
        labels = None
        if len(model.history) > 0:
            labels = {'ratio': 'learn/valid'}

        ydatas = [
            {
                'ratio': np.divide(
                    ut.take_column(epochs, 'learn_loss'),
                    ut.take_column(epochs, 'valid_loss'),
                )
            }
            for epochs in model.history.grouped_epochs()
        ]
        return model._show_era_measure(
            ydatas, labels, ylabel='learn/valid', yscale='linear', **kwargs
        )

    def _show_era_class_pr(
        model, types=['valid', 'learn'], measures=['precision', 'recall'], **kwargs
    ):
        from wbia import plottool as pt

        if getattr(model, 'encoder', None):
            # TODO: keep in data_params?
            class_idxs = model.encoder.transform(model.encoder.classes_)
            class_lbls = model.encoder.classes_
        else:
            class_idxs = list(range(model.output_dims))
            class_lbls = list(range(model.output_dims))

        type_styles = {'learn': '-x', 'valid': '-o'}
        styles = ut.odict(
            ut.flatten(
                [
                    [
                        (
                            (
                                '%s_%s_%d'
                                % (
                                    t,
                                    m,
                                    y,
                                )
                            ),
                            type_styles[t],
                        )
                        for t in types
                        for m in measures
                    ]
                    for y in class_idxs
                ]
            )
        )

        class_colors = pt.distinct_colors(len(class_idxs))
        colors = ut.odict(
            ut.flatten(
                [
                    [
                        (
                            (
                                '%s_%s_%d'
                                % (
                                    t,
                                    m,
                                    y,
                                )
                            ),
                            color,
                        )
                        for t in types
                        for m in measures
                    ]
                    for (y, color) in zip(class_idxs, class_colors)
                ]
            )
        )

        if len(model.history) > 0:
            labels = ut.odict(
                ut.flatten(
                    [
                        [
                            (
                                '%s_%s_%d'
                                % (
                                    t,
                                    m,
                                    y,
                                ),
                                '%s %s %s'
                                % (
                                    t,
                                    m,
                                    category,
                                ),
                            )
                            for t in types
                            for m in measures
                        ]
                        for y, category in zip(class_idxs, class_lbls)
                    ]
                )
            )

        ydatas = [
            ut.odict(
                ut.flatten(
                    [
                        [
                            (
                                '%s_%s_%d'
                                % (
                                    t,
                                    m,
                                    y,
                                ),
                                np.array(ut.take_column(epochs, '%s_%s' % (t, m)))[:, y],
                            )
                            for t in types
                            for m in measures
                        ]
                        for y in class_idxs
                    ]
                )
            )
            for epochs in model.history.grouped_epochs()
        ]

        fig = model._show_era_measure(
            ydatas,
            styles=styles,
            labels=labels,
            colors=colors,
            ylabel='/'.join(measures),
            yscale='linear',
            **kwargs
        )
        return fig

    def _show_era_acc(model, **kwargs):
        # styles = {'valid': '-o'}
        styles = {'learn': '-x', 'valid': '-o'}
        labels = None
        if len(model.history) > 0:
            last_epoch = model.history.epoch_list[-1]
            last_era = model.history.era_list[-1]
            labels = {
                'valid': 'valid acc '
                + str(last_era['num_valid'])
                + ' '
                + ut.repr4(last_epoch['learn_acc'] * 100)
                + '%',
                'learn': 'learn acc '
                + str(last_era['num_learn'])
                + ' '
                + ut.repr4(last_epoch['valid_acc'] * 100)
                + '%',
            }

        ydatas = [
            {
                'valid': ut.take_column(epochs, 'valid_acc'),
                'learn': ut.take_column(epochs, 'learn_acc'),
            }
            for epochs in model.history.grouped_epochs()
        ]

        yspreads = [
            {
                'valid': ut.take_column(epochs, 'valid_acc_std'),
                'learn': ut.take_column(epochs, 'learn_acc_std'),
            }
            for epochs in model.history.grouped_epochs()
        ]

        fig = model._show_era_measure(
            ydatas, labels, styles, ylabel='accuracy', yspreads=yspreads, **kwargs
        )
        # from wbia import plottool as pt
        # ax = pt.gca()
        # ymin, ymax = ax.get_ylim()
        # pt.gca().set_ylim((ymin, 100))
        return fig

    def _show_era_loss(model, **kwargs):
        styles = {'learn': '-x', 'valid': '-o'}
        labels = None
        if len(model.history) > 0:
            labels = {
                'learn': 'learn ' + str(model.history.era_list[-1]['num_learn']),
                'valid': 'valid ' + str(model.history.era_list[-1]['num_valid']),
            }
        epochsT_list = list(model.history.grouped_epochsT())

        ydatas = [
            {'learn': epochsT['learn_loss'], 'valid': epochsT['valid_loss']}
            for epochsT in epochsT_list
        ]

        yspreads = [
            {'learn': epochsT['learn_loss_std'], 'valid': epochsT['valid_loss_std']}
            for epochsT in epochsT_list
        ]

        fig = model._show_era_measure(
            ydatas, labels, styles, ylabel='loss', yspreads=yspreads, **kwargs
        )
        return fig

    def show_weight_updates(model, param_keys=None, **kwargs):
        # has_mag_updates = False
        from wbia import plottool as pt

        xdatas = []
        ydatas = []
        yspreads = []
        labels = None
        colors = None

        if len(model.history) > 0:
            era = model.history.era_list[-1]
            if 'param_update_mags' in era['epoch_info_list']:
                if param_keys is None:
                    param_keys = list(
                        set(
                            ut.flatten(
                                [dict_.keys() for dict_ in era['param_update_mags_list']]
                            )
                        )
                    )
                labels = dict(zip(param_keys, param_keys))
                colors = dict(zip(param_keys, pt.distinct_colors(len(param_keys))))

        # colors = {}
        for index, epochs in enumerate(model.history):
            if 'param_update_mags' not in era['epoch_info_list']:
                continue
            xdata = ut.take_column(epochs, 'epoch_num')
            if index == 0:
                xdata = xdata[1:]
            xdatas.append(xdata)
            # FIXME
            update_mags_list = era['param_update_mags_list']
            # Transpose keys and positions
            # param_keys = list(set(ut.flatten([
            #    dict_.keys() for dict_ in update_mags_list
            # ])))
            param_val_list = [
                [list_[param] for list_ in update_mags_list] for param in param_keys
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
        return model._show_era_measure(
            ydatas,
            labels=labels,
            colors=colors,
            xdatas=xdatas,
            ylabel='update magnitude',
            yspreads=yspreads,
            yscale='linear',
            **kwargs
        )

    def _show_era_measure(
        model,
        ydatas,
        labels=None,
        styles=None,
        xdatas=None,
        xlabel='epoch',
        ylabel='',
        yspreads=None,
        colors=None,
        fnum=None,
        pnum=(1, 1, 1),
        yscale='log',
    ):

        # logger.info('Show Era Measure ylabel = %r' % (ylabel,))
        from wbia import plottool as pt

        num_eras = model.history.total_eras

        if labels is None:
            pass

        if styles is None:
            styles = ut.ddict(lambda: '-o')

        if xdatas is None:
            xdatas = [epochsT['epoch_num'] for epochsT in model.history.grouped_epochsT()]

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
            color_ = colors[index] if isinstance(colors, list) else colors

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
                        ax.fill_between(xdata_, ymin, ymax, alpha=0.12, color=color)
                    pt.plot(xdata_, ydata_, '--', color=color, yscale=yscale)

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
                    ax.fill_between(xdata, ymin, ymax, alpha=0.2, color=color)
                    prev_yspread[key] = std[-1:]
                pt.plot(xdata, ydata_, styles[key], color=color, yscale=yscale, **kw)
                prev_ydata[key] = ydata_[-1:]
            prev_xdata = xdata[-1:]

        # append_phantom_legend_label
        pt.set_xlabel(xlabel)
        pt.set_ylabel(ylabel)
        if len(model.history) > 0 and can_plot:
            pt.legend()
        pt.dark_background()
        return fig

    def show_regularization_stuff(model, fnum=None, pnum=(1, 1, 1)):
        from wbia import plottool as pt

        fnum = pt.ensure_fnum(fnum)
        fig = pt.figure(fnum=fnum, pnum=pnum)
        for index, era in enumerate(model.history):
            # epochs = era['epoch_list']
            epochs = ut.take_column(era['epoch_info_list'], 'epoch')
            if 'update_mags_list' not in era:
                continue
            update_mags_list = era['update_mags_list']
            # Transpose keys and positions
            param_keys = list(
                set(ut.flatten([dict_.keys() for dict_ in update_mags_list]))
            )
            param_val_list = [
                [list_[param] for list_ in update_mags_list] for param in param_keys
            ]
            colors = pt.distinct_colors(len(param_val_list))
            for key, val, color in zip(param_keys, param_val_list, colors):
                update_mag_mean = ut.get_list_column(val, 0)
                update_mag_std = ut.get_list_column(val, 1)
                if index == len(model.history) - 1:
                    pt.interval_line_plot(
                        epochs,
                        update_mag_mean,
                        update_mag_std,
                        marker='x',
                        linestyle='-',
                        color=color,
                        label=key,
                    )
                else:
                    pt.interval_line_plot(
                        epochs,
                        update_mag_mean,
                        update_mag_std,
                        marker='x',
                        linestyle='-',
                        color=color,
                    )
            pass
        if len(model.history) > 0:
            pt.legend()

        pt.dark_background()
        return fig

    # --- IMAGE WRITE

    def render_arch(model, fullinfo=True):
        from wbia import plottool as pt

        savekw = dict(dpi=180)
        with pt.RenderingContext(**savekw) as render:
            model.show_arch(fnum=render.fig.number, fullinfo=fullinfo)
        return render.image

    def imwrite_arch(model, fpath=None, fullinfo=True):
        r"""
        Args:
            fpath (str):  file path string(default = None)

        CommandLine:
            python -m wbia_cnn.models.abstract_models imwrite_arch --show

        Example:
            >>> # DISABLE_DOCTEST
            >>> from wbia_cnn.models.abstract_models import *  # NOQA
            >>> from wbia_cnn.models.mnist import MNISTModel
            >>> model = MNISTModel(batch_size=128, data_shape=(28, 28, 1),
            >>>                    output_dims=10, batch_norm=False, name='mnist')
            >>> model.init_arch()
            >>> fapth = model.imwrite_arch()
            >>> ut.quit_if_noshow()
            >>> ut.startfile(fapth)
        """
        # FIXME
        import vtool as vt

        if fpath is None:
            fpath = './' + model.arch_id + '.jpg'
        image = model.render_arch(fullinfo=fullinfo)
        vt.imwrite(fpath, image)
        return fpath


@ut.reloadable_class
class _ModelStrings(object):
    """"""

    def get_state_str(model, other_override_reprs={}):
        era_history_str = ut.list_str(
            [ut.dict_str(era, truncate=True, sorted_=True) for era in model.history],
            strvals=True,
        )

        override_reprs = {
            'best_results': ut.repr3(model.best_results),
            # 'best_weights': ut.truncate_str(str(model.best_weights)),
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
        CommandLine:
            python -m wbia_cnn.models.abstract_models make_arch_json --show

        Example:
            >>> # ENABLE_DOCTEST
            >>> from wbia_cnn.models.abstract_models import *  # NOQA
            >>> from wbia_cnn.models import mnist
            >>> model, dataset = mnist.testdata_mnist(defaultname='resnet')
            >>> #model = mnist.MNISTModel(batch_size=128, data_shape=(28, 28, 1),
            >>> #                         output_dims=10, batch_norm=True)
            >>> model.init_arch()
            >>> json_str = model.make_arch_json(with_noise=True)
            >>> print(json_str)
        """
        layer_list = model.get_all_layers(with_noise=with_noise)
        layer_json_list = net_strs.make_layers_json(layer_list, extra=with_noise)
        json_dict = ut.odict()
        json_dict['arch_hashid'] = model.get_arch_hashid()
        json_dict['layers'] = layer_json_list

        if False:
            raw_json = ut.to_json(json_dict, pretty=True)
            lines = raw_json.split('\n')
            levels = np.array([ut.get_indentation(line) for line in lines]) / 4
            # Collapse newlines past indent 4
            nl = 4
            newlines = []
            prev = False
            for line, level in zip(lines, levels):
                if level >= nl:
                    if prev:
                        line = line.lstrip(' ')
                        newlines[-1] += '' + line
                    else:
                        newlines.append(line)
                        prev = True
                else:
                    newlines.append(line)
                    prev = False
            json_str = '\n'.join(newlines)
        else:
            # prettier json
            json_str = ut.repr2_json(json_dict, nl=3)
        return json_str

    def get_arch_str(model, sep='_', with_noise=False):
        r"""
        with_noise is a boolean that specifies if layers that doesnt
        affect the flow of information in the determenistic setting are to be
        included. IE get rid of dropout.

        CommandLine:
            python -m wbia_cnn.models.abstract_models _ModelStrings.get_arch_str:0

        Example:
            >>> # ENABLE_DOCTEST
            >>> from wbia_cnn.models.abstract_models import *  # NOQA
            >>> from wbia_cnn.models.mnist import MNISTModel
            >>> model = MNISTModel(batch_size=128, data_shape=(28, 28, 1),
            >>>                    output_dims=10, batch_norm=True)
            >>> model.init_arch()
            >>> result = model.get_arch_str(sep=ut.NEWLINE, with_noise=False)
            >>> print(result)
            InputLayer(shape=(None, 1, 28, 28))
            Conv2DLayer(num_filters=128,filter_size=(3, 3),stride=(1, 1),output_shape=(None, 128, 26, 26),num_groups=1,nonlinearity=linear)
            BatchNormLayer2(alpha=0.1,nonlinearity=rectify)
            MaxPool2DLayer(stride=(2, 2),pool_size=(2, 2),output_shape=(None, 128, 13, 13))
            Conv2DLayer(num_filters=128,filter_size=(3, 3),stride=(1, 1),output_shape=(None, 128, 11, 11),num_groups=1,nonlinearity=linear)
            BatchNormLayer2(alpha=0.1,nonlinearity=rectify)
            Conv2DLayer(num_filters=128,filter_size=(3, 3),stride=(1, 1),output_shape=(None, 128, 9, 9),num_groups=1,nonlinearity=linear)
            BatchNormLayer2(alpha=0.1,nonlinearity=rectify)
            MaxPool2DLayer(stride=(2, 2),pool_size=(2, 2),output_shape=(None, 128, 4, 4))
            Conv2DLayer(num_filters=128,filter_size=(3, 3),stride=(1, 1),output_shape=(None, 128, 2, 2),num_groups=1,nonlinearity=linear)
            BatchNormLayer2(alpha=0.1,nonlinearity=rectify)
            Conv2DLayer(num_filters=128,filter_size=(2, 2),stride=(1, 1),output_shape=(None, 128, 1, 1),num_groups=1,nonlinearity=linear)
            BatchNormLayer2(alpha=0.1,nonlinearity=rectify)
            DenseLayer(num_units=256,num_leading_axes=1,nonlinearity=linear)
            BatchNormLayer2(alpha=0.1,nonlinearity=rectify)
            DenseLayer(num_units=256,num_leading_axes=1,nonlinearity=linear)
            BatchNormLayer2(alpha=0.1,nonlinearity=rectify)
            DenseLayer(num_units=10,num_leading_axes=1,nonlinearity=softmax)
        """
        if model.output_layer is None:
            return ''
        layer_list = model.get_all_layers(with_noise=with_noise)
        layer_str_list = [
            net_strs.make_layer_str(layer, with_name=with_noise) for layer in layer_list
        ]
        architecture_str = sep.join(layer_str_list)
        return architecture_str

    def get_layer_info_str(model):
        """
        CommandLine:
            python -m wbia_cnn.models.abstract_models _ModelStrings.get_layer_info_str:0

        Example:
            >>> # DISABLE_DOCTEST
            >>> from wbia_cnn.models.abstract_models import *  # NOQA
            >>> from wbia_cnn.models.mnist import MNISTModel
            >>> model = MNISTModel(batch_size=128, data_shape=(28, 28, 1),
            >>>                    output_dims=10)
            >>> model.init_arch()
            >>> result = model.get_layer_info_str()
            >>> print(result)
            Network Structure:
            index  Name    Layer       Outputs      Bytes OutShape             Params
            0      I1      Input           784      3,136 (None, 1, 28, 28)    []
            1      C2      Conv2D       86,528    355,328 (None, 128, 26, 26)  [W(128,1,3,3; r,t)]
            2      C2/bn_  BatchNorm    86,528    350,208 (None, 128, 26, 26)  [beta(128; t), gamma(128; r,t), mean(128; b), inv_std(128; b)]
            3      C2/P    MaxPool2D    21,632     86,528 (None, 128, 13, 13)  []
            4      C3      Conv2D       15,488  1,241,600 (None, 128, 11, 11)  [W(128,128,3,3; r,t)]
            5      C3/bn_  BatchNorm    15,488     66,048 (None, 128, 11, 11)  [beta(128; t), gamma(128; r,t), mean(128; b), inv_std(128; b)]
            6      C4      Conv2D       10,368  1,221,120 (None, 128, 9, 9)    [W(128,128,3,3; r,t)]
            7      C4/bn_  BatchNorm    10,368     45,568 (None, 128, 9, 9)    [beta(128; t), gamma(128; r,t), mean(128; b), inv_std(128; b)]
            8      C4/P    MaxPool2D     2,048      8,192 (None, 128, 4, 4)    []
            9      C5      Conv2D          512  1,181,696 (None, 128, 2, 2)    [W(128,128,3,3; r,t)]
            10     C5/bn_  BatchNorm       512      6,144 (None, 128, 2, 2)    [beta(128; t), gamma(128; r,t), mean(128; b), inv_std(128; b)]
            11     C6      Conv2D          128    524,800 (None, 128, 1, 1)    [W(128,128,2,2; r,t)]
            12     C6/bn_  BatchNorm       128      4,608 (None, 128, 1, 1)    [beta(128; t), gamma(128; r,t), mean(128; b), inv_std(128; b)]
            13     D7      Dropout         128        512 (None, 128, 1, 1)    []
            14     F7      Dense           256    263,168 (None, 256)          [W(128,256; r,t)]
            15     F7/bn   BatchNorm       256      9,216 (None, 256)          [beta(256; t), gamma(256; r,t), mean(256; b), inv_std(256; b)]
            16     D8      Dropout         256      1,024 (None, 256)          []
            17     F8      Dense           256    525,312 (None, 256)          [W(256,256; r,t)]
            18     F8/bn   BatchNorm       256      9,216 (None, 256)          [beta(256; t), gamma(256; r,t), mean(256; b), inv_std(256; b)]
            19     D9      Dropout         256      1,024 (None, 256)          []
            20     F9      SoftMax          10     20,600 (None, 10)           [W(256,10; r,t), b(10; t)]
            ...this model has 614,538 learnable parameters
            ...this model will use ~5,925,048 bytes = 5.65 MB per input
            ...this model will use ~758,406,144 bytes = 723.27 MB per batch with a batch size of 128
        """
        return net_strs.get_layer_info_str(model.get_all_layers(), model.batch_size)

    # --- PRINTING

    def print_state_str(model, **kwargs):
        logger.info(model.get_state_str(**kwargs))

    def print_layer_info(model):
        net_strs.print_layer_info(model.get_all_layers())

    def print_arch_str(model, sep='\n  '):
        architecture_str = model.get_arch_str(sep=sep)
        if architecture_str is None or architecture_str == '':
            architecture_str = 'UNDEFINED'
        logger.info('\nArchitecture:' + sep + architecture_str)

    def print_model_info_str(model):
        logger.info('\n---- Arch Str')
        model.print_arch_str(sep='\n')
        logger.info('\n---- Layer Info')
        model.print_layer_info()
        logger.info('\n---- Arch HashID')
        logger.info(
            'arch_hashid=%r' % (model.get_arch_hashid()),
        )
        logger.info('----')


@ut.reloadable_class
class _ModelIDs(object):
    def _init_id_vars(model, kwargs):
        model.name = kwargs.pop('name', None)
        # if model.name is None:
        #    model.name = ut.get_classname(model.__class__, local=True)

    def __nice__(model):
        parts = [model.get_arch_nice(), model.history.get_history_nice()]
        if model.name is not None:
            parts = [model.name] + parts
        return ' '.join(parts)

    @property
    def hash_id(model):
        arch_hashid = model.get_arch_hashid()
        history_hashid = model.history.get_history_hashid()
        hashid = ut.hashstr27(history_hashid + arch_hashid)
        return hashid

    @property
    def arch_id(model):
        """
        CommandLine:
            python -m wbia_cnn.models.abstract_models _ModelIDs.arch_id:0

        Example:
            >>> # ENABLE_DOCTEST
            >>> from wbia_cnn.models.abstract_models import *  # NOQA
            >>> from wbia_cnn.models.mnist import MNISTModel
            >>> model = MNISTModel(batch_size=128, data_shape=(28, 28, 1),
            >>>                    output_dims=10, name='bnorm', batch_norm=True)
            >>> model.init_arch()
            >>> result = str(model.arch_id)
            >>> print(result)
        """
        if model.name is not None:
            parts = ['arch', model.name, model.get_arch_nice(), model.get_arch_hashid()]
        else:
            parts = ['arch', model.get_arch_nice(), model.get_arch_hashid()]
        arch_id = '_'.join(parts)
        return arch_id

    def get_arch_hashid(model):
        """
        Returns a hash identifying the architecture of the determenistic net.
        This does not involve any dropout or noise layers, nor does the
        initialization of the weights matter.
        """
        arch_str = model.get_arch_str(with_noise=False)
        arch_hashid = ut.hashstr27(arch_str, hashlen=8)
        return arch_hashid

    def get_arch_nice(model):
        """
        Makes a string that shows the number of input units, output units,
        hidden units, parameters, and model depth.

        CommandLine:
            python -m wbia_cnn.models.abstract_models get_arch_nice --show

        Example:
            >>> # ENABLE_DOCTEST
            >>> from wbia_cnn.models.abstract_models import *  # NOQA
            >>> from wbia_cnn.models.mnist import MNISTModel
            >>> model = MNISTModel(batch_size=128, data_shape=(28, 28, 1),
            >>>                    output_dims=10, batch_norm=True)
            >>> model.init_arch()
            >>> result = str(model.get_arch_nice())
            >>> print(result)
            o10_d8_c650
        """
        if model.output_layer is None:
            return 'NoArch'
        else:
            weighted_layers = model.get_all_layers(
                with_noise=False, with_weightless=False
            )
            info_list = [net_strs.get_layer_info(layer) for layer in weighted_layers]
            # The number of units depends if you look at things via input or output
            # does a convolutional layer have its outputs or the outputs of the pooling layer?
            # num_units1 = sum([info['num_outputs'] for info in info_list])
            nhidden = sum([info['num_inputs'] for info in info_list[1:]])
            # num_units2 += net_strs.get_layer_info(model.output_layer)['num_outputs']
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
                # logmcomplex=int(np.round(np.log10(nhidden + nparam)))
                mcomplex=int(np.round((nhidden + nparam) / 1000)),
            )
            # nice = 'i{nin}_o{nout}_d{depth}_h{nhidden}_p{nparam}'.format(**fmtdict)
            # Use a complexity measure
            # nice = 'i{nin}_o{nout}_d{depth}_c{mcomplex}'.format(**fmtdict)
            nice = 'o{nout}_d{depth}_c{mcomplex}'.format(**fmtdict)
            return nice


@ut.reloadable_class
class _ModelIO(object):
    def _init_io_vars(model, kwargs):
        model.dataset_dpath = kwargs.pop('dataset_dpath', '.')
        model.training_dpath = kwargs.pop('training_dpath', '.')
        model._arch_dpath = kwargs.pop('arch_dpath', None)

    def print_structure(model):
        """

        CommandLine:
            python -m wbia_cnn.models.abstract_models print_structure --show

        Example:
            >>> from wbia_cnn.ingest_data import *  # NOQA
            >>> dataset = grab_mnist_category_dataset()
            >>> dataset.print_dir_structure()
            >>> # ----
            >>> from wbia_cnn.models.mnist import MNISTModel
            >>> model = MNISTModel(batch_size=128, data_shape=(28, 28, 1),
            >>>                    output_dims=10, dataset_dpath=dataset.dataset_dpath,
            >>>                    batch_norm=True)
            >>> model.print_structure()

        """
        # logger.info(model.model_dpath)
        logger.info(model.arch_dpath)
        # logger.info(model.best_dpath)
        logger.info(model.saved_session_dpath)
        logger.info(model.checkpoint_dpath)
        # logger.info(model.diagnostic_dpath)
        logger.info(model.trained_model_dpath)
        logger.info(model.trained_arch_dpath)

    @property
    def trained_model_dpath(model):
        return join(model.training_dpath, 'trained_models')

    @property
    def trained_arch_dpath(model):
        return join(model.trained_model_dpath, model.arch_id)

    # @property
    # def model_dpath(model):
    #    return join(model.dataset_dpath, 'models')

    @property
    def arch_dpath(model):
        # return join(model.model_dpath, model.arch_id)
        if model._arch_dpath is None:
            return join(model.dataset_dpath, 'models', model.arch_id)
        else:
            return model._arch_dpath

    @arch_dpath.setter
    def arch_dpath(model, arch_dpath):
        model._arch_dpath = arch_dpath

    # @property
    # def best_dpath(model):
    #    return join(model.arch_dpath, 'best')

    @property
    def checkpoint_dpath(model):
        return join(model.arch_dpath, 'checkpoints')

    @property
    def saved_session_dpath(model):
        return join(model.arch_dpath, 'saved_sessions')

    # @property
    # def diagnostic_dpath(model):
    #    return join(model.arch_dpath, 'diagnostics')

    # def get_epoch_diagnostic_dpath(model, epoch=None):
    #    import utool as ut
    #    diagnostic_dpath = model.diagnostic_dpath
    #    ut.ensuredir(diagnostic_dpath)
    #    epoch_dpath = ut.unixjoin(diagnostic_dpath, model.history.hist_id)
    #    ut.ensuredir(epoch_dpath)
    #    return epoch_dpath

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

    def _get_model_file_fpath(model, default_fname, fpath, dpath, fname, checkpoint_tag):
        if fpath is None:
            fname = default_fname if fname is None else fname
            dpath = model._get_model_dpath(dpath, checkpoint_tag)
            fpath = join(dpath, fname)
        else:
            assert checkpoint_tag is None, 'fpath overrides all other settings'
            assert dpath is None, 'fpath overrides all other settings'
            assert fname is None, 'fpath overrides all other settings'
        return fpath

    def resolve_fuzzy_checkpoint_pattern(model, checkpoint_pattern, extern_dpath=None):
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
                    'Could not resolve checkpoint_pattern=%r. No Matches'
                    % (checkpoint_pattern,)
                )
            elif len(matching_dpaths) > 1:
                raise RuntimeError(
                    (
                        'Could not resolve checkpoint_pattern=%r. '
                        'matching_dpaths=%r. Too many matches'
                    )
                    % (checkpoint_pattern, matching_dpaths)
                )
            else:
                checkpoint_tag = basename(matching_dpaths[0])
                logger.info(
                    'Resolved checkpoint pattern to checkpoint_tag=%r' % (checkpoint_tag,)
                )
        return checkpoint_tag

    def has_saved_state(model, checkpoint_tag=None):
        """
        Check if there are any saved model states matching the checkpoing tag.
        """
        fpath = model.get_model_state_fpath(checkpoint_tag=checkpoint_tag)
        if checkpoint_tag is not None:
            ut.assertpath(fpath)
        return ut.checkpath(fpath)

    def get_model_state_fpath(
        model, fpath=None, dpath=None, fname=None, checkpoint_tag=None
    ):
        default_fname = 'model_state_arch_%s.pkl' % (model.get_arch_hashid())
        model_state_fpath = model._get_model_file_fpath(
            default_fname, fpath, dpath, fname, checkpoint_tag
        )
        return model_state_fpath

    def get_model_info_fpath(
        model, fpath=None, dpath=None, fname=None, checkpoint_tag=None
    ):
        default_fname = 'model_info_arch_%s.pkl' % (model.get_arch_hashid())
        model_state_fpath = model._get_model_file_fpath(
            default_fname, fpath, dpath, fname, checkpoint_tag
        )
        return model_state_fpath

    def checkpoint_save_model_state(model):
        fpath = model.get_model_state_fpath(checkpoint_tag=model.history.hist_id)
        ut.ensuredir(dirname(fpath))
        model.save_model_state(fpath=fpath)
        return fpath

    def checkpoint_save_model_info(model):
        fpath = model.get_model_info_fpath(checkpoint_tag=model.history.hist_id)
        ut.ensuredir(dirname(fpath))
        model.save_model_info(fpath=fpath)

    def save_model_state(model, **kwargs):
        """saves current model state"""
        current_weights = model.get_all_param_values()
        model_state = {
            'best_results': model.best_results,
            'data_params': model.data_params,
            'current_weights': current_weights,
            'encoder': getattr(model, 'encoder', None),
            'input_shape': model.input_shape,
            'data_shape': model.data_shape,
            'batch_size': model.data_shape,
            'output_dims': model.output_dims,
            'era_history': model.history,
            'arch_tag': model.arch_tag,
        }
        model_state_fpath = model.get_model_state_fpath(**kwargs)
        # logger.info('saving model state to: %s' % (model_state_fpath,))
        ut.save_cPkl(model_state_fpath, model_state, verbose=False)
        logger.info('saved model state to %r' % (model_state_fpath,))
        # logger.info('finished saving')
        return model_state_fpath

    def save_model_info(model, **kwargs):
        """save model information (history and results but no weights)"""
        model_info = {
            'best_results': model.best_results,
            'input_shape': model.input_shape,
            'output_dims': model.output_dims,
            'era_history': model.history,
        }
        model_info_fpath = model.get_model_state_fpath(**kwargs)
        # logger.info('saving model info to: %s' % (model_info_fpath,))
        ut.save_cPkl(model_info_fpath, model_info, verbose=False)
        logger.info('saved model info')

    def load_model_state(model, **kwargs):
        """
        kwargs = {}
        TODO: resolve load_model_state and load_extern_weights into a single
            function that is less magic in what it does and more
            straightforward

        Example:
            >>> # DISABLE_DOCTEST
            >>> # Assumes mnist is trained
            >>> from wbia_cnn.models.abstract_models import  *  # NOQA
            >>> from wbia_cnn.models import mnist
            >>> model, dataset = mnist.testdata_mnist()
            >>> model.init_arch()
            >>> model.load_model_state()
        """
        model_state_fpath = model.get_model_state_fpath(**kwargs)
        logger.info('[model] loading model state from: %s' % (model_state_fpath,))
        model_state = ut.load_cPkl(model_state_fpath)
        if model.__class__.__name__ != 'BaseModel':
            assert (
                model_state['input_shape'][1:] == model.input_shape[1:]
            ), 'architecture disagreement'
            assert (
                model_state['output_dims'] == model.output_dims
            ), 'architecture disagreement'
            if 'preproc_kw' in model_state:
                model.data_params = model_state['preproc_kw']
                model._fix_center_mean_std()
            else:
                model.data_params = model_state['data_params']
        else:
            # HACK TO LOAD ABSTRACT MODEL FOR DIAGNOSITIC REASONS
            logger.info('WARNING LOADING ABSTRACT MODEL')
        model.best_results = model_state['best_results']
        model.input_shape = model_state['input_shape']
        model.output_dims = model_state['output_dims']
        model.encoder = model_state.get('encoder', None)
        if 'era_history' in model_state:
            try:
                model.history = History.from_oldstyle(model_state['era_history'])
            except TypeError:
                model.history = model_state['era_history']
        else:
            model.history = History()
            model.history.__dict__.update(**model_state['history'])
        if model.__class__.__name__ != 'BaseModel':
            # hack for abstract model
            # model.output_layer is not None
            model.set_all_param_values(model.best_results['weights'])

    def load_extern_weights(model, **kwargs):
        """load weights from another model"""
        model_state_fpath = model.get_model_state_fpath(**kwargs)
        logger.info('[model] loading extern weights from: %s' % (model_state_fpath,))
        model_state = ut.load_cPkl(model_state_fpath)
        if VERBOSE_CNN:
            logger.info('External Model State:')
            logger.info(ut.dict_str(model_state, truncate=True))
        # check compatibility with this architecture
        assert (
            model_state['input_shape'][1:] == model.input_shape[1:]
        ), 'architecture disagreement'
        assert (
            model_state['output_dims'] == model.output_dims
        ), 'architecture disagreement'
        # Just set the weights, no other training state variables
        model.set_all_param_values(model_state['best_weights'])
        # also need to make sure the same preprocessing is used
        # TODO make this a layer?
        if 'preproc_kw' in model_state:
            model.data_params = model_state['preproc_kw']
            model._fix_center_mean_std()
        else:
            model.data_params = model_state['data_params']

        if 'era_history' in model_state:
            model.history = History.from_oldstyle(model_state['era_history'])
        else:
            model.history = History()
            model.history.__dict__.update(**model_state['history'])


class _ModelUtility(object):
    def set_all_param_values(model, weights_list):
        import lasagne

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', '.*topo.*')
            try:
                lasagne.layers.set_all_param_values(model.get_all_layers(), weights_list)
            except TypeError:
                weights_list_fp32 = [
                    weights.astype(np.float32) for weights in weights_list
                ]
                lasagne.layers.set_all_param_values(
                    model.get_all_layers(), weights_list_fp32
                )

    def get_all_param_values(model):
        import lasagne

        weights_list = lasagne.layers.get_all_param_values(model.output_layer)
        return weights_list

    def get_all_params(model, **tags):
        import lasagne

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', '.*topo.*')
            parameters = lasagne.layers.get_all_params(model.output_layer, **tags)
            return parameters

    def get_all_layer_info(model):
        return [net_strs.get_layer_info(layer) for layer in model.get_all_layers()]

    def get_all_layers(model, with_noise=True, with_weightless=True):
        import lasagne

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', '.*topo.*')
            warnings.filterwarnings('ignore', '.*layer.get_all_layers.*')
            assert model.output_layer is not None, 'need to initialize'
            layer_list_ = lasagne.layers.get_all_layers(model.output_layer)
        layer_list = layer_list_
        if not with_noise:
            # Remove dropout / gaussian noise layers
            layer_list = [
                layer
                for layer in layer_list_
                if layer.__class__.__name__ not in lasagne.layers.noise.__all__
            ]
        if not with_weightless:
            # Remove layers without weights
            layer_list = [layer for layer in layer_list if hasattr(layer, 'W')]
        return layer_list

    @property
    def layers_(model):
        """for compatibility with nolearn visualizations"""
        return model.get_all_layers()

    def get_output_layer(model):
        if model.output_layer is not None:
            return model.output_layer
        else:
            return None

    def _validate_data(model, X_train):
        """Check to make sure data agrees with model input"""
        input_layer = model.get_all_layers()[0]
        expected_item_shape = ut.take(input_layer.shape[1:], [1, 2, 0])
        expected_item_shape = tuple(expected_item_shape)
        given_item_shape = X_train.shape[1:]
        if given_item_shape != expected_item_shape:
            raise ValueError(
                'inconsistent item shape: '
                + ('expected_item_shape = %r, ' % (expected_item_shape,))
                + ('given_item_shape = %r' % (given_item_shape,))
            )

    def _validate_labels(model, X, y, w):
        if y is not None:
            assert X.shape[0] == (
                y.shape[0] * model.data_per_label_input
            ), 'bad data / label alignment'
        if w is not None:
            assert X.shape[0] == (
                w.shape[0] * model.data_per_label_input
            ), 'bad data / label alignment'

    def _validate_input(model, X, y=None, w=None):
        model._validate_data(X)
        model._validate_labels(X, y, w)

    def make_random_testdata(model, num=37, rng=0, cv2_format=False, asint=False):
        logger.info('made random testdata')
        rng = ut.ensure_rng(rng)
        num_labels = num
        num_data = num * model.data_per_label_input
        X = rng.rand(num_data, *model.data_shape)
        y = rng.rand(num_labels) * (model.output_dims - 1)
        X = (X * 100).astype(np.int) / 100
        if asint:
            X = (X * 255).astype(np.uint8)
        else:
            X = X.astype(np.float32)
        y = np.round(y).astype(np.int32)
        if not cv2_format:
            X = X.transpose((0, 3, 1, 2))
        return X, y


@ut.reloadable_class
class BaseModel(
    _model_legacy._ModelLegacy,
    _ModelVisualization,
    _ModelIO,
    _ModelStrings,
    _ModelIDs,
    _ModelBackend,
    _ModelFitter,
    _ModelPredicter,
    _ModelBatch,
    _ModelUtility,
    ut.NiceRepr,
):
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
        # Should delayed import be moved? (or deleted)
        delayed_import()
        model._init_io_vars(kwargs)
        model._init_id_vars(kwargs)
        model._init_shape_vars(kwargs)
        model._init_compile_vars(kwargs)
        model._init_fit_vars(kwargs)
        model._init_batch_vars(kwargs)
        model.output_layer = None
        autoinit = kwargs.pop('autoinit', False)
        model.arch_tag = kwargs.pop('arch_tag', None)
        assert len(kwargs) == 0, 'Model was given unused keywords=%r' % (
            list(kwargs.keys())
        )
        if autoinit:
            model.init_arch()

    def _init_shape_vars(model, kwargs):
        input_shape = kwargs.pop('input_shape', None)
        batch_size = kwargs.pop('batch_size', None)
        data_shape = kwargs.pop('data_shape', None)
        output_dims = kwargs.pop('output_dims', None)

        if input_shape is None and data_shape is None:
            report_error('Must specify either input_shape or data_shape')
        elif input_shape is None:
            CONST_BATCH = False
            if CONST_BATCH:
                # Fixed batch size
                input_shape = (batch_size, data_shape[2], data_shape[0], data_shape[1])
            else:
                # Dynamic batch size
                input_shape = (None, data_shape[2], data_shape[0], data_shape[1])
        elif data_shape is None and batch_size is None:
            data_shape = (input_shape[2], input_shape[3], input_shape[1])
            batch_size = input_shape[0]
        else:
            report_error('Dont specify batch_size or data_shape with input_shape')

        model.output_dims = output_dims
        model.input_shape = input_shape
        model.data_shape = data_shape
        model.batch_size = batch_size
        # bad name, says that this network will take
        # 2*N images in a batch and N labels that map to
        # two images a piece
        model.data_per_label_input = 1  # state of network input
        model.data_per_label_output = 1  # state of network output

    # @classmethod
    # def from_saved_state(cls, fpath):
    #    """
    #    fpath = ut.truepath('~/Desktop/manually_saved/arch_injur-shark-resnet_o2_d27_c2942_jzuddodd/model_state_arch_jzuddodd.pkl')
    #    """
    #    arch_dpath = dirname(fpath)
    #    pass

    # @classmethod
    def init_from_json(model, fpath):
        """
        fpath = ut.truepath('~/Desktop/manually_saved/arch_injur-shark-resnet_o2_d27_c2942_jzuddodd/model_state_arch_jzuddodd.pkl')
        """
        # import lasagne
        # arch_dpath = dirname(fpath)
        from wbia_cnn import custom_layers

        arch_json_fpath = '/home/joncrall/Desktop/manually_saved/arch_injur-shark-lenet_o2_d11_c688_acioqbst/arch_info.json'
        state_fpath = '/home/joncrall/Desktop/manually_saved/arch_injur-shark-lenet_o2_d11_c688_acioqbst/model_state_arch_acioqbst.pkl'
        output_layer = custom_layers.load_json_arch_def(arch_json_fpath)
        model.output_layer = output_layer

        model.load_model_state(fpath=state_fpath)

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

    def reinit_weights(model, W=None):
        """
        initailizes weights after the architecture has been defined.
        """
        import lasagne

        if W is None:
            W = 'orthogonal'
        if isinstance(W, six.string_types):
            if W == 'orthogonal':
                W = lasagne.init.Orthogonal()
        logger.info('Reinitializing all weights to %r' % (W,))
        weights_list = model.get_all_params(regularizable=True, trainable=True)
        # logger.info(weights_list)
        for weights in weights_list:
            # logger.info(weights)
            shape = weights.get_value().shape
            new_values = W.sample(shape)
            weights.set_value(new_values)

    # --- ABSTRACT FUNCTIONS

    def init_arch(model):
        raise NotImplementedError('reimplement')

    def augment(model, Xb, yb):
        raise NotImplementedError('data augmentation not implemented')

    def loss_function(model, network_output, truth):
        raise NotImplementedError('need to implement a loss function')


@ut.reloadable_class
class AbstractCategoricalModel(BaseModel):
    """base model for catagory classifiers"""

    def __init__(model, **kwargs):
        # BaseModel.__init__(model, **kwargs)
        # HACKING
        # <Prototype code to fix reload errors>
        # this_class_now = ut.fix_super_reload_error(AbstractCategoricalModel, model)
        this_class_now = AbstractCategoricalModel
        # super(AbstractCategoricalModel, model).__init__(**kwargs)
        super(this_class_now, model).__init__(**kwargs)

        # </Prototype code to fix reload errors>

        model.encoder = None
        # categorical models have a concept of accuracy
        # model.requested_headers += ['valid_acc', 'test_acc']
        model.requested_headers += ['valid_acc']

    def init_encoder(model, labels):
        logger.info('[model] encoding labels')
        from sklearn import preprocessing

        model.encoder = preprocessing.LabelEncoder()
        model.encoder.fit(labels)
        model.output_dims = len(list(np.unique(labels)))
        logger.info('[model] model.output_dims = %r' % (model.output_dims,))

    def loss_function(model, network_output, truth):
        # https://en.wikipedia.org/wiki/Loss_functions_for_classification
        from theano import tensor as T  # NOQA

        # categorical cross-entropy between predictions and targets
        # L_i = -\sum_{j} t_{i,j} \log{p_{i, j}}
        return T.nnet.categorical_crossentropy(network_output, truth)

    def custom_unlabeled_outputs(model, network_output):
        from theano import tensor as T  # NOQA

        # Network outputs define category probabilities
        probs = network_output
        preds = T.argmax(probs, axis=1)
        preds.name = 'predictions'
        confs = probs.max(axis=1)
        confs.name = 'confidences'
        unlabeled_outputs = [preds, confs]
        return unlabeled_outputs

    def custom_labeled_outputs(model, network_output, y_batch):
        from theano import tensor as T  # NOQA

        probs = network_output
        preds = T.argmax(probs, axis=1)
        preds.name = 'predictions'
        is_success = T.eq(preds, y_batch)
        accuracy = T.mean(is_success)
        accuracy.name = 'accuracy'
        labeled_outputs = [accuracy, preds]
        return labeled_outputs


@ut.reloadable_class
class AbstractVectorModel(BaseModel):
    """base model for catagory classifiers"""

    def __init__(model, **kwargs):
        # BaseModel.__init__(model, **kwargs)
        # HACKING
        # <Prototype code to fix reload errors>
        # this_class_now = ut.fix_super_reload_error(AbstractVectorModel, model)
        this_class_now = AbstractVectorModel
        # super(AbstractVectorModel, model).__init__(**kwargs)
        super(this_class_now, model).__init__(**kwargs)

        # </Prototype code to fix reload errors>

        model.encoder = None
        # categorical models have a concept of accuracy
        # model.requested_headers += ['valid_acc', 'test_acc']
        model.requested_headers += ['valid_acc']

    def init_output_dims(model, labels):
        model.output_dims = labels.shape[-1]
        logger.info('[model] model.output_dims = %r' % (model.output_dims,))

    # def loss_function(model, network_output, truth):
    #     from theano import tensor as T  # NOQA
    #     return T.nnet.binary_crossentropy(network_output, truth)

    def loss_function(model, network_output, truth):
        from theano import tensor as T  # NOQA

        return T.nnet.binary_crossentropy(network_output, truth)

    def custom_unlabeled_outputs(model, network_output):
        from theano import tensor as T  # NOQA

        # Network outputs define category probabilities
        probs = network_output
        preds = probs.round()
        preds.name = 'predictions'
        confs = probs
        confs.name = 'confidences'
        unlabeled_outputs = [preds, confs]
        return unlabeled_outputs

    def custom_labeled_outputs(model, network_output, y_batch):
        from theano import tensor as T  # NOQA

        probs = network_output
        preds = probs.clip(0.0, 1.0).round()
        preds.name = 'predictions'
        is_success = T.eq(preds, y_batch)
        shape = is_success.shape[1]
        is_success = T.sum(is_success, axis=1)
        is_success = T.eq(is_success, shape)
        accuracy = T.mean(is_success)
        accuracy.name = 'accuracy'
        labeled_outputs = [accuracy, preds]
        return labeled_outputs


@ut.reloadable_class
class AbstractVectorVectorModel(AbstractVectorModel):
    """base model for catagory classifiers"""

    def __init__(model, **kwargs):
        # BaseModel.__init__(model, **kwargs)
        # HACKING
        # <Prototype code to fix reload errors>
        # this_class_now = ut.fix_super_reload_error(AbstractVectorModel, model)
        this_class_now = AbstractVectorVectorModel
        # super(AbstractVectorModel, model).__init__(**kwargs)
        super(this_class_now, model).__init__(**kwargs)

    def custom_labeled_outputs(model, network_output, y_batch):
        from theano import tensor as T  # NOQA

        probs = network_output
        preds = probs.round()
        preds.name = 'predictions'
        is_success = T.eq(preds, y_batch)
        accuracy = T.mean(is_success)
        accuracy.name = 'accuracy'
        labeled_outputs = [accuracy, preds]
        return labeled_outputs


def report_error(msg):
    if False:
        raise ValueError(msg)
    else:
        logger.info('WARNING:' + msg)


# def evaluate_layer_list(network_layers_def, verbose=None):
#    r"""
#    compiles a sequence of partial functions into a network
#    """
#    if verbose is None:
#        verbose = VERBOSE_CNN
#    total = len(network_layers_def)
#    network_layers = []
#    if verbose:
#        logger.info('Evaluting List of %d Layers' % (total,))
#    layer_fn_iter = iter(network_layers_def)
#    try:
#        with ut.Indenter(' ' * 4, enabled=verbose):
#            next_args = tuple()
#            for count, layer_fn in enumerate(layer_fn_iter, start=1):
#                if verbose:
#                    logger.info('Evaluating layer %d/%d (%s) ' %
#                          (count, total, ut.get_funcname(layer_fn), ))
#                with ut.Timer(verbose=False) as tt:
#                    layer = layer_fn(*next_args)
#                next_args = (layer,)
#                network_layers.append(layer)
#                if verbose:
#                    logger.info('  * took %.4fs' % (tt.toc(),))
#                    logger.info('  * layer = %r' % (layer,))
#                    if hasattr(layer, 'input_shape'):
#                        logger.info('  * layer.input_shape = %r' % (
#                            layer.input_shape,))
#                    if hasattr(layer, 'shape'):
#                        logger.info('  * layer.shape = %r' % (
#                            layer.shape,))
#                    logger.info('  * layer.output_shape = %r' % (
#                        layer.output_shape,))
#    except Exception as ex:
#        keys = ['layer_fn', 'layer_fn.func', 'layer_fn.args',
#                'layer_fn.keywords', 'layer_fn.__dict__', 'layer', 'count']
#        ut.printex(ex, ('Error building layers.\n' 'layer.name=%r') % (layer),
#                   keys=keys)
#        raise
#    return network_layers


def testdata_model_with_history():
    model = BaseModel()
    # make a dummy history
    X_train, y_train = [1, 2, 3], [0, 0, 1]
    model.output_dims = 2
    rng = np.random.RandomState(0)

    def rand():
        return rng.rand() / 10

    def dummy_epoch_dict(num):
        from scipy.special import expit

        mean_loss = 1 / np.exp(num / 10)
        frac = num / total_epochs
        epoch_info = {
            'epoch_num': num,
            'learn_loss': mean_loss + rand(),
            'learn_loss_reg': (mean_loss + np.exp(rand() * num + rand())),
            'learn_loss_std': rand(),
            'valid_loss': mean_loss - rand(),
            'valid_loss_std': rand(),
            'valid_acc': expit(-6 + 12 * np.clip(frac + rand(), 0, 1)),
            'valid_acc_std': rand(),
            'learn_acc': expit(-6 + 12 * np.clip(frac + rand(), 0, 1)),
            'learn_acc_std': rand(),
            'valid_precision': [rng.rand() for _ in range(model.output_dims)],
            'valid_recall': [rng.rand() for _ in range(model.output_dims)],
            'learn_precision': [rng.rand() for _ in range(model.output_dims)],
            'learn_recall': [rng.rand() for _ in range(model.output_dims)],
            'param_update_mags': {
                'C0': (rng.normal() ** 2, rng.rand()),
                'F1': (rng.normal() ** 2, rng.rand()),
            },
            'learn_state': {'learning_rate': 0.01},
        }
        return epoch_info

    def dummy_get_all_layer_info(model):
        return [
            {'param_infos': [{'name': 'C0', 'tags': ['trainable']}]},
            {'param_infos': [{'name': 'F1', 'tags': ['trainable']}]},
        ]

    ut.inject_func_as_method(
        model, dummy_get_all_layer_info, 'get_all_layer_info', allow_override=True
    )
    epoch = 0
    eralens = [4, 4, 4]
    total_epochs = sum(eralens)
    for era_length in eralens:
        model.history._new_era(model, X_train, y_train, X_train, y_train)
        for _ in range(era_length):
            epoch_info = dummy_epoch_dict(num=epoch)
            model.history._record_epoch(epoch_info)
            epoch += 1
    return model


if __name__ == '__main__':
    """
    CommandLine:
        python -m wbia_cnn.abstract_models
        python -m wbia_cnn.abstract_models --allexamples
        python -m wbia_cnn.abstract_models --allexamples --noface --nosrc
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
