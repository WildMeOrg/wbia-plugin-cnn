# -*- coding: utf-8 -*-
"""
tests a test set of data using a specified, pre-trained model and weights

python -c "import wbia_cnn"
"""
import logging
from wbia_cnn import models
from wbia_cnn import _plugin_grabmodels as grabmodels
import utool as ut
import six
import numpy as np
import random
import os
import wbia.constants as const
from six.moves import zip, range

print, rrr, profile = ut.inject2(__name__)
logger = logging.getLogger()


try:
    from wbia.control.controller_inject import make_ibs_register_decorator
    from wbia.constants import VIEWTEXT_TO_YAW_RADIANS

    CLASS_INJECT_KEY, register_ibs_method = make_ibs_register_decorator(__name__)
except ImportError:
    register_ibs_method = ut.identity
    raise


def convert_species_viewpoint(species, viewpoint):
    species_mapping = {
        'ZEBRA_PLAINS': 'zebra_plains',
        'ZEBRA_GREVYS': 'zebra_grevys',
        'ELEPHANT_SAVANNA': 'elephant_savanna',
        'GIRAFFE_RETICULATED': 'giraffe_reticulated',
        'GIRAFFE_MASAI': 'giraffe_masai',
    }
    viewpoint_list = VIEWTEXT_TO_YAW_RADIANS.keys()
    viewpoint_mapping = {
        'LEFT': viewpoint_list[4],
        'FRONT_LEFT': viewpoint_list[3],
        'FRONT': viewpoint_list[2],
        'FRONT_RIGHT': viewpoint_list[1],
        'RIGHT': viewpoint_list[0],
        'BACK_RIGHT': viewpoint_list[7],
        'BACK': viewpoint_list[6],
        'BACK_LEFT': viewpoint_list[5],
    }
    species_ = species_mapping[species]
    viewpoint_ = viewpoint_mapping[viewpoint]
    return species_, viewpoint_


def convert_label(label):
    species, viewpoint = label.strip().split(':')
    species = species.strip()
    viewpoint = viewpoint.strip()
    species_, viewpoint_ = convert_species_viewpoint(species, viewpoint)
    return species_, viewpoint_


@register_ibs_method
def get_neuralnet_dir(ibs):
    nets_dir = ut.unixjoin(ibs.get_cachedir(), ibs.const.PATH_NAMES.nets)
    return nets_dir


@register_ibs_method
def get_verified_aid_pairs(ibs):
    """
    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia_cnn._plugin import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb('NNP_Master3', allow_newdir=True)
        >>> verified_aid1_list, verified_aid2_list = get_verified_aid_pairs(ibs)
    """
    # Grab marked hard cases
    am_rowids = ibs._get_all_annotmatch_rowids()
    remove_photobombs = True
    if remove_photobombs:
        flags = ibs.get_annotmatch_is_photobomb(am_rowids)
        am_rowids = ut.filterfalse_items(am_rowids, flags)
    verified_aid1_list = ibs.get_annotmatch_aid1(am_rowids)
    verified_aid2_list = ibs.get_annotmatch_aid2(am_rowids)
    return verified_aid1_list, verified_aid2_list


@register_ibs_method
def generate_thumbnail_class_list(
    ibs, thumbnail_list, nInput=None, classifier_weight_filepath=None, **kwargs
):

    # Load chips and resize to the target
    data_shape = (192, 192, 3)
    batch_size = None
    # Define model and load weights
    logger.info('\n[wbia_cnn] Loading model...')
    if nInput is None:
        try:
            nInput = len(thumbnail_list)
        except TypeError:
            logger.info('Warning passed in generator without specifying nInput hint')
            logger.info('Explicitly evaluating generator')
            logger.info('type(chip_list) = %r' % (type(thumbnail_list),))
            thumbnail_list = list(thumbnail_list)
            nInput = len(thumbnail_list)

    model = models.ClassifierModel(batch_size=batch_size, data_shape=data_shape)

    if classifier_weight_filepath in [None, 'v3_zebra']:
        weights_path = grabmodels.ensure_model('classifier_v3_zebra', redownload=False)
    elif classifier_weight_filepath in ['coco_zebra']:
        weights_path = grabmodels.ensure_model('classifier_coco_zebra', redownload=False)
    elif classifier_weight_filepath in ['megan1.1']:
        weights_path = grabmodels.ensure_model(
            'classifier_cameratrap_megan_v1', redownload=False
        )
    elif classifier_weight_filepath in ['megan1.2']:
        weights_path = grabmodels.ensure_model(
            'classifier_cameratrap_megan_v2', redownload=False
        )
    elif classifier_weight_filepath in ['megan1.3']:
        weights_path = grabmodels.ensure_model(
            'classifier_cameratrap_megan_v3', redownload=False
        )
    elif classifier_weight_filepath in ['megan1.4']:
        weights_path = grabmodels.ensure_model(
            'classifier_cameratrap_megan_v4', redownload=False
        )
    elif classifier_weight_filepath in ['megan1.5']:
        weights_path = grabmodels.ensure_model(
            'classifier_cameratrap_megan_v5', redownload=False
        )
    elif classifier_weight_filepath in ['megan1.6']:
        weights_path = grabmodels.ensure_model(
            'classifier_cameratrap_megan_v6', redownload=False
        )
    elif classifier_weight_filepath in ['megan2.1']:
        weights_path = grabmodels.ensure_model(
            'classifier_cameratrap_megan2_v1', redownload=False
        )
    elif classifier_weight_filepath in ['megan2.2']:
        weights_path = grabmodels.ensure_model(
            'classifier_cameratrap_megan2_v2', redownload=False
        )
    elif classifier_weight_filepath in ['megan2.3']:
        weights_path = grabmodels.ensure_model(
            'classifier_cameratrap_megan2_v3', redownload=False
        )
    elif classifier_weight_filepath in ['megan2.4']:
        weights_path = grabmodels.ensure_model(
            'classifier_cameratrap_megan2_v4', redownload=False
        )
    elif classifier_weight_filepath in ['megan2.5']:
        weights_path = grabmodels.ensure_model(
            'classifier_cameratrap_megan2_v5', redownload=False
        )
    elif classifier_weight_filepath in ['megan2.6']:
        weights_path = grabmodels.ensure_model(
            'classifier_cameratrap_megan2_v6', redownload=False
        )
    elif classifier_weight_filepath in ['ryan.wbia_cnn.v1']:
        weights_path = grabmodels.ensure_model(
            'classifier_cameratrap_ryan_cnn_v1', redownload=False
        )
    elif os.path.exists(classifier_weight_filepath):
        weights_path = classifier_weight_filepath
    else:
        raise ValueError('Classifier does not have a valid trained model')

    model_state_fpath = model.get_model_state_fpath(fpath=weights_path)
    logger.info('[model] loading model state from: %s' % (model_state_fpath,))
    model_state = ut.load_cPkl(model_state_fpath)

    model.encoder = model_state.get('encoder', None)
    model.output_dims = model_state['output_dims']
    model.data_params = model_state['data_params']
    model._fix_center_mean_std()

    model.init_arch()
    model.batch_size = 128
    model.hyperparams['whiten_on'] = True
    model.best_results = model_state['best_results']
    model.set_all_param_values(model.best_results['weights'])

    # Create the Theano primitives
    # create theano symbolic expressions that define the network
    logger.info('\n[wbia_cnn] --- COMPILING SYMBOLIC THEANO FUNCTIONS ---')
    logger.info('[model] creating Theano primitives...')
    theano_predict = model.build_predict_func()

    logger.info('[wbia_cnn] Performing inference...')
    test_results = model.process_batch(theano_predict, np.array(thumbnail_list))

    prediction_list = model.encoder.inverse_transform(test_results['predictions'])
    confidence_list = test_results['confidences']

    result_list = list(zip(confidence_list, prediction_list))
    return result_list


@register_ibs_method
def generate_thumbnail_class2_list(
    ibs, thumbnail_list, nInput=None, classifier_two_weight_filepath=None, **kwargs
):

    # Load chips and resize to the target
    data_shape = (192, 192, 3)
    batch_size = None
    # Define model and load weights
    logger.info('\n[wbia_cnn] Loading model...')
    if nInput is None:
        try:
            nInput = len(thumbnail_list)
        except TypeError:
            logger.info('Warning passed in generator without specifying nInput hint')
            logger.info('Explicitly evaluating generator')
            logger.info('type(chip_list) = %r' % (type(thumbnail_list),))
            thumbnail_list = list(thumbnail_list)
            nInput = len(thumbnail_list)

    model = models.Classifier2Model(batch_size=batch_size, data_shape=data_shape)

    if classifier_two_weight_filepath in [None, 'v3']:
        weights_path = grabmodels.ensure_model('classifier2_v3', redownload=False)
    elif classifier_two_weight_filepath in ['candidacy']:
        weights_path = grabmodels.ensure_model('classifier2_candidacy', redownload=False)
    elif classifier_two_weight_filepath in ['ggr2']:
        weights_path = grabmodels.ensure_model('classifier2_ggr2', redownload=False)
    elif os.path.exists(classifier_two_weight_filepath):
        weights_path = classifier_two_weight_filepath
    else:
        raise ValueError('Classifier does not have a valid trained model')

    model_state_fpath = model.get_model_state_fpath(fpath=weights_path)
    logger.info('[model] loading model state from: %s' % (model_state_fpath,))
    model_state = ut.load_cPkl(model_state_fpath)

    category_list = model_state['category_list']
    model.encoder = model_state.get('encoder', None)
    model.output_dims = model_state['output_dims']
    model.data_params = model_state['data_params']
    model._fix_center_mean_std()

    model.init_arch()
    model.batch_size = 128
    model.hyperparams['whiten_on'] = True
    model.best_results = model_state['best_results']
    model.set_all_param_values(model.best_results['weights'])

    # Create the Theano primitives
    # create theano symbolic expressions that define the network
    logger.info('\n[wbia_cnn] --- COMPILING SYMBOLIC THEANO FUNCTIONS ---')
    logger.info('[model] creating Theano primitives...')
    theano_predict = model.build_predict_func()

    logger.info('[wbia_cnn] Performing inference...')
    test_results = model.process_batch(theano_predict, np.array(thumbnail_list))

    confidences_list = test_results['confidences']
    confidences_list[confidences_list > 1.0] = 1.0
    confidences_list[confidences_list < 0.0] = 0.0

    confidence_dict_list = [
        dict(zip(category_list, confidence_list)) for confidence_list in confidences_list
    ]

    # zipped = zip(thumbnail_list, confidence_dict_list)
    # for index, (thumbnail, confidence_dict) in enumerate(zipped):
    #     logger.info(index)
    #     y = []
    #     for key in confidence_dict:
    #         y.append('%s-%0.04f' % (key, confidence_dict[key], ))
    #     y = ';'.join(y)
    #     image_path = '/home/jason/Desktop/batch2/image-%s-%s.png'
    #     cv2.imwrite(image_path % (index, y), thumbnail)

    predictions_list = [
        [key for key in confidence_dict if confidence_dict[key] >= 0.5]
        for confidence_dict in confidence_dict_list
    ]

    result_list = list(zip(confidence_dict_list, predictions_list))
    return result_list


@register_ibs_method
def generate_thumbnail_aoi2_list(
    ibs,
    thumbnail_list,
    bbox_list,
    size_list,
    nInput=None,
    aoi_two_weight_filepath=None,
    **kwargs
):
    # Load chips and resize to the target
    data_shape = (192, 192, 4)
    batch_size = None
    # Define model and load weights
    logger.info('\n[wbia_cnn] Loading model...')
    if nInput is None:
        try:
            nInput = len(thumbnail_list)
        except TypeError:
            logger.info('Warning passed in generator without specifying nInput hint')
            logger.info('Explicitly evaluating generator')
            logger.info('type(chip_list) = %r' % (type(thumbnail_list),))
            thumbnail_list = list(thumbnail_list)
            nInput = len(thumbnail_list)

    model = models.AoI2Model(batch_size=batch_size, data_shape=data_shape)

    if aoi_two_weight_filepath in [None, 'candidacy']:
        weights_path = grabmodels.ensure_model('aoi2_candidacy', redownload=False)
    elif aoi_two_weight_filepath in ['ggr2']:
        weights_path = grabmodels.ensure_model('aoi2_ggr2', redownload=False)
    elif aoi_two_weight_filepath in ['hammerhead']:
        weights_path = grabmodels.ensure_model('aoi2_hammerhead', redownload=False)
    elif aoi_two_weight_filepath in ['jaguar']:
        weights_path = grabmodels.ensure_model('aoi2_jaguar', redownload=False)
    elif os.path.exists(aoi_two_weight_filepath):
        weights_path = aoi_two_weight_filepath
    else:
        raise ValueError('AoI2 does not have a valid trained model')

    model_state_fpath = model.get_model_state_fpath(fpath=weights_path)
    logger.info('[model] loading model state from: %s' % (model_state_fpath,))
    model_state = ut.load_cPkl(model_state_fpath)

    model.encoder = model_state.get('encoder', None)
    model.output_dims = model_state['output_dims']
    model.data_params = model_state['data_params']
    model._fix_center_mean_std()

    model.init_arch()
    model.batch_size = 128
    model.hyperparams['whiten_on'] = True
    model.best_results = model_state['best_results']
    model.set_all_param_values(model.best_results['weights'])

    # Create the Theano primitives
    # create theano symbolic expressions that define the network
    logger.info('\n[wbia_cnn] --- COMPILING SYMBOLIC THEANO FUNCTIONS ---')
    logger.info('[model] creating Theano primitives...')
    theano_predict = model.build_predict_func()

    mask = np.zeros((192, 192, 1), dtype=np.uint8)
    data_list = []
    for thumbnail, bbox, size in zip(thumbnail_list, bbox_list, size_list):
        xtl, ytl, width, height = bbox
        w, h = size
        w = float(w)
        h = float(h)
        xbr = xtl + width
        ybr = ytl + height
        xtl = int(np.round((xtl / w) * data_shape[1]))
        ytl = int(np.round((ytl / h) * data_shape[0]))
        xbr = int(np.round((xbr / w) * data_shape[1]))
        ybr = int(np.round((ybr / h) * data_shape[0]))
        mask_ = np.copy(mask)
        mask_[ytl:ybr, xtl:xbr] = 255
        data = np.dstack((thumbnail, mask_))
        data_list.append(data)

    logger.info('[wbia_cnn] Performing inference...')
    test_results = model.process_batch(theano_predict, np.array(data_list))

    confidence_list = test_results['confidences']
    prediction_list = test_results['predictions']
    prediction_list = [
        'positive' if prediction == 1 else 'negative' for prediction in prediction_list
    ]

    result_list = list(zip(confidence_list, prediction_list))
    return result_list


@register_ibs_method
def generate_chip_label_list(
    ibs, chip_list, nInput=None, labeler_weight_filepath=None, **kwargs
):

    # Load chips and resize to the target
    data_shape = (128, 128, 3)
    batch_size = None
    # Define model and load weights
    logger.info('\n[wbia_cnn] Loading model...')
    if nInput is None:
        try:
            nInput = len(chip_list)
        except TypeError:
            logger.info('Warning passed in generator without specifying nInput hint')
            logger.info('Explicitly evaluating generator')
            logger.info('type(chip_list) = %r' % (type(chip_list),))
            chip_list = list(chip_list)
            nInput = len(chip_list)

    model = models.LabelerModel(batch_size=batch_size, data_shape=data_shape)

    if labeler_weight_filepath in [None, 'v3']:
        weights_path = grabmodels.ensure_model('labeler_v3', redownload=False)
    elif labeler_weight_filepath in ['v1']:
        weights_path = grabmodels.ensure_model('labeler_v1', redownload=False)
    elif labeler_weight_filepath in ['cheetah']:
        weights_path = grabmodels.ensure_model('labeler_cheetah_v0', redownload=False)
    elif labeler_weight_filepath in ['lynx', 'lynx_pardinus']:
        weights_path = grabmodels.ensure_model('labeler_lynx_v2', redownload=False)
    elif labeler_weight_filepath in ['candidacy']:
        weights_path = grabmodels.ensure_model('labeler_candidacy', redownload=False)
    elif labeler_weight_filepath in ['jaguar', 'jaguar_v2']:
        weights_path = grabmodels.ensure_model('labeler_jaguar_v2', redownload=False)
    elif labeler_weight_filepath in ['manta']:
        weights_path = grabmodels.ensure_model('labeler_manta', redownload=False)
    elif labeler_weight_filepath in ['dorsal', 'hendrik_dorsal']:
        weights_path = grabmodels.ensure_model('labeler_hendrik_dorsal', redownload=False)
    elif labeler_weight_filepath in ['seaturtle']:
        weights_path = grabmodels.ensure_model('labeler_seaturtle_v2', redownload=False)
    elif os.path.exists(labeler_weight_filepath):
        weights_path = labeler_weight_filepath
    else:
        raise ValueError('Labeler does not have a valid trained model')

    model_state_fpath = model.get_model_state_fpath(fpath=weights_path)
    logger.info('[model] loading model state from: %s' % (model_state_fpath,))
    model_state = ut.load_cPkl(model_state_fpath)

    model.encoder = model_state.get('encoder', None)
    model.output_dims = model_state['output_dims']
    model.data_params = model_state['data_params']
    model._fix_center_mean_std()

    model.best_results = model_state['best_results']

    model.init_arch()
    model.batch_size = 128
    model.hyperparams['whiten_on'] = True
    model.set_all_param_values(model.best_results['weights'])

    # Create the Theano primitives
    # create theano symbolic expressions that define the network
    logger.info('\n[wbia_cnn] --- COMPILING SYMBOLIC THEANO FUNCTIONS ---')
    logger.info('[model] creating Theano primitives...')
    theano_predict = model.build_predict_func()

    logger.info('[wbia_cnn] Performing inference...')
    test_results = model.process_batch(theano_predict, np.array(chip_list))

    class_list = list(model.encoder.classes_)
    prediction_list = model.encoder.inverse_transform(test_results['predictions'])
    confidence_list = test_results['confidences']
    probability_list = test_results['network_output_determ']

    class_list = list(
        map(
            lambda x: x if isinstance(x, six.text_type) else x.decode('utf-8'), class_list
        )
    )
    prediction_list = list(
        map(
            lambda x: x if isinstance(x, six.text_type) else x.decode('utf-8'),
            prediction_list,
        )
    )

    species_list = []
    viewpoint_list = []
    for prediction in prediction_list:
        prediction = prediction.strip()
        if ':' in prediction:
            prediction = prediction.split(':')
            species, viewpoint = prediction
        else:
            species = prediction
            viewpoint = None
        if species.lower() == 'ignore':
            species = const.UNKNOWN
        species_list.append(species)
        viewpoint_list.append(viewpoint)

    quality_list = [const.QUAL_UNKNOWN] * len(prediction_list)
    orientation_list = [0.0] * len(prediction_list)

    probability_dict_list = []
    for probability in probability_list:
        probability_dict = {class_: prob for class_, prob in zip(class_list, probability)}
        probability_dict_list.append(probability_dict)

    result_list = list(
        zip(
            confidence_list,
            species_list,
            viewpoint_list,
            quality_list,
            orientation_list,
            probability_dict_list,
        )
    )

    return result_list


@register_ibs_method
def detect_annot_zebra_background_mask(ibs, aid_list, species=None, config2_=None):
    r"""
    Args:
        ibs (IBEISController):  ibeis controller object
        aid_list (int):  list of annotation ids

    Returns:
        list: mask_list
    """
    # Read the data
    logger.info('\n[wbia_cnn] Loading chips...')
    chip_list = ibs.get_annot_chips(aid_list, verbose=True, config2_=config2_)
    mask_list = list(generate_species_background(ibs, chip_list, species=species))
    return mask_list


@register_ibs_method
def detect_annot_whale_fluke_background_mask(
    ibs, aid_list, species='whale_fluke', config2_=None
):
    r"""
    Args:
        ibs (IBEISController):  ibeis controller object
        aid_list (int):  list of annotation ids

    Returns:
        list: mask_list
    """
    # Read the data
    logger.info('\n[wbia_cnn] Loading chips...')
    chip_list = ibs.get_annot_chips(aid_list, verbose=True, config2_=config2_)
    mask_list = list(generate_species_background(ibs, chip_list, species=species))
    return mask_list


@register_ibs_method
def generate_species_background_mask(ibs, chip_fpath_list, species=None):
    r"""
    Args:
        ibs (IBEISController):  ibeis controller object
        aid_list (int):  list of annotation ids

    Returns:
        list: species_viewpoint_list

    CommandLine:
        python -m wbia_cnn._plugin --exec-generate_species_background_mask --show --db PZ_Master1
        python -m wbia_cnn --tf generate_species_background_mask --show --db PZ_Master1 --aid 9970

    Example:
        >>> # DISABLE_DOCTEST
        >>> import wbia_cnn
        >>> import wbia
        >>> from wbia_cnn._plugin import *  # NOQA
        >>> ibs = wbia.opendb(defaultdb='testdb1', allow_newdir=True)
        >>> aid_list = ut.get_argval(('--aids', '--aid'), type_=list, default=ibs.get_valid_aids()[0:2])
        >>> chip_fpath_list = ibs.get_annot_chip_fpath(aid_list)
        >>> species = ibs.const.TEST_SPECIES.ZEB_PLAIN
        >>> mask_list = generate_species_background_mask(ibs, chip_fpath_list, species)
        >>> mask_list = list(mask_list)
        >>> ut.quit_if_noshow()
        >>> from wbia import plottool as pt
        >>> iteract_obj = pt.interact_multi_image.MultiImageInteraction(mask_list, nPerPage=4)
        >>> #pt.imshow(mask_list[0])
        >>> ut.show_if_requested()

        #>>> from wbia_cnn.draw_results import *  # NOQA
        #>>> from wbia_cnn import ingest_data
        #>>> data, labels = ingest_data.testdata_patchmatch2()
        #>>> flat_metadata = {'fs': np.arange(len(labels))}
        #>>> result = interact_siamsese_data_patches(labels, data, flat_metadata)
        #>>> ut.show_if_requested()
    """
    # Read the data
    logger.info('\n[wbia_cnn] Loading chips...')
    import vtool as vt

    nInput = len(chip_fpath_list)

    def bufgen2(_iter, size=64, nInput=None, **kwargs):
        nTotal = None if nInput is None else int(np.ceil(nInput / size))
        chunk_iter = ut.ichunks(_iter, size)
        chunk_iter_ = ut.ProgressIter(chunk_iter, nTotal=nTotal, **kwargs)
        for chunk in chunk_iter_:
            for item in chunk:
                yield item

    chip_list = bufgen2(
        (vt.imread(fpath) for fpath in chip_fpath_list),
        lbl='loading chip chunk',
        nInput=nInput,
        adjust=True,
        time_thresh=30.0,
    )
    # mask_list = list(generate_species_background(ibs, chip_list, species=species, nInput=nInput))
    mask_gen = generate_species_background(ibs, chip_list, species=species, nInput=nInput)
    return mask_gen


@register_ibs_method
def generate_species_background(ibs, chip_list, species=None, nInput=None):
    """
    TODO: Use this as the primary function

    CommandLine:
        python -m wbia_cnn._plugin --exec-generate_species_background --db PZ_MTEST --species=zebra_plains --show
        python -m wbia_cnn._plugin --exec-generate_species_background --db GZ_Master1 --species=zebra_grevys --save cnn_detect_results_gz.png --diskshow --clipwhite
        python -m wbia_cnn._plugin --exec-generate_species_background --db PZ_Master1 --species=zebra_plains --save cnn_detect_results_pz.png --diskshow --clipwhite
        python -m wbia_cnn._plugin --exec-generate_species_background --db PZ_Master1 --show
        python -m wbia_cnn._plugin --exec-generate_species_background --db GZ_Master1 --show
        python -m wbia_cnn._plugin --exec-generate_species_background --db GIRM_Master1 --show --species=giraffe_masai

    Example:
        >>> # ENABLE_DOCTEST
        >>> import wbia_cnn
        >>> import wbia
        >>> from wbia_cnn._plugin import *  # NOQA
        >>> ibs = wbia.opendb(defaultdb='testdb1', allow_newdir=True)
        >>> aid_list = ibs.get_valid_aids()[0:8]
        >>> species = ut.get_argval('--species', type_=str, default='zebra_plains')
        >>> config2_ = None
        >>> nInput = len(aid_list)
        >>> chip_iter = ibs.get_annot_chips(aid_list, verbose=True, config2_=config2_, eager=False)
        >>> mask_iter = generate_species_background(ibs, chip_iter, species=species, nInput=nInput)
        >>> mask_list = list(mask_iter)
        >>> ut.quit_if_noshow()
        >>> from wbia import plottool as pt
        >>> import vtool as vt
        >>> chip_list = ibs.get_annot_chips(aid_list, verbose=True, config2_=config2_, eager=True)
        >>> stacked_list = [vt.stack_images(chip, mask)[0] for chip, mask in  zip(chip_list, mask_list)]
        >>> iteract_obj = pt.interact_multi_image.MultiImageInteraction(stacked_list, nPerPage=4)
        >>> iteract_obj.start()
        >>> #hough_cpath = ibs.get_annot_probchip_fpath(aid_list, config2_=config2_)
        >>> #iteract_obj2 = pt.interact_multi_image.MultiImageInteraction(hough_cpath, nPerPage=4)
        >>> #pt.imshow(mask_list[0])
        >>> ut.show_if_requested()

    Ignore:
        #>>> from wbia_cnn.draw_results import *  # NOQA
        #>>> from wbia_cnn import ingest_data
        #>>> data, labels = ingest_data.testdata_patchmatch2()
        #>>> flat_metadata = {'fs': np.arange(len(labels))}
        #>>> result = interact_siamsese_data_patches(labels, data, flat_metadata)
        #>>> ut.show_if_requested()
    """
    if species is None:
        # species = 'zebra_plains'
        raise ValueError('must specify a species for the background detector')

    # Load chips and resize to the target
    data_shape = (256, 256, 3)
    # Define model and load weights
    logger.info('\n[wbia_cnn] Loading model...')
    if nInput is None:
        try:
            nInput = len(chip_list)
        except TypeError:
            logger.info('Warning passed in generator without specifying nInput hint')
            logger.info('Explicitly evaluating generator')
            logger.info('type(chip_list) = %r' % (type(chip_list),))
            chip_list = list(chip_list)
            nInput = len(chip_list)

    # batch_size = int(min(128, 2 ** np.floor(np.log2(nInput))))
    batch_size = None

    LEGACY = True
    NEW = True
    confidence_thresh = 0.5
    logger.info(species)

    candidacy_species_list = [
        'giraffe_masai',
        'giraffe_reticulated',
        # 'turtle_sea',
        'whale_fluke',
        'zebra_grevys',
        'zebra_plains',
        'giraffa_tippelskirchi',
        'giraffa_camelopardalis_reticulata',
        'megaptera_novaeangliae',
        'equus_grevyi',
        'equus_quagga',
    ]

    CANDIDACY = True
    CANDIDACY = CANDIDACY and species in candidacy_species_list

    if CANDIDACY:
        species_list = candidacy_species_list
        assert species in species_list

        if species in ['giraffa_tippelskirchi']:
            species = 'giraffe_masai'
        if species in ['giraffa_camelopardalis_reticulata']:
            species = 'giraffe_reticulated'
        if species in ['megaptera_novaeangliae']:
            species = 'whale_fluke'
        if species in ['equus_grevyi']:
            species = 'zebra_grevys'
        if species in ['equus_quagga']:
            species = 'zebra_plains'

        LEGACY = False
        confidence_thresh = 0.2
        model = models.BackgroundModel(batch_size=batch_size, data_shape=data_shape)
        weights_path = grabmodels.ensure_model(
            'background_candidacy_' + species, redownload=False
        )
        canvas_key = 1
    elif species in ['zebra_plains', 'equus_quagga', 'zebra_grevys', 'equus_grevyi']:
        if NEW:
            assert species in [
                'zebra_plains',
                'equus_quagga',
                'zebra_grevys',
                'equus_grevyi',
            ]
            model = models.BackgroundModel(
                batch_size=batch_size, data_shape=data_shape, num_output=3
            )
            weights_path = grabmodels.ensure_model(
                'background_zebra_plains_grevys', redownload=False
            )
            canvas_key = species
        else:
            assert species in ['zebra_plains', 'equus_quagga']
            model = models.BackgroundModel(batch_size=batch_size, data_shape=data_shape)
            weights_path = grabmodels.ensure_model(
                'background_zebra_plains', redownload=False
            )
            canvas_key = 'positive'
    elif species in ['zebra_mountain', 'equus_zebra']:
        LEGACY = False
        species = 'zebra_mountain'  # Misspelled from zebra_mountain during training
        confidence_thresh = 0.2
        model = models.BackgroundModel(batch_size=batch_size, data_shape=data_shape)
        weights_path = grabmodels.ensure_model(
            'background_zebra_mountain_v0', redownload=False
        )
        canvas_key = 1
    elif species in ['giraffe_masai', 'giraffa_tippelskirchi']:
        model = models.BackgroundModel(batch_size=batch_size, data_shape=data_shape)
        weights_path = grabmodels.ensure_model(
            'background_giraffe_masai', redownload=False
        )
        canvas_key = 'giraffe_masai'
    elif species in [
        'whale_fluke',
        'whale_humpback',
        'megaptera_novaeangliae',
        # 'physeter_macrocephalus',
    ]:
        LEGACY = False
        species = 'whale_fluke'
        confidence_thresh = 0.2
        model = models.BackgroundModel(batch_size=batch_size, data_shape=data_shape)
        weights_path = grabmodels.ensure_model(
            'background_candidacy_whale_fluke', redownload=False
        )
        canvas_key = 1
        # species = 'whale_fluke'
        # model = models.BackgroundModel(batch_size=batch_size, data_shape=data_shape)
        # weights_path = grabmodels.ensure_model('background_whale_fluke', redownload=False)
        # canvas_key = species
    elif species in ['lynx', 'lynx_pardinus']:
        LEGACY = False
        species = 'lynx'
        confidence_thresh = 0.2
        model = models.BackgroundModel(batch_size=batch_size, data_shape=data_shape)
        weights_path = grabmodels.ensure_model('background_lynx_v3', redownload=False)
        canvas_key = 1
    elif species in ['cheetah', 'acinonyx_jubatus']:
        LEGACY = False
        species = 'cheetah'
        confidence_thresh = 0.2
        model = models.BackgroundModel(batch_size=batch_size, data_shape=data_shape)
        weights_path = grabmodels.ensure_model('background_cheetah_v2', redownload=False)
        canvas_key = 1
    elif species in ['hyaena', 'hyaena_hyaena']:
        LEGACY = False
        species = 'hyaena'
        confidence_thresh = 0.2
        model = models.BackgroundModel(batch_size=batch_size, data_shape=data_shape)
        weights_path = grabmodels.ensure_model('background_hyaena_v0', redownload=False)
        canvas_key = 1
    elif species in ['jaguar', 'panthera_onca']:
        LEGACY = False
        species = 'jaguar'
        confidence_thresh = 0.2
        model = models.BackgroundModel(batch_size=batch_size, data_shape=data_shape)
        weights_path = grabmodels.ensure_model('background_jaguar_v2', redownload=False)
        canvas_key = 1
    elif species in [
        'manta_ray_giant',
        'mobula_birostris',
        'manta_birostris',
        'mobula_alfredi',
    ]:
        LEGACY = False
        species = 'manta'
        confidence_thresh = 0.2
        model = models.BackgroundModel(batch_size=batch_size, data_shape=data_shape)
        weights_path = grabmodels.ensure_model('background_manta', redownload=False)
        canvas_key = 1
    elif species in ['skunk_spotted', 'spilogale_gracilis']:
        LEGACY = False
        species = 'skunk_spotted'
        confidence_thresh = 0.2
        model = models.BackgroundModel(batch_size=batch_size, data_shape=data_shape)
        weights_path = grabmodels.ensure_model(
            'background_skunk_spotted_v1', redownload=False
        )
        canvas_key = 1
    elif species in ['right_whale_head', 'eubalaena_australis', 'eubalaena_glacialis']:
        LEGACY = False
        species = 'right_whale_head'
        confidence_thresh = 0.2
        model = models.BackgroundModel(batch_size=batch_size, data_shape=data_shape)
        weights_path = grabmodels.ensure_model(
            'background_right_whale_head_v0', redownload=False
        )
        canvas_key = 1
    elif species in [
        'whale_orca',
        'whale_orca+fin_dorsal',
        'orcinus_orca',
        'orcinus_orca+fin_dorsal',
    ]:
        LEGACY = False
        species = 'whale_orca'
        confidence_thresh = 0.2
        model = models.BackgroundModel(batch_size=batch_size, data_shape=data_shape)
        weights_path = grabmodels.ensure_model('background_orca_v0', redownload=False)
        canvas_key = 1
    elif species in [
        'whale_sperm',
        'whale_sperm+fluke',
        'physeter_macrocephalus',
        'physeter_macrocephalus+fluke',
    ]:
        LEGACY = False
        species = 'whale_sperm+fluke'
        confidence_thresh = 0.2
        model = models.BackgroundModel(batch_size=batch_size, data_shape=data_shape)
        weights_path = grabmodels.ensure_model(
            'background_whale_sperm_v0', redownload=False
        )
        canvas_key = 1
    elif species in ['seadragon_leafy', 'phycodurus_eques']:
        LEGACY = False
        species = 'seadragon_leafy'
        confidence_thresh = 0.2
        model = models.BackgroundModel(batch_size=batch_size, data_shape=data_shape)
        weights_path = grabmodels.ensure_model(
            'background_seadragon_leafy_v1', redownload=False
        )
        canvas_key = 1
    elif species in ['seadragon_weedy', 'phyllopteryx_taeniolatus']:
        LEGACY = False
        species = 'seadragon_weedy'
        confidence_thresh = 0.2
        model = models.BackgroundModel(batch_size=batch_size, data_shape=data_shape)
        weights_path = grabmodels.ensure_model(
            'background_seadragon_weedy_v1', redownload=False
        )
        canvas_key = 1
    elif species in ['seadragon_leafy+head', 'phycodurus_eques+head']:
        LEGACY = False
        species = 'seadragon_leafy+head'
        confidence_thresh = 0.2
        model = models.BackgroundModel(batch_size=batch_size, data_shape=data_shape)
        weights_path = grabmodels.ensure_model(
            'background_seadragon_leafy_head_v1', redownload=False
        )
        canvas_key = 1
    elif species in ['seadragon_weedy+head', 'phyllopteryx_taeniolatus+head']:
        LEGACY = False
        species = 'seadragon_weedy+head'
        confidence_thresh = 0.2
        model = models.BackgroundModel(batch_size=batch_size, data_shape=data_shape)
        weights_path = grabmodels.ensure_model(
            'background_seadragon_weedy_head_v1', redownload=False
        )
        canvas_key = 1

    # elif OLD and species in ['turtle_green', 'turtle_green+head', 'turtle_hawksbill', 'turtle_hawksbill+head']:
    #     LEGACY = False
    #     species = 'turtle_sea'
    #     confidence_thresh = 0.2
    #     model = models.BackgroundModel(batch_size=batch_size, data_shape=data_shape)
    #     weights_path = grabmodels.ensure_model('background_candidacy_turtle_sea', redownload=False)
    #     canvas_key = 1

    # elif species in ['turtle_green', 'chelonia_mydas']:
    #     LEGACY = False
    #     species = 'turtle_green'
    #     confidence_thresh = 0.2
    #     model = models.BackgroundModel(batch_size=batch_size, data_shape=data_shape)
    #     weights_path = grabmodels.ensure_model('background_turtle_green_v1', redownload=False)
    #     canvas_key = 1
    # elif species in ['turtle_hawksbill', 'eretmochelys_imbricata']:
    #     LEGACY = False
    #     species = 'turtle_hawksbill'
    #     confidence_thresh = 0.2
    #     model = models.BackgroundModel(batch_size=batch_size, data_shape=data_shape)
    #     weights_path = grabmodels.ensure_model('background_turtle_hawksbill_v1', redownload=False)
    #     canvas_key = 1
    # elif species in ['turtle_green+head', 'chelonia_mydas+head']:
    #     LEGACY = False
    #     species = 'turtle_green+head'
    #     confidence_thresh = 0.2
    #     model = models.BackgroundModel(batch_size=batch_size, data_shape=data_shape)
    #     weights_path = grabmodels.ensure_model('background_turtle_green_head_v1', redownload=False)
    #     canvas_key = 1
    # elif species in ['turtle_hawksbill+head', 'eretmochelys_imbricata+head']:
    #     LEGACY = False
    #     species = 'turtle_hawksbill+head'
    #     confidence_thresh = 0.2
    #     model = models.BackgroundModel(batch_size=batch_size, data_shape=data_shape)
    #     weights_path = grabmodels.ensure_model('background_turtle_hawksbill_head_v1', redownload=False)
    #     canvas_key = 1
    elif species in [
        'turtle_sea',
        'chelonioidea',
        'turtle_sea+head',
        'chelonioidea+head',
        'turtle_green',
        'chelonia_mydas',
        'turtle_green+head',
        'chelonia_mydas+head',
        'turtle_hawksbill',
        'eretmochelys_imbricata',
        'turtle_hawksbill+head',
        'eretmochelys_imbricata+head',
        'turtle_oliveridley',
        'lepidochelys_olivacea',
        'turtle_oliveridley+head',
        'lepidochelys_olivacea+head',
        'turtle_asian',
    ]:
        LEGACY = False
        species = 'seaturtle'
        confidence_thresh = 0.2
        model = models.BackgroundModel(batch_size=batch_size, data_shape=data_shape)
        weights_path = grabmodels.ensure_model(
            'background_sea_turtle_v4', redownload=False
        )
        canvas_key = 1
    elif species in [
        'spotted_eagle_ray',
        'aetobatus_narinari',
    ]:
        LEGACY = False
        species = 'spotted_eagle_ray'
        confidence_thresh = 0.2
        model = models.BackgroundModel(batch_size=batch_size, data_shape=data_shape)
        weights_path = grabmodels.ensure_model(
            'background_spotted_eagle_ray_v0', redownload=False
        )
        canvas_key = 1
    elif species in [
        'yellow_bellied_toad',
        'bombina_variegata',
    ]:
        LEGACY = False
        species = 'yellow_bellied_toad'
        confidence_thresh = 0.2
        model = models.BackgroundModel(batch_size=batch_size, data_shape=data_shape)
        weights_path = grabmodels.ensure_model(
            'background_yellow_bellied_toad_v0', redownload=False
        )
        canvas_key = 1
    elif species in ['dolphin_spotted', 'stenella_frontalis']:
        LEGACY = False
        species = 'dolphin_spotted'
        confidence_thresh = 0.2
        model = models.BackgroundModel(batch_size=batch_size, data_shape=data_shape)
        weights_path = grabmodels.ensure_model(
            'background_dolphin_spotted', redownload=False
        )
        canvas_key = 1
    elif species in ['leopard', 'panthera_pardus']:
        LEGACY = False
        species = 'leopard'
        confidence_thresh = 0.2
        model = models.BackgroundModel(batch_size=batch_size, data_shape=data_shape)
        weights_path = grabmodels.ensure_model('background_leopard_v0', redownload=False)
        canvas_key = 1
    elif species in ['snow_leopard', 'panthera_uncia']:
        LEGACY = False
        species = 'snow_leopard'
        confidence_thresh = 0.2
        model = models.BackgroundModel(batch_size=batch_size, data_shape=data_shape)
        weights_path = grabmodels.ensure_model(
            'background_snow_leopard_v0', redownload=False
        )
        canvas_key = 1
    elif species in [
        'wild_dog',
        'wild_dog_dark',
        'wild_dog_light',
        'wild_dog_puppy',
        'wild_dog_standard',
        'wild_dog_tan',
        'lycaon_pictus',
    ]:
        LEGACY = False
        species = 'wild_dog'
        confidence_thresh = 0.2
        model = models.BackgroundModel(batch_size=batch_size, data_shape=data_shape)
        weights_path = grabmodels.ensure_model('background_wilddog_v0', redownload=False)
        canvas_key = 1
    elif species in [
        'dolphin_spotted+dorsal',
        'dolphin_spotted+fin_dorsal',
        'stenella_frontalis+dorsal',
        'stenella_frontalis+fin_dorsal',
        'dolphin_bottlenose+dorsal',
        'dolphin_bottlenose+fin_dorsal',
        'tursiops_truncatus+dorsal',
        'tursiops_truncatus+fin_dorsal',
    ]:
        LEGACY = False
        species = 'dolphin_spotted+fin_dorsal'
        confidence_thresh = 0.2
        model = models.BackgroundModel(batch_size=batch_size, data_shape=data_shape)
        weights_path = grabmodels.ensure_model(
            'background_dolphin_spotted_fin_dorsal', redownload=False
        )
        canvas_key = 1
    elif species in ['whale_humpback+fin_dorsal', 'physeter_macrocephalus+fin_dorsal']:
        LEGACY = False
        species = 'whale_humpback+fin_dorsal'
        confidence_thresh = 0.2
        model = models.BackgroundModel(batch_size=batch_size, data_shape=data_shape)
        weights_path = grabmodels.ensure_model(
            'background_humpback_dorsal', redownload=False
        )
        canvas_key = 1
    elif species in ['gray_whale', 'whale_grey', 'gray_whale+fluke', 'whale_grey+fluke']:
        LEGACY = False
        species = 'whale_grey'
        confidence_thresh = 0.2
        model = models.BackgroundModel(batch_size=batch_size, data_shape=data_shape)
        weights_path = grabmodels.ensure_model(
            'background_whale_grey_v0', redownload=False
        )
        canvas_key = 1
    elif species in ['beluga_whale', 'whale_beluga', 'delphinapterus_leucas']:
        LEGACY = False
        species = 'whale_beluga'
        confidence_thresh = 0.2
        model = models.BackgroundModel(batch_size=batch_size, data_shape=data_shape)
        weights_path = grabmodels.ensure_model(
            'background_whale_beluga_v0', redownload=False
        )
        canvas_key = 1
    elif species in [
        'mediterranean_monk_seal',
        'grey_seal',
        'grey_seal_unknown',
        'grey_seal_male',
        'grey_seal_femaleyoung',
        'grey_seal_pup',
        'harbour_seal',
        'hawaiian_monk_seal',
        'monachus_monachus',
        'phoca_vitulina',
        'halichoerus_grypus',
        'monachus_schauinslandi',
    ]:
        LEGACY = False
        species = 'seals'
        confidence_thresh = 0.2
        model = models.BackgroundModel(batch_size=batch_size, data_shape=data_shape)
        weights_path = grabmodels.ensure_model('background_seals_v0', redownload=False)
        canvas_key = 1
    elif species in [
        'grouper_nassau',
        'grouper_tiger',
        'grouper_nassau_bicolor',
        'epinephelus_striatus',
    ]:
        LEGACY = False
        species = 'grouper_nassau'
        confidence_thresh = 0.2
        model = models.BackgroundModel(batch_size=batch_size, data_shape=data_shape)
        weights_path = grabmodels.ensure_model(
            'background_grouper_nassau_v0', redownload=False
        )
        canvas_key = 1
    elif species in [
        'salamandra_salamandra',
        'salamandra_salamandra_larvae',
        'salanader_fire',
        'salanader_fire_larvae',
    ]:
        LEGACY = False
        species = 'salanader_fire'
        confidence_thresh = 0.2
        model = models.BackgroundModel(batch_size=batch_size, data_shape=data_shape)
        weights_path = grabmodels.ensure_model('salanader_fire_v0', redownload=False)
        canvas_key = 1
    elif species in ['salamandra_salamandra_adult', 'salanader_fire_adult']:
        LEGACY = False
        species = 'salanader_fire_adult'
        confidence_thresh = 0.2
        model = models.BackgroundModel(batch_size=batch_size, data_shape=data_shape)
        weights_path = grabmodels.ensure_model(
            'salanader_fire_adult_v0', redownload=False
        )
        canvas_key = 1
    else:
        raise ValueError('species %r key does not have a trained model' % (species,))

    if LEGACY:
        old_weights_fpath = weights_path
        model.load_old_weights_kw2(old_weights_fpath)
    else:
        model_state_fpath = model.get_model_state_fpath(fpath=weights_path)
        logger.info('[model] loading model state from: %s' % (model_state_fpath,))
        model_state = ut.load_cPkl(model_state_fpath)

        model.output_dims = model_state['output_dims']
        model.data_params = model_state['data_params']
        model._fix_center_mean_std()

        model.best_results = model_state['best_results']

        model.init_arch()
        model.batch_size = 128
        model.data_params['center_mean'] = np.mean(model.data_params['center_mean'])
        model.data_params['center_std'] = np.mean(model.data_params['center_std'])
        model.hyperparams['whiten_on'] = True
        model.set_all_param_values(model.best_results['weights'])

    # Create the Theano primitives
    # create theano symbolic expressions that define the network
    logger.info('\n[wbia_cnn] --- COMPILING SYMBOLIC THEANO FUNCTIONS ---')
    logger.info('[model] creating Theano primitives...')
    model.build_predict_func()

    logger.info('[wbia_cnn] Performing inference...')

    _iter = ut.ProgressIter(
        chip_list,
        nTotal=nInput,
        lbl=species + ' fgdetect',
        adjust=True,
        freq=10,
        time_thresh=30.0,
    )
    for chip in _iter:
        try:
            if LEGACY:
                samples, canvas_dict = test_convolutional(
                    model, chip, padding=24, confidence_thresh=confidence_thresh
                )
            else:
                samples, canvas_dict = test_convolutional(
                    model, chip, padding=25, confidence_thresh=confidence_thresh
                )
            if NEW and LEGACY:
                mask = np.maximum(255 - canvas_dict['negative'], canvas_dict[canvas_key])
            else:
                mask = canvas_dict[canvas_key]
        except Exception as ex:
            ut.printex(
                ex,
                ('Error running convnet with ' 'chip.shape=%r, chip.dtype=%r')
                % (chip.shape, chip.dtype),
            )
            raise
        yield mask


def test_convolutional(
    model,
    image,
    patch_size='auto',
    stride='auto',
    padding=32,
    batch_size=None,
    verbose=False,
    confidence_thresh=0.5,
    **kwargs
):
    """Using a network, test an entire image full convolutionally

    This function will test an entire image full convolutionally (or a close
    approximation of full convolutionally).  The CUDA framework and driver is a
    limiting factor for how large an image can be given to a network for full
    convolutional inference.  As a result, we implement a non-overlapping (or
    little overlapping) patch extraction approximation that processes the entire
    image within a single batch or very few batches.  This is an extremely
    efficient process for processing an image with a CNN.

    The patches are given a slight overlap in order to smooth the effects of
    boundary conditions, which are seen on every patch.  We also mirror the
    border of each patch and add an additional amount of padding to cater to the
    architecture's receptive field reduction.

    See :func:`utils.extract_patches_stride` for patch extraction behavior.

    Args:
        model (Model): the network to use to perform feedforward inference
        image (numpy.ndarray): the image passed in to make a coreresponding
            sized dictionarf of response maps
        patch_size (int, tuple of int, optional): the size of the patches
            extracted across the image, passed in as a 2-tuple of (width,
            height).  Defaults to (200, 200).
        stride (int, tuple of int, optional): the stride of the patches
            extracted across the image.  Defaults to [patch_size - padding].
        padding (int, optional): the mirrored padding added to every patch
            during testing, which can be used to offset the effects of the
            receptive field reduction in the network.  Defaults to 32.
        **kwargs: arbitrary keyword arguments, passed to
            :func:`model.test()`

    Returns:
        samples, canvas_dict (tuple of int and dict): the number of total
            samples used to generate the response map and the actual response
            maps themselves as a dictionary.  The dictionary uses the class
            labels as the strings and the numpy array image as the values.
    """
    from wbia_cnn import utils
    import cv2

    def _add_pad(data_):
        if len(data_.shape) == 2:
            data_padded = np.pad(data_, padding, 'reflect', reflect_type='even')
        else:
            h, w, c = data_.shape
            data_padded = np.dstack(
                [
                    np.pad(data_[:, :, _], padding, 'reflect', reflect_type='even')
                    for _ in range(c)
                ]
            )
        return data_padded

    def _resize_target(image, target_height=None, target_width=None):
        import cv2

        assert target_height is not None or target_width is not None
        height, width = image.shape[:2]
        if target_height is not None and target_width is not None:
            h = target_height
            w = target_width
        elif target_height is not None:
            h = target_height
            w = (width / height) * h
        elif target_width is not None:
            w = target_width
            h = (height / width) * w
        w, h = int(w), int(h)
        return cv2.resize(image, (w, h), interpolation=cv2.INTER_LANCZOS4)

    if verbose:
        # Start timer
        tt = ut.tic()
        logger.info('[harness] Loading the testing data (convolutional)...')
    # Try to get the image's shape
    h, w = image.shape[:2]

    original_shape = None
    if h < w and h < 256:
        original_shape = image.shape
        image = _resize_target(image, target_height=256)
    if w < h and w < 256:
        original_shape = image.shape
        image = _resize_target(image, target_width=256)

    h, w = image.shape[:2]

    # GLOBAL_LIMIT = min(256, w, h)
    # HACK, this only works for square data shapes
    GLOBAL_LIMIT = model.data_shape[0]
    # Inference
    if patch_size == 'auto':
        patch_size = (GLOBAL_LIMIT - 2 * padding, GLOBAL_LIMIT - 2 * padding)
    if stride == 'auto':
        psx, psy = patch_size
        stride = (psx - padding, psy - padding)
    _tup = utils.extract_patches_stride(image, patch_size, stride)
    data_list, coord_list = _tup
    samples = len(data_list)
    if batch_size is None:
        batch_size = samples
    start = 0
    label_list = []
    confidence_list = []

    theano_predict = model.build_predict_func()
    while start < samples:
        end = min(samples, start + batch_size)
        data_list_segment = data_list[start:end]
        # coord_list_segment = coord_list[start: end]

        # Augment the data_list by adding a reflected pad
        data_list_ = np.array([_add_pad(data_) for data_ in data_list_segment])

        # batchiter_kw = dict(
        #    fix_output=False,
        #    showprog=True,
        #    spatial=True
        # )
        # test_results = model._predict(data_list_)
        test_results = model.process_batch(theano_predict, data_list_, unwrap=False)
        # test_results2 = batch.process_batch(model, data_list_, None,
        #                                   theano_predict, **batchiter_kw)
        # label_list.extend(test_results['labeled_predictions'])
        if model.encoder is not None:
            labeled_predictions = model.encoder.inverse_transform(
                test_results['predictions']
            )
        else:
            labeled_predictions = test_results['predictions']
        label_list.extend(labeled_predictions)
        confidence_list.extend(test_results['confidences'])
        start += batch_size

    # Get all of the labels for the data, inheritted from the model
    if model.encoder is not None:
        # python2 backwards compatibility
        if isinstance(model.encoder.classes_, np.ndarray):
            label_list_ = model.encoder.classes_.tolist()
        else:
            label_list_ = list(model.encoder.classes_)
        label_list_ = list(
            map(
                lambda x: x if isinstance(x, six.text_type) else x.decode('utf-8'),
                label_list_,
            )
        )
    else:
        label_list_ = list(range(model.output_dims))
    # Create a dictionary of canvases
    canvas_dict = {}
    for label in label_list_:
        canvas_dict[label] = np.zeros((h, w))  # We want float precision
    # Construct the canvases using the forward inference results
    label_list_ = label_list_[::-1]
    # logger.info('[harness] Labels: %r' %(label_list_, ))
    zipped = list(zip(data_list, coord_list, label_list, confidence_list))
    for label in label_list_:
        for data, coord, label_, confidence in zipped:
            x1, y1, x2, y2 = coord
            # Get label and apply to confidence
            confidence_ = np.copy(confidence)

            # OLD
            # confidence_[label_ != label] = 0

            # NEW
            # if isinstance(label_, np.ndarray):
            #     flip_index = (label_ != label).astype(np.int)
            # else:
            #     flip_index = int(label_ != label)
            if isinstance(label, six.text_type):
                # fix for python3, can't compare numpy byte arrays with
                # unicode.
                label2_ = label.encode('utf-8')
            else:
                label2_ = label
            flip_index = label_ != label2_
            confidence_[flip_index] = 1.0 - confidence_[flip_index]
            confidence_[confidence_ <= confidence_thresh] = 0

            confidence_ *= 255.0

            # Blow up canvas
            mask = cv2.resize(confidence_, data.shape[0:2])
            # Get the current values
            current = canvas_dict[label][y1:y2, x1:x2]
            # Where the current canvas is zero (most of it), make it mask
            flags = current == 0
            current[flags] = mask[flags]
            # Average the current with the mask, which address overlapping areas
            mask = 0.5 * mask + 0.5 * current
            # Aggregate
            canvas_dict[label][y1:y2, x1:x2] = mask
        # Blur
        # FIXME: Should this postprocessing step applied here?
        # There is postprocessing in ibeis/algos/preproc/preproc_probchip.py
        ksize = 3
        kernel = (ksize, ksize)
        canvas_dict[label] = cv2.blur(canvas_dict[label], kernel)
    # Cast all images to uint8
    for label in label_list_:
        canvas = np.around(canvas_dict[label])
        canvas = canvas.astype(np.uint8)
        if original_shape is not None:
            canvas = _resize_target(
                canvas, target_height=original_shape[0], target_width=original_shape[1]
            )
        canvas_dict[label] = canvas
    if verbose:
        # End timer
        duration = ut.toc(tt, verbose=False)
        logger.info('[harness] Interface took %s seconds...' % (duration,))
    # Return the canvas dict
    return samples, canvas_dict


@register_ibs_method
def fix_annot_species_viewpoint_quality_cnn(ibs, aid_list, min_conf=0.8):
    r"""
    Args:
        ibs (IBEISController):  ibeis controller object
        aid_list (int):  list of annotation ids
    """
    import cv2

    # Load chips and resize to the target
    data_shape = (96, 96, 3)
    # Define model and load weights
    logger.info('Loading model...')
    batch_size = int(min(128, 2 ** np.floor(np.log2(len(aid_list)))))
    model = models.ViewpointModel(batch_size=batch_size, data_shape=data_shape)
    weights_path = grabmodels.ensure_model('viewpoint', redownload=False)
    old_weights_fpath = weights_path
    model.load_old_weights_kw(old_weights_fpath)
    # Read the data
    target = data_shape[0:2]
    logger.info('Loading chips...')
    chip_list = ibs.get_annot_chips(aid_list, verbose=True)
    logger.info('Resizing chips...')
    chip_list_resized = [
        cv2.resize(chip, target, interpolation=cv2.INTER_LANCZOS4)
        for chip in ut.ProgressIter(chip_list, lbl='resizing chips')
    ]
    # Build data for network
    X_test = np.array(chip_list_resized, dtype=np.uint8)
    # Predict on the data and convert labels to IBEIS namespace
    test_outputs = model.predict2(X_test)
    label_list = test_outputs['labeled_predictions']
    conf_list = test_outputs['confidences']
    species_viewpoint_list = [convert_label(label) for label in label_list]
    zipped = zip(aid_list, species_viewpoint_list, conf_list)
    skipped_list = []
    for aid, (species, viewpoint), conf in zipped:
        if conf >= min_conf:
            species_ = species
            viewpoint_ = viewpoint
            quality_ = const.QUAL_GOOD
        else:
            skipped_list.append(aid)
            species_ = const.UNKNOWN
            viewpoint_ = None
            quality_ = const.QUAL_UNKNOWN

        ibs.set_annot_species([aid], [species_])
        ibs.set_annot_yaw_texts([aid], [viewpoint_])
        ibs.set_annot_quality_texts([aid], [quality_])

    return skipped_list


@register_ibs_method
def detect_annot_species_viewpoint_cnn(ibs, aid_list):
    r"""
    Args:
        ibs (IBEISController):  ibeis controller object
        aid_list (int):  list of annotation ids

    Returns:
        list: species_viewpoint_list

    CommandLine:
        python -m wbia_cnn._plugin --exec-detect_annot_species_viewpoint_cnn

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia_cnn._plugin import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb(defaultdb='testdb1', allow_newdir=True)
        >>> aid_list = ibs.get_valid_aids()
        >>> species_viewpoint_list = detect_annot_species_viewpoint_cnn(ibs, aid_list)
        >>> result = ('species_viewpoint_list = %s' % (str(species_viewpoint_list),))
        >>> print(result)
    """
    import cv2

    # Load chips and resize to the target
    data_shape = (96, 96, 3)
    # Define model and load weights
    logger.info('Loading model...')
    batch_size = int(min(128, 2 ** np.floor(np.log2(len(aid_list)))))
    model = models.ViewpointModel(batch_size=batch_size, data_shape=data_shape)
    weights_path = grabmodels.ensure_model('viewpoint', redownload=False)
    old_weights_fpath = weights_path
    model.load_old_weights_kw(old_weights_fpath)
    # Read the data
    target = data_shape[0:2]
    logger.info('Loading chips...')
    chip_list = ibs.get_annot_chips(aid_list, verbose=True)
    logger.info('Resizing chips...')
    chip_list_resized = [
        cv2.resize(chip, target, interpolation=cv2.INTER_LANCZOS4)
        for chip in ut.ProgressIter(chip_list, lbl='resizing chips')
    ]
    # Build data for network
    X_test = np.array(chip_list_resized, dtype=np.uint8)
    # Predict on the data and convert labels to IBEIS namespace
    test_outputs = model.predict2(X_test)
    label_list = test_outputs['labeled_predictions']
    species_viewpoint_list = [convert_label(label) for label in label_list]
    # pred_list, label_list, conf_list = test.test_data(X_test, y_test, model, weights_path)
    # species_viewpoint_list = [ convert_label(label) for label in label_list ]
    return species_viewpoint_list


@register_ibs_method
def validate_annot_species_viewpoint_cnn(ibs, aid_list, verbose=False):
    r"""
    Args:
        ibs (IBEISController):  ibeis controller object
        aid_list (int):  list of annotation ids
        verbose (bool):  verbosity flag(default = False)

    Returns:
        tuple: (bad_species_list, bad_viewpoint_list)
    """
    # Load chips and metadata
    species_list = ibs.get_annot_species(aid_list)
    viewpoint_list = ibs.get_annot_yaw_texts(aid_list)
    species_viewpoint_list = ibs.detect_annot_species_viewpoint_cnn(aid_list)
    # Find all bad
    bad_species_list = []
    bad_viewpoint_list = []
    data = zip(aid_list, species_list, viewpoint_list, species_viewpoint_list)
    for aid, species, viewpoint, (species_, viewpoint_) in data:
        if species != species_:
            bad_species_list.append((aid, species, species_))
            continue
        if viewpoint != viewpoint_:
            bad_viewpoint_list.append((aid, species, viewpoint, viewpoint_))
            continue
    # Print bad if verbose
    if verbose:
        logger.info('Found conflicting species:')
        for bad_species in bad_species_list:
            logger.info('    AID %4d (%r) should be %r' % bad_species)
        logger.info('Found conflicting viewpoints:')
        for bad_viewpoint in bad_viewpoint_list:
            logger.info('    AID %4d (%r, %r) should be %r' % bad_viewpoint)
    # Return bad
    return bad_species_list, bad_viewpoint_list


def _suggest_random_candidate_regions(ibs, image, min_size, num_candidates=2000):
    h, w, c = image.shape
    h -= 1
    w -= 1
    min_x, min_y = min_size

    def _candidate():
        x0, y0, x1, y1 = 0, 0, 0, 0
        while x1 - x0 < min_x or y1 - y0 < min_y:
            x0 = int(random.uniform(0, w))
            y0 = int(random.uniform(0, h))
            x1 = int(random.uniform(0, w))
            y1 = int(random.uniform(0, h))
            if x0 > x1:
                x0, x1 = x1, x0
            if y0 > y1:
                y0, y1 = y1, y0
        return x0, y0, x1, y1

    candidate_list = [_candidate() for _ in range(num_candidates)]
    return candidate_list


def _suggest_bing_candidate_regions(ibs, image_path_list):
    def _dedictify(dict_list):
        return [[d_['minx'], d_['miny'], d_['maxx'], d_['maxy']] for d_ in dict_list]

    from pybing import BING_Detector

    detector = BING_Detector()
    results_list = detector.detect(image_path_list)
    result_list = [_dedictify(results[1]) for results in results_list]
    return result_list


def non_max_suppression_fast(box_list, conf_list, overlapThresh=0.5):
    """
    Python version of Malisiewicz's Matlab code:
    https://github.com/quantombone/exemplarsvm

    NOTE: This is adapted from Pedro Felzenszwalb's version (nms.m),
    but an inner loop has been eliminated to significantly speed it
    up in the case of a large number of boxes

    Reference: https://github.com/rbgirshick/rcnn/blob/master/nms/nms.m
    Reference: http://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
    """
    # if there are no boxes, return an empty list
    if len(box_list) == 0:
        return []

    # Convert to Numpy
    box_list = np.array(box_list)
    conf_list = np.array(conf_list)

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if box_list.dtype.kind == 'i':
        box_list = box_list.astype('float')

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    # Our boxes are stored as y1, y2, x1, x2 to be in-line with OpenCV indexing
    x1 = box_list[:, 0]
    y1 = box_list[:, 1]
    x2 = box_list[:, 2]
    y2 = box_list[:, 3]
    s = conf_list

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(s)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(
            idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0]))
        )

    # return only the bounding boxes that were picked using the
    # integer data type
    return pick


@register_ibs_method
def detect_image_cnn(ibs, gid, confidence=0.90, extraction='bing'):
    r"""
    Args:
        ibs (IBEISController):  ibeis controller object
        gid (?):
        confidence (float): (default = 0.9)
        extraction (str): (default = 'bing')

    CommandLine:
        python -m wbia_cnn._plugin --exec-detect_image_cnn

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia_cnn._plugin import *  # NOQA
        >>> from wbia_cnn._plugin import _suggest_random_candidate_regions, _suggest_bing_candidate_regions  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb(defaultdb='testdb1', allow_newdir=True)
        >>> gid = 1
        >>> confidence = 0.9
        >>> extraction = 'bing'
        >>> result = detect_image_cnn(ibs, gid, confidence, extraction)
        >>> print(result)
    """
    import cv2

    # Load chips and resize to the target
    target = (96, 96)
    targetx, targety = target
    # gid = gid_list[random.randint(0, len(gid_list))]
    # gid = gid_list[0]
    logger.info('Detecting with gid=%r...' % (gid,))
    image = ibs.get_images(gid)
    rects = np.copy(image)
    h, w, c = image.shape

    logger.info('Querrying for candidate regions...')
    image_path = ibs.get_image_paths(gid)
    if extraction == 'random':
        candidate_list = _suggest_random_candidate_regions(ibs, image, (32, 32))
    else:
        candidate_list = _suggest_bing_candidate_regions(ibs, [image_path])[0]

    logger.info('Num candidates: %r' % (len(candidate_list),))
    chip_list_resized = []
    logger.info('Extracting candidate regions...')
    for candidate in candidate_list:
        x0, y0, x1, y1 = candidate
        chip = image[y0:y1, x0:x1]
        chip = cv2.resize(chip, target, interpolation=cv2.INTER_LANCZOS4)
        chip_list_resized.append(chip)
        color = (255, 0, 0)
        # cv2.rectangle(rects, (x0, y0), (x1, y1), color)
        mx = int((x1 - x0) * 0.5)
        my = int((y1 - y0) * 0.5)
        cv2.circle(rects, (x0 + mx, y0 + my), 5, color, -1)
    # cv2.imshow('', rects)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Build data for network
    X_test = np.array(chip_list_resized, dtype=np.uint8)
    # Define model and load weights
    logger.info('Loading model...')
    data_shape = (96, 96, 3)
    # Define model and load weights
    logger.info('Loading model...')
    # batch_size = int(min(128, 2 ** np.floor(np.log2(len(chip_list_resized)))))
    batch_size = None
    model = models.ViewpointModel(batch_size=batch_size, data_shape=data_shape)
    weights_path = grabmodels.ensure_model('viewpoint', redownload=False)
    old_weights_fpath = weights_path
    model.load_old_weights_kw(old_weights_fpath)

    # Predict on the data and convert labels to IBEIS namespace
    test_outputs = model.predict2(X_test)
    conf_list = test_outputs['confidences']
    label_list = test_outputs['labeled_predictions']
    pred_list = test_outputs['predictions']
    # pred_list, label_list, conf_list = test.test_data(X_test, y_test, model, weights_path)
    species_viewpoint_list = [convert_label(label) for label in label_list]

    num_all_candidates = len(conf_list)
    index_list = non_max_suppression_fast(candidate_list, conf_list)
    logger.info('Surviving candidates: %r' % (index_list,))
    num_supressed_candidates = num_all_candidates - len(index_list)
    logger.info('Supressed: %d candidates' % (num_supressed_candidates,))

    candidate_list = np.take(candidate_list, index_list, axis=0)
    pred_list = np.take(pred_list, index_list, axis=0)
    species_viewpoint_list = np.take(species_viewpoint_list, index_list, axis=0)
    conf_list = np.take(conf_list, index_list, axis=0)

    values = zip(candidate_list, pred_list, species_viewpoint_list, conf_list)
    rects = np.copy(image)
    color_dict = {
        'giraffe': (255, 0, 0),
        'giraffe_masai': (255, 255, 0),
        'zebra_plains': (0, 0, 255),
        'zebra_grevys': (0, 255, 0),
        'elephant_savanna': (0, 0, 0),
    }
    skipped = 0
    for candidate, pred, species_viewpoint, conf in values:
        x0, y0, x1, y1 = tuple(candidate)
        species, viewpoint = species_viewpoint
        if conf < confidence:
            skipped += 1
            continue
        logger.info(
            '%r Found %s (%s, %s) at %s'
            % (
                candidate,
                pred,
                species,
                viewpoint,
                conf,
            )
        )
        color = color_dict[species]
        cv2.rectangle(rects, (x0, y0), (x1, y1), color)
        # mx = int((x1 - x0) * 0.5)
        # my = int((y1 - y0) * 0.5)
        # cv2.circle(rects, (x0 + mx, y0 + my), 5, color, -1)
    logger.info(
        'Skipped [ %d / %d ]'
        % (
            skipped,
            len(values),
        )
    )

    cv2.imshow('', rects)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_siam_l2_model():
    """
    model.show_weights_image()
    """
    model_url = (
        'https://wildbookiarepository.azureedge.net/models/siaml2_128_model_state.pkl'
    )
    model_dpath = ut.ensure_app_resource_dir('wbia_cnn', 'models')
    model_fpath = ut.grab_file_url(model_url, download_dir=model_dpath)
    model_state = ut.load_cPkl(model_fpath)
    import wbia_cnn

    wbia_cnn.models
    model = models.SiameseL2(
        input_shape=model_state['input_shape'],
        arch_tag=model_state['arch_tag'],
        autoinit=True,
    )
    model.load_model_state(fpath=model_fpath)
    return model


def generate_siam_l2_128_feats(ibs, cid_list, config2_=None):
    r"""
    Args:
        ibs (IBEISController):  ibeis controller object
        cid_list (list):
        config2_ (dict): (default = None)

    CommandLine:
        python -m wbia_cnn._plugin --test-generate_siam_l2_128_feats
        python -m wbia_cnn._plugin --test-generate_siam_l2_128_feats --db PZ_Master0

    SeeAlso:
        ~/code/ibeis/ibeis/algo/preproc/preproc_feat.py

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia_cnn._plugin import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb(defaultdb='testdb1', allow_newdir=True)
        >>> cid_list = ibs.depc_annot.get_rowids('chips', ibs.get_valid_aids())
        >>> config2_ = None
        >>> # megahack
        >>> cfg = Config.FeatureConfig() # fixme
        >>> config2_ = dict(feat_type='hesaff+siam128',
        >>>                 feat_cfgstr=cfg.feat_cfg.get_cfgstr().replace('sift', 'siam128'),
        >>>                 hesaff_params=cfg.feat_cfg.get_hesaff_params())
        >>> featgen = generate_siam_l2_128_feats(ibs, cid_list, config2_)
        >>> result = ut.depth_profile(list(featgen))
        >>> print(result)
    """
    # hack because we need the old features
    import vtool as vt

    model = get_siam_l2_model()
    colorspace = 'gray' if model.input_shape[1] else None  # 'bgr'
    patch_size = model.input_shape[-1]
    if config2_ is not None:
        # Get config from config2_ object
        # logger.info('id(config2_) = ' + str(id(config2_)))
        feat_cfgstr = config2_.get('feat_cfgstr')
        hesaff_params = config2_.get('hesaff_params')
        assert feat_cfgstr is not None
        assert hesaff_params is not None
    else:
        assert False
    hack_config2_ = dict(
        feat_type='hesaff+sift',
        feat_cfgstr=feat_cfgstr.replace('siam128', 'sift'),
        hesaff_params=hesaff_params,
    )
    logger.info('Generating siam128 features for %d chips' % (len(cid_list),))
    BATCHED = True
    if BATCHED:
        ibs.get_chip_feat_rowid(cid_list, config2_=hack_config2_, ensure=True)
        for cid_batch in ut.ProgressIter(
            list(ut.ichunks(cid_list, 128)), lbl='siam128 chip chunk'
        ):
            sift_fid_list = ibs.get_chip_feat_rowid(cid_batch, config2_=hack_config2_)
            logger.info('Reading keypoints')
            kpts_list = ibs.get_feat_kpts(sift_fid_list)
            logger.info('Reading chips')
            chip_list = vt.convert_image_list_colorspace(
                ibs.get_chips(cid_batch, ensure=True), colorspace
            )
            logger.info('Warping patches')
            warped_patches_list = [
                vt.get_warped_patches(chip, kpts, patch_size=patch_size)[0]
                for chip, kpts in zip(chip_list, kpts_list)
            ]
            flat_list, cumlen_list = ut.invertible_flatten2(warped_patches_list)
            stacked_patches = np.transpose(np.array(flat_list)[None, :], (1, 2, 3, 0))

            test_outputs = model.predict2(X_test=stacked_patches)
            network_output_determ = test_outputs['network_output_determ']
            # network_output_determ.min()
            # network_output_determ.max()
            siam128_vecs_list = ut.unflatten2(network_output_determ, cumlen_list)

            for cid, kpts, vecs in zip(cid_batch, kpts_list, siam128_vecs_list):
                yield cid, len(kpts), kpts, vecs
    else:
        sift_fid_list = ibs.get_chip_feat_rowid(
            cid_list, config2_=hack_config2_, ensure=True
        )  # NOQA
        logger.info('Reading keypoints')
        kpts_list = ibs.get_feat_kpts(sift_fid_list)
        logger.info('Reading chips')
        chip_list = vt.convert_image_list_colorspace(
            ibs.get_chips(cid_list, ensure=True), colorspace
        )
        logger.info('Warping patches')
        warped_patches_list = [
            vt.get_warped_patches(chip, kpts, patch_size=patch_size)[0]
            for chip, kpts in zip(chip_list, kpts_list)
        ]
        flat_list, cumlen_list = ut.invertible_flatten2(warped_patches_list)
        stacked_patches = np.transpose(np.array(flat_list)[None, :], (1, 2, 3, 0))

        test_outputs = model.predict2(X_test=stacked_patches)
        network_output_determ = test_outputs['network_output_determ']
        # network_output_determ.min()
        # network_output_determ.max()
        siam128_vecs_list = ut.unflatten2(network_output_determ, cumlen_list)

        for cid, kpts, vecs in zip(cid_list, kpts_list, siam128_vecs_list):
            yield cid, len(kpts), kpts, vecs


def extract_siam128_vecs(chip_list, kpts_list):
    """
    Duplicate testing func for vtool
    """
    import vtool as vt

    model = get_siam_l2_model()
    colorspace = 'gray' if model.input_shape[1] else None  # 'bgr'
    patch_size = model.input_shape[-1]
    chip_list_ = vt.convert_image_list_colorspace(chip_list, colorspace)

    warped_patches_list = [
        vt.get_warped_patches(chip, kpts, patch_size=patch_size)[0]
        for chip, kpts in zip(chip_list_, kpts_list)
    ]
    flat_list, cumlen_list = ut.invertible_flatten2(warped_patches_list)
    stacked_patches = np.transpose(np.array(flat_list)[None, :], (1, 2, 3, 0))
    X_test = stacked_patches
    test_outputs = model.predict2(X_test)
    network_output_determ = test_outputs['network_output_determ']
    # network_output_determ.min()
    # network_output_determ.max()
    siam128_vecs_list = ut.unflatten2(network_output_determ, cumlen_list)
    return siam128_vecs_list


if __name__ == '__main__':
    """
    CommandLine:
        python -m wbia_cnn._plugin
        python -m wbia_cnn._plugin --allexamples
        python -m wbia_cnn._plugin --allexamples --noface --nosrc
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
