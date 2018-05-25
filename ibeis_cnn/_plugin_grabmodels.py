# -*- coding                                        :utf-8 -*-
from __future__ import absolute_import, division, print_function
import utool as ut
(print, rrr, profile) = ut.inject2(
    __name__, '[ibeis_cnn._plugin_grabmodels]')


#DEFAULT_CNNMODELS_DIR = ut.get_app_resource_dir('ibeis_cnn', 'pretrained')

MODEL_DOMAIN = 'https://lev.cs.rpi.edu/public/models/'
MODEL_URLS = {
    'classifier_cameratrap_megan_v1'           : 'classifier.cameratrap.megan.v1.pkl',
    'classifier_cameratrap_megan_v2'           : 'classifier.cameratrap.megan.v2.pkl',
    'classifier_cameratrap_megan_v3'           : 'classifier.cameratrap.megan.v3.pkl',
    'classifier_cameratrap_megan_v4'           : 'classifier.cameratrap.megan.v4.pkl',
    'classifier_cameratrap_megan_v5'           : 'classifier.cameratrap.megan.v5.pkl',
    'classifier_cameratrap_megan_v6'           : 'classifier.cameratrap.megan.v6.pkl',

    'classifier_cameratrap_megan2_v1'          : 'classifier.cameratrap.megan2.v1.pkl',
    'classifier_cameratrap_megan2_v2'          : 'classifier.cameratrap.megan2.v2.pkl',
    'classifier_cameratrap_megan2_v3'          : 'classifier.cameratrap.megan2.v3.pkl',
    'classifier_cameratrap_megan2_v4'          : 'classifier.cameratrap.megan2.v4.pkl',
    'classifier_cameratrap_megan2_v5'          : 'classifier.cameratrap.megan2.v5.pkl',

    'classifier_coco_zebra'                    : 'classifier.coco.zebra.pkl',
    'classifier_v3_zebra'                      : 'classifier.29.zebra.pkl',

    'classifier2_v3'                           : 'classifier2.29.pkl',
    'classifier2_candidacy'                    : 'classifier2.candidacy.pkl',
    'classifier2_ggr2'                         : 'classifier2.ggr2.pkl',

    'labeler_v1'                               : 'labeler.v1.pkl',
    'labeler_cheetah'                          : 'labeler.cheetah.pkl',
    'labeler_lynx'                             : 'labeler.lynx.pkl',
    'labeler_v3'                               : 'labeler.29.pkl',
    'labeler_candidacy'                        : 'labeler.candidacy.pkl',

    'background_giraffe_masai'                 : 'background.giraffe_masai.npy',
    'background_zebra_plains'                  : 'background.zebra_plains.npy',
    'background_zebra_plains_grevys'           : 'background.zebra_plains_grevys.npy',
    'background_whale_fluke'                   : 'background.whale_fluke.npy',
    'background_lynx'                          : 'background.lynx.pkl',
    'background_lynx_v2'                       : 'background.lynx_v2.pkl',
    'background_cheetah'                       : 'background.cheetah.pkl',

    'background_candidacy_giraffe_masai'       : 'background.candidacy.giraffe_masai.pkl',
    'background_candidacy_giraffe_reticulated' : 'background.candidacy.giraffe_reticulated.pkl',
    'background_candidacy_turtle_sea'          : 'background.candidacy.turtle_sea.pkl',
    'background_candidacy_whale_fluke'         : 'background.candidacy.whale_fluke.pkl',
    'background_candidacy_zebra_grevys'        : 'background.candidacy.zebra_grevys.pkl',
    'background_candidacy_zebra_plains'        : 'background.candidacy.zebra_plains.pkl',

    'aoi2_candidacy'                           : 'aoi2.candidacy.pkl',

    'viewpoint'                                : 'viewpoint.v1.pkl',

    'caffenet'                                 : 'pretrained.caffe.caffenet.slice_0_6_None.pkl',
    'caffenet_conv'                            : 'pretrained.caffe.caffenet.slice_0_10_None.pkl',
    'caffenet_full'                            : 'pretrained.caffe.caffenet.pkl',
    'vggnet'                                   : 'pretrained.caffe.vgg.slice_0_6_None.pkl',
    'vggnet_conv'                              : 'pretrained.caffe.vgg.slice_0_32_None.pkl',
    'vggnet_full'                              : 'pretrained.caffe.vgg.pkl',
}


def ensure_model(model, redownload=False):
    try:
        url = MODEL_DOMAIN + MODEL_URLS[model]
        extracted_fpath = ut.grab_file_url(url, appname='ibeis_cnn',
                                           redownload=redownload,
                                           check_hash=True)
    except KeyError as ex:
        ut.printex(ex, 'model is not uploaded', iswarning=True)
        extracted_fpath = ut.unixjoin(ut.get_app_resource_dir('ibeis_cnn'), model)
        ut.assert_exists(extracted_fpath)
    return extracted_fpath


if __name__ == '__main__':
    """

    CommandLine:
        python -m ibeis_cnn._plugin_grabmodels.ensure_models
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
