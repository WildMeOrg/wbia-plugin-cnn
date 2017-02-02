# -*- coding                                        :utf-8 -*-
from __future__ import absolute_import, division, print_function
import utool as ut
(print, rrr, profile) = ut.inject2(
    __name__, '[ibeis_cnn._plugin_grabmodels]')


#DEFAULT_CNNMODELS_DIR = ut.get_app_resource_dir('ibeis_cnn', 'pretrained')

MODEL_DOMAIN = 'https://lev.cs.rpi.edu/public/models/'
MODEL_URLS = {
    'classifier_coco_zebra'          : 'classifier.coco.zebra.pkl',

    'labeler_v1'                     : 'labeler.cheetah.pkl',
    'labeler_cheetah'                : 'labeler.cheetah.pkl',
    'labeler_lynx'                   : 'labeler.lynx.pkl',

    'background_giraffe_masai'       : 'background.giraffe_masai.npy',
    'background_zebra_plains'        : 'background.zebra_plains.npy',
    'background_zebra_plains_grevys' : 'background.zebra_plains_grevys.npy',
    'background_whale_fluke'         : 'background.whale_fluke.npy',
    'background_lynx'                : 'background.lynx.pkl',
    'background_lynx_v2'             : 'background.lynx_v2.pkl',
    'background_cheetah'             : 'background.cheetah.pkl',

    'viewpoint'                      : 'viewpoint.v1.pkl',

    'caffenet'                       : 'pretrained.caffe.caffenet.slice_0_6_None.pkl',
    'caffenet_conv'                  : 'pretrained.caffe.caffenet.slice_0_10_None.pkl',
    'caffenet_full'                  : 'pretrained.caffe.caffenet.pkl',
    'vggnet'                         : 'pretrained.caffe.vgg.slice_0_6_None.pkl',
    'vggnet_conv'                    : 'pretrained.caffe.vgg.slice_0_32_None.pkl',
    'vggnet_full'                    : 'pretrained.caffe.vgg.pkl',
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
