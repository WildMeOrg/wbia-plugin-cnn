# -*- coding: utf-8 -*-
# -*- coding                                        :utf-8 -*-
import logging
import utool as ut

(print, rrr, profile) = ut.inject2(__name__, '[wbia_cnn._plugin_grabmodels]')
logger = logging.getLogger()


# DEFAULT_CNNMODELS_DIR = ut.get_app_resource_dir('wbia_cnn', 'pretrained')

MODEL_DOMAIN = 'https://wildbookiarepository.azureedge.net/models/'
MODEL_URLS = {
    'classifier_cameratrap_megan_v1': 'classifier.cameratrap.megan.v1.pkl',
    'classifier_cameratrap_megan_v2': 'classifier.cameratrap.megan.v2.pkl',
    'classifier_cameratrap_megan_v3': 'classifier.cameratrap.megan.v3.pkl',
    'classifier_cameratrap_megan_v4': 'classifier.cameratrap.megan.v4.pkl',
    'classifier_cameratrap_megan_v5': 'classifier.cameratrap.megan.v5.pkl',
    'classifier_cameratrap_megan_v6': 'classifier.cameratrap.megan.v6.pkl',
    'classifier_cameratrap_megan2_v1': 'classifier.cameratrap.megan2.v1.pkl',
    'classifier_cameratrap_megan2_v2': 'classifier.cameratrap.megan2.v2.pkl',
    'classifier_cameratrap_megan2_v3': 'classifier.cameratrap.megan2.v3.pkl',
    'classifier_cameratrap_megan2_v4': 'classifier.cameratrap.megan2.v4.pkl',
    'classifier_cameratrap_megan2_v5': 'classifier.cameratrap.megan2.v5.pkl',
    'classifier_cameratrap_megan2_v6': 'classifier.cameratrap.megan2.v6.pkl',
    'classifier_cameratrap_ryan_cnn_v1': 'classifier.cameratrap.ryan.wbia_cnn.v1.pkl',
    'classifier_coco_zebra': 'classifier.coco.zebra.pkl',
    'classifier_v3_zebra': 'classifier.29.zebra.pkl',
    'classifier2_v3': 'classifier2.29.pkl',
    'classifier2_candidacy': 'classifier2.candidacy.pkl',
    'classifier2_ggr2': 'classifier2.ggr2.pkl',
    'labeler_v1': 'labeler.v1.pkl',
    'labeler_v3': 'labeler.29.pkl',
    'labeler_candidacy': 'labeler.candidacy.pkl',
    'labeler_cheetah_v0': 'labeler.cheetah.v0.pkl',
    'labeler_lynx_v1': 'labeler.lynx.v1.pkl',
    'labeler_lynx_v2': 'labeler.lynx.v2.pkl',
    'labeler_jaguar_v1': 'labeler.jaguar.v1.pkl',
    'labeler_jaguar_v2': 'labeler.jaguar.v2.pkl',
    'labeler_manta': 'labeler.manta_ray_giant.pkl',
    'labeler_hendrik_dorsal': 'labeler.hendrik_dorsal.pkl',
    'labeler_seaturtle_v1': 'labeler.seaturtle.v1.pkl',
    'labeler_seaturtle_v2': 'labeler.seaturtle.v2.pkl',
    'background_giraffe_masai': 'background.giraffe_masai.npy',
    'background_zebra_plains': 'background.zebra_plains.npy',
    'background_zebra_plains_grevys': 'background.zebra_plains_grevys.npy',
    'background_whale_fluke': 'background.whale_fluke.npy',
    'background_lynx_v2': 'background.lynx.v2.pkl',
    'background_lynx_v3': 'background.lynx.v3.pkl',
    'background_cheetah_v0': 'background.cheetah.v0.pkl',
    'background_cheetah_v1': 'background.cheetah.v1.pkl',
    'background_cheetah_v2': 'background.cheetah.v2.pkl',
    'background_hyaena_v0': 'background.hyaena.v0.pkl',
    'background_jaguar_v1': 'background.jaguar.v1.pkl',
    'background_jaguar_v2': 'background.jaguar.v2.pkl',
    'background_manta': 'background.manta_ray_giant.pkl',
    'background_right_whale_head_v0': 'background.right_whale_head.v0.pkl',
    'background_orca_v0': 'background.whale_orca.v0.pkl',
    'background_whale_sperm_v0': 'background.whale_sperm.v0.pkl',
    'background_skunk_spotted_v0': 'background.skunk_spotted.v0.pkl',
    'background_skunk_spotted_v1': 'background.skunk_spotted.v1.pkl',
    'background_dolphin_spotted': 'background.dolphin_spotted.pkl',
    'background_dolphin_spotted_fin_dorsal': 'background.dolphin_spotted+fin_dorsal.pkl',
    'background_humpback_dorsal': 'background.whake_humpback.dorsal.v0.pkl',
    'background_seadragon_leafy_v1': 'background.seadragon_leafy.v1.pkl',
    'background_seadragon_weedy_v1': 'background.seadragon_weedy.v1.pkl',
    'background_seadragon_leafy_head_v1': 'background.seadragon_leafy+head.v1.pkl',
    'background_seadragon_weedy_head_v1': 'background.seadragon_weedy+head.v1.pkl',
    'background_turtle_green_v1': 'background.turtle_green.v1.pkl',
    'background_turtle_hawksbill_v1': 'background.turtle_hawksbill.v1.pkl',
    'background_turtle_green_head_v1': 'background.turtle_green+head.v1.pkl',
    'background_turtle_hawksbill_head_v1': 'background.turtle_hawksbill+head.v1.pkl',
    'background_candidacy_giraffe_masai': 'background.candidacy.giraffe_masai.pkl',
    'background_candidacy_giraffe_reticulated': 'background.candidacy.giraffe_reticulated.pkl',
    'background_candidacy_turtle_sea': 'background.candidacy.turtle_sea.pkl',
    'background_candidacy_whale_fluke': 'background.candidacy.whale_fluke.pkl',
    'background_candidacy_zebra_grevys': 'background.candidacy.zebra_grevys.pkl',
    'background_candidacy_zebra_plains': 'background.candidacy.zebra_plains.pkl',
    'background_zebra_mountain_v0': 'background.zebra_mountain.v0.pkl',
    'background_iot_v0': 'background.iot.v0.pkl',
    'background_wilddog_v0': 'background.wild_dog.v0.pkl',
    'background_leopard_v0': 'background.leopard.v0.pkl',
    'background_snow_leopard_v0': 'background.snow_leopard.v0.pkl',
    'background_whale_grey_v0': 'background.whale_grey.v0.pkl',
    'background_whale_beluga_v0': 'background.whale_beluga.v0.pkl',
    'background_seals_v0': 'background.seals.v0.pkl',
    'background_grouper_nassau_v0': 'background.grouper_nassau.v0.pkl',
    'background_sea_turtle_v4': 'background.sea_turtle.v4.pkl',
    'background_spotted_eagle_ray_v0': 'background.spotted_eagle_ray.v0.pkl',
    'background_yellow_bellied_toad_v0': 'background.yellow_bellied_toad.v0.pkl',
    'salanader_fire_v0': 'background.salanader_fire.v0.pkl',
    'salanader_fire_adult_v0': 'background.salanader_fire_adult.v0.pkl',
    'aoi2_candidacy': 'aoi2.candidacy.pkl',
    'aoi2_ggr2': 'aoi2.ggr2.pkl',
    'aoi2_hammerhead': 'aoi2.shark_hammerhead.pkl',
    'aoi2_jaguar_v1': 'aoi2.jaguar.v1.pkl',
    'aoi2_jaguar_v2': 'aoi2.jaguar.v2.pkl',
    'viewpoint': 'viewpoint.v1.pkl',
    'caffenet': 'pretrained.caffe.caffenet.slice_0_6_None.pkl',
    'caffenet_conv': 'pretrained.caffe.caffenet.slice_0_10_None.pkl',
    'caffenet_full': 'pretrained.caffe.caffenet.pkl',
    'vggnet': 'pretrained.caffe.vgg.slice_0_6_None.pkl',
    'vggnet_conv': 'pretrained.caffe.vgg.slice_0_32_None.pkl',
    'vggnet_full': 'pretrained.caffe.vgg.pkl',
}


def ensure_model(model, redownload=False):
    try:
        url = MODEL_DOMAIN + MODEL_URLS[model]
        extracted_fpath = ut.grab_file_url(
            url, appname='wbia_cnn', redownload=redownload, check_hash=True
        )
    except KeyError as ex:
        ut.printex(ex, 'model is not uploaded', iswarning=True)
        extracted_fpath = ut.unixjoin(ut.get_app_resource_dir('wbia_cnn'), model)
        ut.assert_exists(extracted_fpath)
    return extracted_fpath


if __name__ == '__main__':
    """

    CommandLine:
        python -m wbia_cnn._plugin_grabmodels.ensure_models
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
