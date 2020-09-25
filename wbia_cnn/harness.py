#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DEPRICATED:
    code is now in abstract_model under fitting

constructs the Theano optimization and trains a learning model,
optionally by initializing the network with pre-trained weights.

http://cs231n.github.io/neural-networks-3/#distr

Pretrained Models:
    https://github.com/fchollet/deep-learning-models
"""
import logging
from six.moves import input, zip, range  # NOQA
import utool as ut

print, rrr, profile = ut.inject2(__name__)
logger = logging.getLogger()


def _clean(model, theano_forward, X_list, y_list, min_conf=0.95):
    from wbia_cnn import batch_processing as batch
    import random

    # Perform testing
    clean_outputs = batch.process_batch(
        model,
        X_list,
        y_list,
        theano_forward,
        augment_on=False,
        randomize_batch_order=False,
    )
    prediction_list = clean_outputs['labeled_predictions']
    confidence_list = clean_outputs['confidences']
    enumerated = enumerate(zip(y_list, prediction_list, confidence_list))

    switched_counter = 0
    switched = {}
    for index, (y, prediction, confidence) in enumerated:
        if confidence < min_conf:
            continue
        if y == prediction:
            continue
        if random.uniform(0.0, 1.0) > confidence:
            continue
        # Perform the switching
        y_list[index] = prediction
        switched_counter += 1
        # Keep track of changes
        y = str(y)
        prediction = str(prediction)
        if y not in switched:
            switched[y] = {}
        if prediction not in switched[y]:
            switched[y][prediction] = 0
        switched[y][prediction] += 1

    total = len(y_list)
    ratio = switched_counter / total
    args = (
        switched_counter,
        total,
        ratio,
    )
    logger.info('[_clean] Cleaned Data... [ %d / %d ] ( %0.04f )' % args)
    for src in sorted(switched.keys()):
        for dst in sorted(switched[src].keys()):
            logger.info(
                '[_clean] \t%r -> %r : %d'
                % (
                    src,
                    dst,
                    switched[src][dst],
                )
            )

    return y_list
