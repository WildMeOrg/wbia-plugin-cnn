#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""
import logging
from wbia.detecttools.directory import Directory
from os.path import join, abspath, exists, basename
import utool as ut
import numpy as np

(print, rrr, profile) = ut.inject2(__name__)
logger = logging.getLogger()


def process_image_directory(project_name, size, reset=True):
    import cv2

    # Raw folders
    raw_path = abspath(join('..', 'data', 'raw'))
    processed_path = abspath(join('..', 'data', 'processed'))
    # Project folders
    project_raw_path = join(raw_path, project_name)
    project_processed_path = join(processed_path, project_name)

    # Load raw data
    direct = Directory(project_raw_path, include_file_extensions='images')

    # Reset / create paths if not exist
    if exists(project_processed_path) and reset:
        ut.remove_dirs(project_processed_path)
    ut.ensuredir(project_processed_path)

    # Process by resizing the images into the desired shape
    for file_path in direct.files():
        file_name = basename(file_path)
        logger.info('Processing %r' % (file_name,))
        image = cv2.imread(file_path)
        image = cv2.resize(image, size, interpolation=cv2.INTER_LANCZOS4)
        dest_path = join(project_processed_path, file_name)
        cv2.imwrite(dest_path, image)


def numpy_processed_directory(
    project_name,
    numpy_ids_file_name='ids.npy',
    numpy_x_file_name='X.npy',
    numpy_y_file_name='y.npy',
    labels_file_name='labels.csv',
    reset=True,
):
    import cv2

    # Raw folders
    processed_path = abspath(join('..', 'data', 'processed'))
    labels_path = abspath(join('..', 'data', 'labels'))
    numpy_path = abspath(join('..', 'data', 'numpy'))
    # Project folders
    project_processed_path = join(processed_path, project_name)
    project_labels_path = join(labels_path, project_name)
    project_numpy_path = join(numpy_path, project_name)
    # Project files
    project_numpy_ids_file_name = join(project_numpy_path, numpy_ids_file_name)
    project_numpy_x_file_name = join(project_numpy_path, numpy_x_file_name)
    project_numpy_y_file_name = join(project_numpy_path, numpy_y_file_name)
    project_numpy_labels_file_name = join(project_labels_path, labels_file_name)

    # Load raw data
    direct = Directory(project_processed_path, include_file_extensions='images')
    label_dict = {}
    for line in open(project_numpy_labels_file_name):
        line = line.strip().split(',')
        file_name = line[0].strip()
        label = line[1].strip()
        label_dict[file_name] = label

    # Reset / create paths if not exist
    if exists(project_numpy_path) and reset:
        ut.remove_dirs(project_numpy_path)
    ut.ensuredir(project_numpy_path)

    # Get shape for all images
    shape_x = list(cv2.imread(direct.files()[0]).shape)
    if len(shape_x) == 2:
        shape_x = shape_x + [1]
    shape_x = tuple([len(direct.files())] + shape_x[::-1])  # NOQA
    shape_y = shape_x[0:1]  # NOQA

    # Create numpy arrays
    # X = np.empty(shape_x, dtype=np.uint8)
    # y = np.empty(shape_y, dtype=np.uint8)
    ids = []
    X = []
    y = []

    # Process by loading images into the numpy array for saving
    for index, file_path in enumerate(direct.files()):
        file_name = basename(file_path)
        logger.info('Processing %r' % (file_name,))
        image = cv2.imread(file_path)
        try:
            label = label_dict[file_name]
            # X[index] = np.array(cv2.split(image))
            # y[index] = label
            # X.append(np.array(cv2.split(image)))  # Lasagne format
            ids.append(file_name)
            X.append(image)  # cv2 format
            y.append(label)
        except KeyError:
            logger.info('Cannot find label...skipping')
            # raw_input()

    ids = np.array(ids)
    X = np.array(X, dtype=np.uint8)
    # y = np.array(y, dtype=np.uint8)
    y = np.array(y)

    # Save numpy array
    logger.info('  ids.shape = %r' % (ids.shape,))
    logger.info('  ids.dtype = %r' % (ids.dtype,))
    logger.info('  X.shape   = %r' % (X.shape,))
    logger.info('  X.dtype   = %r' % (X.dtype,))
    logger.info('  y.shape   = %r' % (y.shape,))
    logger.info('  y.dtype   = %r' % (y.dtype,))
    np.save(project_numpy_ids_file_name, ids)
    np.save(project_numpy_x_file_name, X)
    np.save(project_numpy_y_file_name, y)


def numpy_processed_directory2(
    extracted_path,
    numpy_ids_file_name='ids.npy',
    numpy_x_file_name='X.npy',
    numpy_y_file_name='y.npy',
    labels_file_name='labels.csv',
    reset=True,
    verbose=False,
):
    import cv2

    logger.info('Caching images into Numpy files...')

    raw_path = join(extracted_path, 'raw')
    labels_path = join(extracted_path, 'labels')

    # Project files
    project_numpy_ids_file_name = join(raw_path, numpy_ids_file_name)
    project_numpy_x_file_name = join(raw_path, numpy_x_file_name)
    project_numpy_y_file_name = join(labels_path, numpy_y_file_name)
    project_numpy_labels_file_name = join(labels_path, labels_file_name)

    # Load raw data
    direct = Directory(raw_path, include_file_extensions='images')
    label_dict = {}
    for line in open(project_numpy_labels_file_name):
        line = line.strip().split(',')
        file_name = line[0].strip()
        label = line[1].strip()
        label_dict[file_name] = label

    # Get shape for all images
    shape_x = list(cv2.imread(direct.files()[0]).shape)
    if len(shape_x) == 2:
        shape_x = shape_x + [1]
    shape_x = tuple([len(direct.files())] + shape_x[::-1])  # NOQA
    shape_y = shape_x[0:1]  # NOQA

    # Create numpy arrays
    # X = np.empty(shape_x, dtype=np.uint8)
    # y = np.empty(shape_y, dtype=np.uint8)
    ids = []
    X = []
    y = []

    # Process by loading images into the numpy array for saving
    for index, file_path in enumerate(direct.files()):
        file_name = basename(file_path)
        if verbose:
            logger.info('Processing %r' % (file_name,))
        image = cv2.imread(file_path)
        try:
            label = label_dict[file_name]
            # X[index] = np.array(cv2.split(image))
            # y[index] = label
            # X.append(np.array(cv2.split(image)))  # Lasagne format
            ids.append(file_name)
            X.append(image)  # cv2 format
            y.append(label)
        except KeyError:
            logger.info('Cannot find label...skipping')
            # raw_input()

    ids = np.array(ids)
    X = np.array(X, dtype=np.uint8)
    # y = np.array(y, dtype=np.uint8)
    y = np.array(y)

    # Save numpy array
    logger.info('  ids.shape = %r' % (ids.shape,))
    logger.info('  ids.dtype = %r' % (ids.dtype,))
    logger.info('  X.shape   = %r' % (X.shape,))
    logger.info('  X.dtype   = %r' % (X.dtype,))
    logger.info('  y.shape   = %r' % (y.shape,))
    logger.info('  y.dtype   = %r' % (y.dtype,))
    np.save(project_numpy_ids_file_name, ids)
    np.save(project_numpy_x_file_name, X)
    np.save(project_numpy_y_file_name, y)

    return (
        project_numpy_ids_file_name,
        project_numpy_x_file_name,
        project_numpy_y_file_name,
    )


def numpy_processed_directory3(
    extracted_path,
    numpy_ids_file_name='ids.npy',
    numpy_x_file_name='X.npy',
    numpy_y_file_name='y.npy',
    labels_file_name='labels.csv',
    categories_file_name='categories.csv',
    reset=True,
    verbose=False,
):
    import cv2

    logger.info('Caching images into Numpy files with category vector...')

    raw_path = join(extracted_path, 'raw')
    labels_path = join(extracted_path, 'labels')

    # Project files
    project_numpy_ids_file_name = join(raw_path, numpy_ids_file_name)
    project_numpy_x_file_name = join(raw_path, numpy_x_file_name)
    project_numpy_y_file_name = join(labels_path, numpy_y_file_name)
    project_numpy_labels_file_name = join(labels_path, labels_file_name)
    project_numpy_categories_file_name = join(labels_path, categories_file_name)

    category_list = []
    for line in open(project_numpy_categories_file_name):
        category = line.strip()
        if len(category) > 0:
            category_list.append(category)

    # Load raw data
    direct = Directory(raw_path, include_file_extensions='images')
    label_dict = {}
    count_dict = {}
    for line in open(project_numpy_labels_file_name):
        line = line.strip().split(',')
        file_name = line[0].strip()
        label = line[1].strip()
        label_list = label.split(';')
        label_set = set(label_list)
        label = [1 if category_ in label_set else 0 for category_ in category_list]
        assert 1 in label
        count = label.count(1)
        if count not in count_dict:
            count_dict[count] = 0
        count_dict[count] += 1
        label_dict[file_name] = label

    logger.info('count_dict = %s' % (ut.repr3(count_dict),))

    # Get shape for all images
    shape_x = list(cv2.imread(direct.files()[0]).shape)
    if len(shape_x) == 2:
        shape_x = shape_x + [1]
    shape_x = tuple([len(direct.files())] + shape_x[::-1])  # NOQA
    shape_y = shape_x[0:1]  # NOQA

    # Create numpy arrays
    # X = np.empty(shape_x, dtype=np.uint8)
    # y = np.empty(shape_y, dtype=np.uint8)
    ids = []
    X = []
    y = []

    # Process by loading images into the numpy array for saving
    for index, file_path in enumerate(direct.files()):
        file_name = basename(file_path)
        if verbose:
            logger.info('Processing %r' % (file_name,))
        image = cv2.imread(file_path)
        try:
            label = np.array(label_dict[file_name])
            # X[index] = np.array(cv2.split(image))
            # y[index] = label
            # X.append(np.array(cv2.split(image)))  # Lasagne format
            ids.append(file_name)
            X.append(image)  # cv2 format
            y.append(label)
        except KeyError:
            logger.info('Cannot find label...skipping')
            # raw_input()

    ids = np.array(ids)
    X = np.array(X, dtype=np.uint8)
    # y = np.array(y, dtype=np.uint8)
    y = np.vstack(y)

    # Save numpy array
    logger.info('  ids.shape  = %r' % (ids.shape,))
    logger.info('  ids.dtype  = %r' % (ids.dtype,))
    logger.info('  X.shape    = %r' % (X.shape,))
    logger.info('  X.dtype    = %r' % (X.dtype,))
    logger.info('  y.shape    = %r' % (y.shape,))
    logger.info('  y.dtype    = %r' % (y.dtype,))
    logger.info('  categories = %r' % (category_list,))
    np.save(project_numpy_ids_file_name, ids)
    np.save(project_numpy_x_file_name, X)
    np.save(project_numpy_y_file_name, y)

    return (
        project_numpy_ids_file_name,
        project_numpy_x_file_name,
        project_numpy_y_file_name,
    )


def numpy_processed_directory4(
    extracted_path,
    numpy_ids_file_name='ids.npy',
    numpy_x_file_name='X.npy',
    numpy_y_file_name='y.npy',
    labels_file_name='labels.csv',
    reset=True,
    verbose=False,
):
    logger.info('Caching images into Numpy files with category vector...')

    raw_path = join(extracted_path, 'raw')
    labels_path = join(extracted_path, 'labels')

    # Project files
    project_numpy_ids_file_name = join(raw_path, numpy_ids_file_name)
    project_numpy_x_file_name = join(raw_path, numpy_x_file_name)
    project_numpy_y_file_name = join(labels_path, numpy_y_file_name)
    project_numpy_labels_file_name = join(labels_path, labels_file_name)

    # Load raw data
    direct = Directory(raw_path, include_file_extensions=['npy'])
    label_dict = {}
    for line in open(project_numpy_labels_file_name):
        line = line.strip().split(',')
        file_name = line[0].strip()
        label = line[1].strip()
        label_list = label.split(';')
        label_list = [list(map(float, _.split('^'))) for _ in label_list]
        label = np.array(label_list)
        label_dict[file_name] = label

    # Create numpy arrays
    ids = []
    X = []
    y = []

    # Process by loading images into the numpy array for saving
    for index, file_path in enumerate(direct.files()):
        file_name = basename(file_path)
        if verbose:
            logger.info('Processing %r' % (file_name,))

        with open(file_path, 'r') as file_:
            data = np.load(file_)
        try:
            label = label_dict[file_name]
            ids.append(file_name)
            X.append(data)
            y.append(label)
        except KeyError:
            logger.info('Cannot find label...skipping')

    ids = np.array(ids)
    X = np.array(X, dtype=np.float32)
    y = np.array(y)

    # Save numpy array
    logger.info('  ids.shape  = %r' % (ids.shape,))
    logger.info('  ids.dtype  = %r' % (ids.dtype,))
    logger.info('  X.shape    = %r' % (X.shape,))
    logger.info('  X.dtype    = %r' % (X.dtype,))
    logger.info('  y.shape    = %r' % (y.shape,))
    logger.info('  y.dtype    = %r' % (y.dtype,))
    np.save(project_numpy_ids_file_name, ids)
    np.save(project_numpy_x_file_name, X)
    np.save(project_numpy_y_file_name, y)

    return (
        project_numpy_ids_file_name,
        project_numpy_x_file_name,
        project_numpy_y_file_name,
    )


def numpy_processed_directory5(
    extracted_path,
    numpy_ids_file_name='ids.npy',
    numpy_x_file_name='X.npy',
    numpy_y_file_name='y.npy',
    labels_file_name='labels.csv',
    reset=True,
    verbose=False,
):
    import cv2

    logger.info('Caching images into Numpy files with category vector...')

    raw_path = join(extracted_path, 'raw')
    labels_path = join(extracted_path, 'labels')

    # Project files
    project_numpy_ids_file_name = join(raw_path, numpy_ids_file_name)
    project_numpy_x_file_name = join(raw_path, numpy_x_file_name)
    project_numpy_y_file_name = join(labels_path, numpy_y_file_name)
    project_numpy_labels_file_name = join(labels_path, labels_file_name)

    # Load raw data
    direct = Directory(raw_path, include_file_extensions='images')
    label_dict = {}
    for line in open(project_numpy_labels_file_name):
        line = line.strip().split(',')
        file_name = line[0].strip()
        label = line[1].strip()
        label_list = label.split(';')
        label_list = [list(map(float, _.split('^'))) for _ in label_list]
        label = np.array(label_list)
        label_dict[file_name] = label

    # Create numpy arrays
    ids = []
    X = []
    y = []

    # Process by loading images into the numpy array for saving
    for index, file_path in enumerate(direct.files()):
        file_name = basename(file_path)
        if verbose:
            logger.info('Processing %r' % (file_name,))

        image = cv2.imread(file_path, -1)
        try:
            label = label_dict[file_name]
            ids.append(file_name)
            X.append(image)
            y.append(label)
        except KeyError:
            logger.info('Cannot find label...skipping')

    ids = np.array(ids)
    X = np.array(X, dtype=np.uint8)
    y = np.array(y)

    # Save numpy array
    logger.info('  ids.shape  = %r' % (ids.shape,))
    logger.info('  ids.dtype  = %r' % (ids.dtype,))
    logger.info('  X.shape    = %r' % (X.shape,))
    logger.info('  X.dtype    = %r' % (X.dtype,))
    logger.info('  y.shape    = %r' % (y.shape,))
    logger.info('  y.dtype    = %r' % (y.dtype,))
    np.save(project_numpy_ids_file_name, ids)
    np.save(project_numpy_x_file_name, X)
    np.save(project_numpy_y_file_name, y)

    return (
        project_numpy_ids_file_name,
        project_numpy_x_file_name,
        project_numpy_y_file_name,
    )


def view_numpy_data(project_namel, numpy_x_file_name='X.npy', numpy_y_file_name='y.npy'):
    # Raw folders
    numpy_path = abspath(join('..', 'data', 'numpy'))
    # Project folders
    project_numpy_path = join(numpy_path, project_name)
    # Project files
    project_numpy_x_file_name = join(project_numpy_path, numpy_x_file_name)
    project_numpy_y_file_name = join(project_numpy_path, numpy_y_file_name)

    X = np.load(project_numpy_x_file_name)
    y = np.load(project_numpy_y_file_name)

    logger.info('  X.shape = %r' % (X.shape,))
    logger.info('  X.dtype = %r' % (X.dtype,))
    logger.info('  y.shape = %r' % (y.shape,))
    logger.info('  y.dtype = %r' % (y.dtype,))


if __name__ == '__main__':
    # project_name = 'viewpoint_large'
    # # size = (96, 96)
    # # process_image_directory(project_name, size)
    # numpy_processed_directory(project_name)

    # project_name = 'viewpoint_pz'
    # # size = (64, 64)
    # # process_image_directory(project_name, size)
    # numpy_processed_directory(project_name)

    # project_name = 'quality_pz'
    # # size = (64, 64)
    # # process_image_directory(project_name, size)
    # numpy_processed_directory(project_name)

    project_name = 'background_patches'
    # size = (64, 64)
    # process_image_directory(project_name, size)
    numpy_processed_directory(project_name)

    view_numpy_data(project_name)
