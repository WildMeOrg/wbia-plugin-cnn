# -*- coding: utf-8 -*-
def main():  # nocover
    import wbia_cnn  # NOQA
    import wbia_cnn._plugin  # NOQA

    print('Looks like the imports worked')
    print('wbia_cnn = {!r}'.format(wbia_cnn))
    print('wbia_cnn.__file__ = {!r}'.format(wbia_cnn.__file__))
    print('wbia_cnn.__version__ = {!r}'.format(wbia_cnn.__version__))

    import wbia  # NOQA

    print('[wbia_cnn] Importing ibeis')
    print('wbia = {!r}'.format(wbia))
    print('wbia.__file__ = {!r}'.format(wbia.__file__))
    print('wbia.__version__ = {!r}'.format(wbia.__version__))


if __name__ == '__main__':
    """
    CommandLine:
       python -m utool
    """
    main()
