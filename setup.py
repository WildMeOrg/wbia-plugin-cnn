#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import six
from utool import util_setup
from setuptools import setup
import utool as ut

(print, rrr, profile) = ut.inject2(__name__)


CHMOD_PATTERNS = [
    'run_tests.sh',
]

PROJECT_DIRS = [
    'wbia_cnn',
]

CLUTTER_PATTERNS = [
    "'",
    '*.dump.txt',
    '*.prof',
    '*.prof.txt',
    '*.lprof',
    '*.ln.pkg',
    'timeings.txt',
]

CLUTTER_DIRS = [
    'logs/',
    'dist/',
    'testsuite',
    '__pycache__/',
]

"""
Need special theano
References:
    http://lasagne.readthedocs.org/en/latest/user/installation.html
    pip install -r https://raw.githubusercontent.com/Lasagne/Lasagne/v0.1/requirements.txt
"""

INSTALL_REQUIRES = [
    'scikit-learn >= 0.16.1',
    'theano',
    'lasagne',
    # 'h5py',  # Install this instead 'sudo apt-get install libhdf5-dev' due to Numpy versioning issues
    #'pylearn2',
    #'git+git://github.com/lisa-lab/pylearn2.git'
    #'utool >= 1.0.0.dev1',
    #'vtool >= 1.0.0.dev1',
    ##'pyhesaff >= 1.0.0.dev1',
    #'pyrf >= 1.0.0.dev1',
    #'guitool >= 1.0.0.dev1',
    #'plottool >= 1.0.0.dev1',
    #'matplotlib >= 1.3.1',
    #'scipy >= 0.13.2',
    #'numpy >= 1.8.0',
    #'Pillow >= 2.4.0',
    #'psutil',
    #'requests >= 0.8.2',
    #'setproctitle >= 1.1.8',
    ##'decorator',
    #'lockfile >= 0.10.2',
    #'apipkg',
    #'objgraph',
    #'pycallgraph',
    #'gevent',
    #'PyQt 4/5 >= 4.9.1', # cannot include because pyqt4 is not in pip
]

# INSTALL_OPTIONAL = [
#    'tornado',
#    'flask',
#    'autopep8',
#    'pyfiglet',
#    'theano',
#    'pylearn2'
#    'lasenge'
# ]

if six.PY2:
    INSTALL_REQUIRES.append('requests >= 0.8.2')


def parse_requirements(fname='requirements.txt', with_version=False):
    """
    Parse the package dependencies listed in a requirements file but strips
    specific versioning information.

    Args:
        fname (str): path to requirements file
        with_version (bool, default=False): if true include version specs

    Returns:
        List[str]: list of requirements items

    CommandLine:
        python -c "import setup; print(setup.parse_requirements())"
        python -c "import setup; print(chr(10).join(setup.parse_requirements(with_version=True)))"
    """
    from os.path import exists
    import re

    require_fpath = fname

    def parse_line(line):
        """
        Parse information from a line in a requirements text file
        """
        if line.startswith('-r '):
            # Allow specifying requirements in other files
            target = line.split(' ')[1]
            for info in parse_require_file(target):
                yield info
        else:
            info = {'line': line}
            if line.startswith('-e '):
                info['package'] = line.split('#egg=')[1]
            else:
                # Remove versioning from the package
                pat = '(' + '|'.join(['>=', '==', '>']) + ')'
                parts = re.split(pat, line, maxsplit=1)
                parts = [p.strip() for p in parts]

                info['package'] = parts[0]
                if len(parts) > 1:
                    op, rest = parts[1:]
                    if ';' in rest:
                        # Handle platform specific dependencies
                        # http://setuptools.readthedocs.io/en/latest/setuptools.html#declaring-platform-specific-dependencies
                        version, platform_deps = map(str.strip, rest.split(';'))
                        info['platform_deps'] = platform_deps
                    else:
                        version = rest  # NOQA
                    info['version'] = (op, version)
            yield info

    def parse_require_file(fpath):
        with open(fpath, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line and not line.startswith('#'):
                    for info in parse_line(line):
                        yield info

    def gen_packages_items():
        if exists(require_fpath):
            for info in parse_require_file(require_fpath):
                parts = [info['package']]
                if with_version and 'version' in info:
                    parts.extend(info['version'])
                if not sys.version.startswith('3.4'):
                    # apparently package_deps are broken in 3.4
                    platform_deps = info.get('platform_deps')
                    if platform_deps is not None:
                        parts.append(';' + platform_deps)
                item = ''.join(parts)
                yield item

    packages = list(gen_packages_items())
    return packages


if __name__ == '__main__':
    print('[setup] Entering IBEIS setup')
    kwargs = util_setup.setuptools_setup(
        setup_fpath=__file__,
        name='wbia_cnn',
        # author='Hendrik Weideman, Jason Parham, and Jon Crall',
        # author_email='erotemic@gmail.com',
        packages=util_setup.find_packages(),
        version=util_setup.parse_package_for_version('wbia_cnn'),
        license=util_setup.read_license('LICENSE'),
        long_description=util_setup.parse_readme('README.md'),
        ext_modules=util_setup.find_ext_modules(),
        cmdclass=util_setup.get_cmdclass(),
        project_dirs=PROJECT_DIRS,
        chmod_patterns=CHMOD_PATTERNS,
        clutter_patterns=CLUTTER_PATTERNS,
        clutter_dirs=CLUTTER_DIRS,
        install_requires=INSTALL_REQUIRES
        # cython_files=CYTHON_FILES,
    )
    import utool as ut

    print('kwargs = %s' % (ut.dict_str(kwargs),))
    setup(**kwargs)
