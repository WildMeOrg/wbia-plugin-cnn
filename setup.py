#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
from utool import util_setup
from setuptools import setup
import sys
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


def parse_requirements(fname='requirements.txt', with_version=True):
    """
    Parse the package dependencies listed in a requirements file but strips
    specific versioning information.

    Args:
        fname (str): path to requirements file
        with_version (bool, default=True): if true include version specs

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
            if line.startswith('-e ') or line.startswith('git+'):
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


# INSTALL_OPTIONAL = [
#    'tornado',
#    'flask',
#    'autopep8',
#    'pyfiglet',
#    'theano',
#    'pylearn2'
#    'lasenge'
# ]


if __name__ == '__main__':
    print('[setup] Entering IBEIS setup')
    kwargs = dict(
        setup_fpath=__file__,
        name='wbia_cnn',
        # author='Hendrik Weideman, Jason Parham, and Jon Crall',
        # author_email='erotemic@gmail.com',
        packages=util_setup.find_packages() + ['lasagne'],
        package_dir={'lasagne': 'Lasagne/lasagne'},
        # --- VERSION ---
        # The following settings retreive the version from git.
        # See https://github.com/pypa/setuptools_scm/ for more information
        setup_requires=['setuptools_scm'],
        use_scm_version={
            'write_to': 'wbia_cnn/_version.py',
            'write_to_template': '__version__ = "{version}"',
            'tag_regex': '^(?P<prefix>v)?(?P<version>[^\\+]+)(?P<suffix>.*)?$',
            'local_scheme': 'dirty-tag',
        },
        license=util_setup.read_license('LICENSE'),
        long_description=util_setup.parse_readme('README.md'),
        ext_modules=util_setup.find_ext_modules(),
        cmdclass=util_setup.get_cmdclass(),
        project_dirs=PROJECT_DIRS,
        chmod_patterns=CHMOD_PATTERNS,
        clutter_patterns=CLUTTER_PATTERNS,
        clutter_dirs=CLUTTER_DIRS,
        install_requires=parse_requirements('requirements/runtime.txt'),
        extras_require={
            'all': parse_requirements('requirements.txt'),
            'build': parse_requirements('requirements/build.txt'),
            'runtime': parse_requirements('requirements/runtime.txt'),
        },
        # cython_files=CYTHON_FILES,
    )

    print('kwargs = %s' % (ut.dict_str(kwargs),))
    setup(**kwargs)
