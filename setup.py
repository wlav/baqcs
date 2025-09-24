import codecs, glob, os, sys, re
from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))
with codecs.open(os.path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='baqcs',
    long_description=long_description,

    classifiers=[
        'Development Status :: 3 - Alpha',

        'Intended Audience :: Developers',

        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.12',

        'Natural Language :: English'
    ],

    package_dir={'': 'src'},
    packages=find_packages('src', include='baqcs'),
)
