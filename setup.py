from os import path

from setuptools import setup, find_packages

from fit.version import __version__

_dir = path.abspath(path.dirname(__file__))

with open(path.join(_dir, 'fit', 'version.py')) as f:
    exec(f.read())

with open(path.join(_dir, 'README.md')) as f:
    long_description = f.read()

setup(name='fourier-image-transformers',
      version=__version__,
      description='Fourier Image Transformer',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='https://github.com/juglab/FourierImageTransformer/',
      author='Tim-Oliver Buchholz, Florian Jug',
      author_email='tibuch@mpi-cbg.de, jug@mpi-cbg.de',
      license='BSD 3-Clause License',
      packages=find_packages(),

      project_urls={
          'Repository': 'https://github.com/juglab/FourierImageTransformer/',
      },

      classifiers=[
          'Development Status :: 4 - Beta',
          'Intended Audience :: Science/Research',
          'Topic :: Scientific/Engineering',
          'License :: OSI Approved :: BSD License',

          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
      ],

      install_requires=[
          "numpy",
          "scipy",
          "matplotlib",
          "torch>=1.7.0",
          "torchvision",
          "tifffile",
          "tqdm",
          "pytorch-fast-transformers",
          "dival",
          "pytorch-lightning",
          "jupyter"
      ]
      )
