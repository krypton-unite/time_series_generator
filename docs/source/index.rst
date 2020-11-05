.. Time Series Generator documentation master file, created by
   sphinx-quickstart on Tue Jun  2 17:36:26 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

###################################
Time Series Generator documentation
###################################

.. image:: https://badge.fury.io/py/time-series-generator.svg
   :target: https://badge.fury.io/py/time-series-generator
   :alt: PyPI version

.. image:: https://readthedocs.org/projects/time-series-generator/badge/?version=latest
   :target: https://readthedocs.org/projects/time-series-generator/
   :alt: Documentation Status

.. image:: https://travis-ci.org/krypton-unite/time_series_generator.svg?branch=master
   :target: https://travis-ci.org/github/krypton-unite/time_series_generator
   :alt: Travis

.. image:: https://codecov.io/gh/krypton-unite/time_series_generator/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/krypton-unite/time_series_generator
   :alt: Code Coverage

.. image:: https://img.shields.io/github/license/krypton-unite/time_series_generator
   :target: https://github.com/krypton-unite/time_series_generator/blob/master/LICENSE
   :alt: GitHub license

Welcome to Time Series Generator's documentation!

This documents the `python package`_ sourced from the following `repository`_.

.. _python package: https://pypi.org/project/time-series-generator/
.. _repository: https://github.com/krypton-unite/time_series_generator.git

.. image:: images/read_online.*
   :target: https://time-series-generator.readthedocs.io/en/latest/index.html#
   :alt: `Read Online`
   :class: third

.. image:: images/download_pdf.*
   :target: https://readthedocs.org/projects/time_series_generator/downloads/pdf/latest
   :alt: `Download PDF`
   :class: third

.. image:: images/download_epub.*
   :target: https://readthedocs.org/projects/time_series_generator/downloads/epub/latest/
   :alt: `Download EPUB`
   :class: third

***********
Description
***********

Emulates Teras Tensorflow TimeSeriesGenerator functionality presenting a candidate solution for the direct multi-step outputs limitation in Keras version.


************
Installation
************

.. code-block:: terminal

   pip install time-series-generator


*****
Usage
*****

Main Functionality
==================

.. image:: images/machine_learning_mastery.png
   :target: https://machinelearningmastery.com/how-to-use-the-timeseriesgenerator-for-time-series-forecasting-in-keras/
   :alt: `Read Online`
   :class: third
   :align: center

Candidate Improvement
=====================

.. role:: python(code)
   :language: python

Addition of the keyworded argument :python:`length_output`.

.. code-block:: python
   :emphasize-lines: 7

   # define dataset
   series = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
   target = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
   # define generator
   n_input = 2
   n_output = 2
   generator = TimeseriesGenerator(series, target, length=n_input, length_output=n_output, batch_size=1)
   # print each sample
   for i in range(len(generator)):
      x, y = generator[i]
      print('%s => %s' % (x, y))

Output
------

.. code-block:: terminal

   [[1 2]] => [[3 4]]
   [[2 3]] => [[4 5]]
   [[3 4]] => [[5 6]]
   [[4 5]] => [[6 7]]
   [[5 6]] => [[7 8]]
   [[6 7]] => [[8 9]]
   [[7 8]] => [[9 10]]


**************************
Documentation for the Code
**************************
.. toctree::
   :maxdepth: 2
   
   modules.rst

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
