# Time Series Generator

[![PyPI version](https://badge.fury.io/py/time-series-generator.svg)](https://pypi.org/project/time-series-generator/) [![Documentation Status](https://readthedocs.org/projects/time-series-generator/badge/?version=latest)](https://readthedocs.org/projects/time-series-generator/) [![travis](https://travis-ci.org/krypton-unite/time_series_generator.svg?branch=master)](https://travis-ci.org/github/krypton-unite/time_series_generator) [![codecov](https://codecov.io/gh/krypton-unite/time_series_generator/branch/master/graph/badge.svg)](https://codecov.io/gh/krypton-unite/time_series_generator) [![GitHub license](https://img.shields.io/github/license/krypton-unite/time_series_generator)](https://github.com/krypton-unite/time_series_predictor)


## Description
Emulates Teras Tensorflow TimeSeriesGenerator functionality

## Instalation

```terminal
pip install time-series-generator
```

## Usage

```python
import numpy as np
from time_series_generator import TimeseriesGenerator

data = np.array([[i] for i in range(50)])
targets = np.array([[i] for i in range(50)])

data_gen = TimeSeriesGenerator(data, targets,
                                length=10, sampling_rate=2,
                                batch_size=2)
assert len(data_gen) == 20

batch_0 = data_gen[0]
x, y = batch_0
assert np.array_equal(x,
                        np.array([[[0], [2], [4], [6], [8]],
                                [[1], [3], [5], [7], [9]]]))
assert np.array_equal(y,
                        np.array([[10], [11]]))
```

## Test

Run in the terminal at project root folder:

```terminal
pytest
```

## Development opportunities

> A limitation of the TimeseriesGenerator is that it does not directly support multi-step outputs. Specifically, it will not create the multiple steps that may be required in the target sequence.

> Nevertheless, if you prepare your target sequence to have multiple steps, it will honor and use them as the output portion of each sample. This means the onus is on you to prepare the expected output for each time step.

Brownlee, Jason

- The above process might be automatable.


## References
- [Brownlee, Jason. How to Use the TimeseriesGenerator for Time Series Forecasting in Keras](https://machinelearningmastery.com/how-to-improve-deep-learning-model-robustness-by-adding-noise/)