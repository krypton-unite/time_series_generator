from time_series_generator import TimeseriesGenerator, timeseries_generator_from_json
# from keras_preprocessing.sequence import TimeseriesGenerator as TimeseriesGenerator
from keras_preprocessing.sequence import TimeseriesGenerator as kerasGenerator
import numpy as np
import pytest
import json
import os
from pathlib import Path
from datetime import datetime

def test_default_keras_generator():
    data = np.array([[i] for i in range(50)])
    targets = np.array([[i] for i in range(50)])

    data_gen = kerasGenerator(data, targets,
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


def test_default_TSG_generator():
    data = np.array([[i] for i in range(50)])
    targets = np.array([[i] for i in range(50)])

    data_gen = TimeseriesGenerator(data, targets,
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


def test_diffent_lengths():
    data = np.array([[i] for i in range(50)])
    targets = np.array([[i] for i in range(51)])

    with pytest.raises(ValueError) as ve:
        data_gen = TimeseriesGenerator(data, targets,
                                       length=10, sampling_rate=2,
                                       batch_size=2)
    assert str(
        ve.value) == 'Data and targets have to be of same length. Data length is 50 while target length is 51'


def test_too_big_start_index():
    data = np.array([[i] for i in range(50)])
    targets = np.array([[i] for i in range(50)])

    with pytest.raises(ValueError) as ve:
        data_gen = TimeseriesGenerator(data, targets,
                                       length=10, start_index=45, sampling_rate=2,
                                       batch_size=2)
    assert str(
        ve.value) == '`start_index+length=55 > end_index=49` is disallowed, as no part of the sequence would be left to be used as current step.'


def test_shuffled_TSG_generator():
    data = np.array([[i] for i in range(50)])
    targets = np.array([[i] for i in range(50)])

    data_gen = TimeseriesGenerator(data, targets,
                                   length=10, sampling_rate=2,
                                   batch_size=2, shuffle=True)
    assert len(data_gen) == 20

    batch_0 = data_gen[0]
    x, y = batch_0
    assert not np.array_equal(x,
                              np.array([[[0], [2], [4], [6], [8]],
                                        [[1], [3], [5], [7], [9]]]))
    assert not np.array_equal(y,
                              np.array([[10], [11]]))


def test_reversed_TSG_generator():
    data = np.array([[i] for i in range(50)])
    targets = np.array([[i] for i in range(50)])

    data_gen = TimeseriesGenerator(data, targets,
                                   length=10, sampling_rate=2,
                                   batch_size=2, reverse=True)
    assert len(data_gen) == 20

    batch_0 = data_gen[0]
    x, y = batch_0
    assert np.array_equal(x,
                          np.array([[[8],
                                     [6],
                                     [4],
                                     [2],
                                     [0]],
                                    [[9],
                                     [7],
                                     [5],
                                     [3],
                                     [1]]]))
    assert np.array_equal(y,
                          np.array([[10],
                                    [11]]))


def test_TSG_generator_get_config():
    data = np.array([[i] for i in range(50)])
    targets = np.array([[i] for i in range(50)])

    data_gen = TimeseriesGenerator(data, targets,
                                   length=10, sampling_rate=2,
                                   batch_size=2)
    config = data_gen.get_config()
    with open(Path(os.path.abspath('tests'), 'dumped_expected_config.json')) as f:
        read_config = json.load(f)
        assert read_config == config


def test_TSG_generator_to_json():
    data = np.array([[i] for i in range(50)])
    targets = np.array([[i] for i in range(50)])

    data_gen = TimeseriesGenerator(data, targets,
                                   length=10, sampling_rate=2,
                                   batch_size=2)
    ts_json = data_gen.to_json()
    with open(Path(os.path.abspath('tests'), 'dumped_expected_to_json.json')) as f:
        read_ts_json = json.load(f)
        assert read_ts_json == json.loads(ts_json)


def test_TSG_generator_from_json():
    with open(Path(os.path.abspath('tests'), 'dumped_default_tsg.json')) as f:
        read_file = f.read()
        data_gen = timeseries_generator_from_json(read_file)

        assert len(data_gen) == 15

        batch_0 = data_gen[0]
        x, y = batch_0
        assert np.array_equal(x,
                              np.array([[[10],
                                         [12],
                                         [14],
                                         [16],
                                         [18]],
                                        [[11],
                                         [13],
                                         [15],
                                         [17],
                                         [19]]]))
        assert np.array_equal(y, np.array([[20],
                                           [21]]))

def test_Data_not_JSON_Serializable():
    data = np.array([datetime(1982, 12, 23), datetime(2009, 2, 4)])
    targets = np.array([[i] for i in range(2)])

    data_gen = TimeseriesGenerator(data, targets,
                                   length=1, sampling_rate=1,
                                   batch_size=1)
    with pytest.raises(TypeError) as ve:
        data_gen.get_config()

    assert str(ve.value) == "('Data not JSON Serializable:', [datetime.datetime(1982, 12, 23, 0, 0), datetime.datetime(2009, 2, 4, 0, 0)])"

def test_Targets_not_JSON_Serializable():
    data = np.array([[i] for i in range(2)])
    targets = np.array([datetime(1982, 12, 23), datetime(2009, 2, 4)])

    data_gen = TimeseriesGenerator(data, targets,
                                   length=1, sampling_rate=1,
                                   batch_size=1)
    with pytest.raises(TypeError) as ve:
        data_gen.get_config()

    assert str(ve.value) == "('Targets not JSON Serializable:', [datetime.datetime(1982, 12, 23, 0, 0), datetime.datetime(2009, 2, 4, 0, 0)])"