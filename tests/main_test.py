from time_series_generator import TimeseriesGenerator, timeseries_generator_from_json
from keras_preprocessing.sequence import TimeseriesGenerator as kerasGenerator
import numpy as np
import pytest
import json
from datetime import datetime
from .helpers import get_json_from_file, get_data_from_file, write_data_to_file

# @pytest.mark.skip
def test_default_keras_generator():
    data = np.array([[i] for i in range(50)])
    targets = np.array([[i] for i in range(50)])

    data_gen = kerasGenerator(data, targets,
                            length=10,
                            sampling_rate=2,
                            batch_size=2)
    assert len(data_gen) == 20

    batch_0 = data_gen[0]
    x, y = batch_0
    assert np.array_equal(x,
                          np.array([[[0], [2], [4], [6], [8]],
                                    [[1], [3], [5], [7], [9]]]))
    assert np.array_equal(y,
                          np.array([[10], [11]]))

# @pytest.mark.skip
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

# @pytest.mark.skip
def test_diffent_lengths():
    data = np.array([[i] for i in range(50)])
    targets = np.array([[i] for i in range(51)])

    with pytest.raises(ValueError) as ve:
        data_gen = TimeseriesGenerator(data, targets,
                                    length=10, sampling_rate=2,
                                    batch_size=2)
    assert str(
        ve.value) == 'Data and targets have to be of same length. Data length is 50 while target length is 51'

# @pytest.mark.skip
def test_too_big_start_index():
    data = np.array([[i] for i in range(50)])
    targets = np.array([[i] for i in range(50)])

    with pytest.raises(ValueError) as ve:
        data_gen = TimeseriesGenerator(data, targets,
                                    length=10, start_index=45, sampling_rate=2,
                                    batch_size=2)
    assert str(
        ve.value) == '`start_index+length=55 > end_index=49` is disallowed, as no part of the sequence would be left to be used as current step.'

# @pytest.mark.skip
def test_shuffled_TSG_generator():
    data = np.array([[i] for i in range(50)])
    targets = data

    results = {}
    for i in range(10):
        res_n = {}
        gen = TimeseriesGenerator(data, targets,
                                    length=10, sampling_rate=2,
                                    batch_size=4, shuffle=True)
        assert len(gen) == 10

        data_batch_0 = gen[0]
        res_n['x_0'], res_n['y_0'] = data_batch_0
        results[str(i)] = res_n
    
    shuffling = [not (np.array_equal(results[i]['x_0'],
                                     np.array([[[0], [2], [4], [6], [8]],
                                               [[1], [3], [5], [7], [9]],
                                               [[2], [4], [6], [8], [10]],
                                               [[3], [5], [7], [9], [11]]]))
                      and np.array_equal(results[i]['y_0'],
                                         np.array([[10], [11], [12], [13]])))
                 for i in results]
    # Although possible, it is extremelly unlikelly that all 10 shuffles will result as if unshuffled.
    assert any(shuffling)

# @pytest.mark.skip
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

# @pytest.mark.skip
def test_TSG_generator_get_config():
    data = np.array([[i] for i in range(50)])
    targets = np.array([[i] for i in range(50)])

    data_gen = TimeseriesGenerator(data, targets,
                                length=10, sampling_rate=2,
                                batch_size=2)
    config = data_gen.get_config()
    # write_data_to_file('tests/helpers', 'dumped_expected_config.json', config)
    read_config = get_json_from_file('tests/helpers', 'dumped_expected_config.json')
    assert read_config == config

# @pytest.mark.skip
def test_TSG_generator_to_json():
    data = np.array([[i] for i in range(50)])
    targets = np.array([[i] for i in range(50)])

    data_gen = TimeseriesGenerator(data, targets,
                                length=10, sampling_rate=2,
                                batch_size=2)
    keras_data_gen = kerasGenerator(data, targets,
                                  length=10, sampling_rate=2,
                                  batch_size=2)
    ts_json = data_gen.to_json()

    # assert ts_json == keras_data_gen.to_json()
    read_ts_json = get_json_from_file('tests/helpers', 'dumped_default_tsg.json')
    # write_data_to_file('tests/helpers', 'dumped_default_tsg.json', data_gen)
    assert read_ts_json == json.loads(ts_json)

# @pytest.mark.skip
def test_TSG_generator_from_json():
    read_file= get_data_from_file('tests/helpers', 'dumped_default_tsg.json')
    data_gen = timeseries_generator_from_json(read_file)

    assert len(data_gen) == 20

    batch_0 = data_gen[0]
    x, y = batch_0
    assert np.array_equal(x,
                          np.array([[[0],
                                     [2],
                                     [4],
                                     [6],
                                     [8]],
                                    [[1],
                                     [3],
                                     [5],
                                     [7],
                                     [9]]]))
    assert np.array_equal(y,
                          np.array([[10],
                                    [11]]))

# @pytest.mark.skip
def test_Data_not_JSON_Serializable():
    data = np.array([datetime(1982, 12, 23), datetime(2009, 2, 4)])
    targets = np.array([[i] for i in range(2)])

    data_gen = TimeseriesGenerator(data, targets,
                                length=1, sampling_rate=1,
                                batch_size=1)
    with pytest.raises(TypeError) as ve:
        data_gen.get_config()

    assert str(ve.value) == "('Data not JSON Serializable:', [datetime.datetime(1982, 12, 23, 0, 0), datetime.datetime(2009, 2, 4, 0, 0)])"

# @pytest.mark.skip
def test_Targets_not_JSON_Serializable():
    data = np.array([[i] for i in range(2)])
    targets = np.array([datetime(1982, 12, 23), datetime(2009, 2, 4)])

    data_gen = TimeseriesGenerator(data, targets,
                                length=1, sampling_rate=1,
                                batch_size=1)
    with pytest.raises(TypeError) as ve:
        data_gen.get_config()

    assert str(ve.value) == "('Targets not JSON Serializable:', [datetime.datetime(1982, 12, 23, 0, 0), datetime.datetime(2009, 2, 4, 0, 0)])"

# @pytest.mark.skip
def test_univariate_multi_step_TSG_generator():
    expected_result = get_json_from_file('tests/helpers', 'expected_augmented_result.json')
    # define dataset
    series = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    target = np.array([[1,2],[2,3],[3,4],[4,5],[5,6],[6,7],[7,8],[8,9],[9,10],[10,11]])
    # define generator
    n_input = 2
    generator = TimeseriesGenerator(series, target, length=n_input, batch_size=1)
    # print each sample
    for i in range(len(generator)):
        x, y = generator[i]
        # expected[str(i)]=(x.tolist(), y.tolist())
        # print('%s => %s' % (x, y))
        assert np.all(x == expected_result[str(i)][0])
        assert np.all(y == expected_result[str(i)][1])
