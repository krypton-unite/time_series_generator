from time_series_generator import TimeseriesGenerator, timeseries_generator_from_json
from .helpers import get_json_from_file, get_data_from_file
import numpy as np
import pytest
from .fixtures import expected_msr_result


# @pytest.mark.skip
def test_augmented_univariate_multi_step_TSG_generator():
    expected_result = get_json_from_file(
        'tests/helpers', 'expected_augmented_result.json')
    # define dataset
    series = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    target = series
    # define generator
    n_input = 2
    n_output = 2
    generator = TimeseriesGenerator(
        series, target, length=n_input, length_output=n_output, batch_size=1)
    # print each sample
    for i in range(len(generator)):
        x, y = generator[i]
        print('%s => %s' % (x, y))
        # expected[str(i)]=(x.tolist(), y.tolist())
        assert np.all(x == np.array(expected_result[str(i)][0]))
        assert np.all(y == np.array(expected_result[str(i)][1]))

# @pytest.mark.skip
@pytest.mark.usefixtures("expected_msr_result")
@pytest.mark.parametrize('length_output', [2, 3])
@pytest.mark.parametrize('sampling_rate_output', [1, 2])
def test_default_TSG_generator(length_output, sampling_rate_output, expected_msr_result):
    data = np.array([[i] for i in range(50)])
    targets = data

    data_gen = TimeseriesGenerator(data,
                                targets,
                                length=5,
                                length_output=sampling_rate_output*length_output,
                                sampling_rate=1,
                                sampling_rate_output=sampling_rate_output,
                                stride=10,
                                batch_size=2)

    expected_result = expected_msr_result(length_output, sampling_rate_output)
    batch_0 = data_gen[0]
    x, y = batch_0
    assert np.array_equal(x,expected_result['x_0'])
    assert np.array_equal(y,expected_result['y_0'])

    batch_1 = data_gen[1]
    x, y = batch_1
    assert np.array_equal(x,expected_result['x_1'])
    assert np.array_equal(y,expected_result['y_1'])

