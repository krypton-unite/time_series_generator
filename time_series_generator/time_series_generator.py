import numpy as np
import json
import sys

class TimeseriesGenerator(object):
    """Utility class for generating batches of temporal data.

    This class takes in a sequence of data-points gathered at
    equal intervals, along with time series parameters such as
    stride, length of history, etc., to produce batches for
    training/validation.

    # Arguments
        data: Indexable generator (such as list or Numpy array)
            containing consecutive data points (timesteps).
            The data should be at 2D, and axis 0 is expected
            to be the time dimension.
        targets: Targets corresponding to timesteps in `data`.
            It should have same length as `data`.
        length: Length of the output sequences (in number of timesteps).
        sampling_rate: Period between successive individual timesteps
            within sequences. For rate `r`, timesteps
            `data[i]`, `data[i-r]`, ... `data[i - length]`
            are used for create a sample sequence.
        stride: Period between successive output sequences.
            For stride `s`, consecutive output samples would
            be centered around `data[i]`, `data[i+s]`, `data[i+2*s]`, etc.
        start_index: Data points earlier than `start_index` will not be used
            in the output sequences. This is useful to reserve part of the
            data for test or validation.
        end_index: Data points later than `end_index` will not be used
            in the output sequences. This is useful to reserve part of the
            data for test or validation.
        shuffle: Whether to shuffle output samples,
            or instead draw them in chronological order.
        reverse: Boolean: if `true`, timesteps in each output sample will be
            in reverse chronological order.
        batch_size: Number of timeseries samples in each batch
            (except maybe the last one).

    # Returns
        A [Sequence](/utils/#sequence) instance.

    # Examples

    ```python
    from keras.preprocessing.sequence import TimeseriesGenerator
    import numpy as np

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
    ```
    """

    def __init__(self, data, targets,
                 length,
                 sampling_rate=1,
                 length_output=1,
                 sampling_rate_output=1,
                 stride=1,
                 start_index=0,
                 end_index=None,
                 shuffle=False,
                 reverse=False,
                 batch_size=sys.maxsize,
                 augmentation=0,
                 overlap=0):

        if len(data) != len(targets):
            raise ValueError('Data and targets have to be' +
                             ' of same length. '
                             'Data length is {}'.format(len(data)) +
                             ' while target length is {}'.format(len(targets)))

        self.data = data
        self.targets = targets
        self.length = length
        self.length_output = length_output
        self.sampling_rate = sampling_rate
        self.sampling_rate_output = sampling_rate_output
        self.stride = stride
        self.start_index = start_index
        if end_index is None:
            end_index = len(data) - 1
        self.end_index = end_index
        self.shuffle = shuffle
        self.reverse = reverse
        self.batch_size = batch_size
        self.augmentation = augmentation
        self.overlap = overlap

        # the check below the way it was before didn't make sense since the generator might be used to represent only past data too.
        # Adding one to the right side of the comparison for that very reason!
        if self.start_index + length > self.end_index + 1:
            raise ValueError('`start_index+length=%i > end_index=%i` '
                             'is disallowed, as no part of the sequence '
                             'would be left to be used as current step.'
                             % (self.start_index + length, self.end_index))

    def __len__(self):
        if self.batch_size == sys.maxsize:
            return 1
        return int((self.end_index - self.start_index - self.length + 1 - self.length_output + self.overlap + self.augmentation)//(self.batch_size * self.stride)) + 1

    def __getitem__(self, index):
        i = self.start_index + self.length
        if index != 0:
            i = i + self.batch_size * self.stride * index
        rows = np.arange(
            i,
            min(
                i + self.batch_size * self.stride,
                self.end_index + 2 - self.length_output
            ),
            self.stride
        )
        if self.shuffle:
            np.random.shuffle(rows)

        samples = np.stack([self.data[row - self.length:row:self.sampling_rate]
                            for row in rows])
        if self.augmentation:
            augmented_rows = [row + np.random.randint(-self.augmentation, self.augmentation+1) for row in rows]
        else:
            augmented_rows = rows
        targets = np.stack([
            self.targets[
                row - self.overlap : row + self.length_output : self.sampling_rate_output
            ] for row in augmented_rows
        ])

        if targets.shape[1] == 1:
            targets = targets.squeeze(1)

        if self.reverse:
            return samples[:, ::-1, ...], targets
        return samples, targets

    def get_config(self):
        '''Returns the TimeseriesGenerator configuration as Python dictionary.

        # Returns
            A Python dictionary with the TimeseriesGenerator configuration.
        '''
        data = self.data
        if type(self.data).__module__ == np.__name__:
            data = self.data.tolist()
        try:
            json_data = json.dumps(data)
        except TypeError:
            raise TypeError('Data not JSON Serializable:', data)

        targets = self.targets
        if type(self.targets).__module__ == np.__name__:
            targets = self.targets.tolist()
        try:
            json_targets = json.dumps(targets)
        except TypeError:
            raise TypeError('Targets not JSON Serializable:', targets)

        return {
            'data': json_data,
            'targets': json_targets,
            'length': self.length,
            'length_output': self.length_output,
            'sampling_rate': self.sampling_rate,
            'sampling_rate_output': self.sampling_rate_output,
            'stride': self.stride,
            'start_index': self.start_index,
            'end_index': self.end_index,
            'shuffle': self.shuffle,
            'reverse': self.reverse,
            'batch_size': self.batch_size,
            'augmentation': self.augmentation,
            'overlap': self.overlap,
        }

    def to_json(self, **kwargs):
        """Returns a JSON string containing the timeseries generator
        configuration. To load a generator from a JSON string, use
        `keras.preprocessing.sequence.timeseries_generator_from_json(json_string)`.

        # Arguments
            **kwargs: Additional keyword arguments
                to be passed to `json.dumps()`.

        # Returns
            A JSON string containing the tokenizer configuration.
        """
        config = self.get_config()
        timeseries_generator_config = {
            'class_name': self.__class__.__name__,
            'config': config
        }
        return json.dumps(timeseries_generator_config, **kwargs)


def timeseries_generator_from_json(json_string):
    """Parses a JSON timeseries generator configuration file and
    returns a timeseries generator instance.

    # Arguments
        json_string: JSON string encoding a timeseries
            generator configuration.

    # Returns
        A Keras TimeseriesGenerator instance
    """
    full_config = json.loads(json_string)
    config = full_config.get('config')

    data = json.loads(config.pop('data'))
    config['data'] = data
    targets = json.loads(config.pop('targets'))
    config['targets'] = targets

    return TimeseriesGenerator(**config)
