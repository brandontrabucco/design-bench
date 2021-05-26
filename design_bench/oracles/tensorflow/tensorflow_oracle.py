from design_bench.oracles.approximate_oracle import ApproximateOracle
from design_bench.datasets.discrete_dataset import DiscreteDataset
import tensorflow as tf
import abc
for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)


class TensorflowOracle(ApproximateOracle, abc.ABC):
    """An abstract class for managing the ground truth score functions f(x)
    for model-based optimization problems, where the
    goal is to find a design 'x' that maximizes a prediction 'y':

    max_x { y = f(x) }

    Public Attributes:

    external_dataset: DatasetBuilder
        an instance of a subclass of the DatasetBuilder class which points to
        the mutable task dataset for a model-based optimization problem

    internal_dataset: DatasetBuilder
        an instance of a subclass of the DatasetBuilder class which has frozen
        statistics and is used for training the oracle

    is_batched: bool
        a boolean variable that indicates whether the evaluation function
        implemented for a particular oracle is batched, which effects
        the scaling coefficient of its computational cost

    internal_batch_size: int
        an integer representing the number of design values to process
        internally at the same time, if None defaults to the entire
        tensor given to the self.score method
    internal_measurements: int
        an integer representing the number of independent measurements of
        the prediction made by the oracle, which are subsequently
        averaged, and is useful when the oracle is stochastic

    noise_std: float
        the standard deviation of gaussian noise added to the prediction
        values 'y' coming out of the ground truth score function f(x)
        in order to make the optimization problem difficult

    expect_normalized_y: bool
        a boolean indicator that specifies whether the inputs to the oracle
        score function are expected to be normalized
    expect_normalized_x: bool
        a boolean indicator that specifies whether the outputs of the oracle
        score function are expected to be normalized
    expect_logits: bool
        a boolean that specifies whether the oracle score function is
        expecting logits when the dataset is discrete

    Public Methods:

    predict(np.ndarray) -> np.ndarray:
        a function that accepts a batch of design values 'x' as input and for
        each design computes a prediction value 'y' which corresponds
        to the score in a model-based optimization problem

    check_input_format(DatasetBuilder) -> bool:
        a function that accepts a list of integers as input and returns true
        when design values 'x' with the shape specified by that list are
        compatible with this class of approximate oracle

    fit(np.ndarray, np.ndarray):
        a function that accepts a data set of design values 'x' and prediction
        values 'y' and fits an approximate oracle to serve as the ground
        truth function f(x) in a model-based optimization problem

    """

    def create_tensorflow_dataset(self, dataset, batch_size=32,
                                  shuffle_buffer=1000, repeat=1):
        """Helper function that converts a model-based optimization dataset
        into a tensorflow dataset using the tf.data.Dataset API

        Arguments:

        dataset: DatasetBuilder
            an instance of a subclass of the DatasetBuilder class which has
            a set of design values 'x' and prediction values 'y', and defines
            batching and sampling methods for those attributes
        batch_size: int
            an integer passed to tf.data.Dataset.batch that determines
            the number of samples per batch
        shuffle_buffer: int
            an integer passed to tf.data.Dataset.shuffle that determines
            the number of samples to load before shuffling
        repeat: int
            an integer passed to tf.data.Dataset.repeat that determines
            the number of epochs the dataset repeats for

        Returns:

        tf_dataset: tf.data.Dataset
            a dataset using the tf.data.Dataset API that samples from a
            model-based optimization dataset

        """

        # obtain the expected shape of samples to the model
        input_shape = dataset.input_shape
        if isinstance(dataset,
                      DiscreteDataset) and dataset.is_logits:
            input_shape = input_shape[:-1]

        # map from dataset format to oracle format using numpy
        def dataset_to_oracle_numpy(x, y):
            return self.dataset_to_oracle_x(x, dataset=dataset), \
                   self.dataset_to_oracle_y(y, dataset=dataset)

        # map from dataset format to oracle format using tensorflow
        def dataset_to_oracle_tensorflow(x, y):
            dtype = tf.int32 if isinstance(
                dataset, DiscreteDataset) else tf.float32

            # process the input tensors using numpy
            x, y = tf.numpy_function(dataset_to_oracle_numpy,
                                     (x, y), (dtype, tf.float32))

            # add shape information for the returned tensors
            x.set_shape([None, *input_shape])
            y.set_shape([None, 1])
            return x, y

        # create a dataset from individual samples
        generator = tf.data.Dataset.from_generator(
            dataset.iterate_samples,
            (dataset.input_dtype, dataset.output_dtype),
            (tf.TensorShape(dataset.input_shape),
             tf.TensorShape(dataset.output_shape)))

        # create randomly shuffled batches
        generator = generator.shuffle(shuffle_buffer)
        generator = generator.batch(batch_size)
        generator = generator.repeat(repeat)

        # update the dataset format and dtype the return
        auto = tf.data.experimental.AUTOTUNE
        return generator.map(dataset_to_oracle_tensorflow,
                             num_parallel_calls=auto).prefetch(auto)
