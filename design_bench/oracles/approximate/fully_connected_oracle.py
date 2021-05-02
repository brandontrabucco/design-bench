from design_bench.oracles.approximate_oracle import ApproximateOracle
from design_bench.datasets.discrete_dataset import DiscreteDataset
from design_bench.datasets.continuous_dataset import ContinuousDataset
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tempfile


class FullyConnectedOracle(ApproximateOracle):
    """An abstract class for managing the ground truth score functions f(x)
    for model-based optimization problems, where the
    goal is to find a design 'x' that maximizes a prediction 'y':

    max_x { y = f(x) }

    Public Attributes:

    dataset: DatasetBuilder
        an instance of a subclass of the DatasetBuilder class which has
        a set of design values 'x' and prediction values 'y', and defines
        batching and sampling methods for those attributes

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

    score(np.ndarray) -> np.ndarray:
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

    name = "fully_connected"

    def __init__(self, dataset, noise_std=0.0, **kwargs):
        """Initialize the ground truth score function f(x) for a model-based
        optimization problem, which involves loading the parameters of an
        oracle model and estimating its computational cost

        Arguments:

        dataset: DiscreteDataset
            an instance of a subclass of the DatasetBuilder class which has
            a set of design values 'x' and prediction values 'y', and defines
            batching and sampling methods for those attributes
        noise_std: float
            the standard deviation of gaussian noise added to the prediction
            values 'y' coming out of the ground truth score function f(x)
            in order to make the optimization problem difficult

        """

        # initialize the oracle using the super class
        super(FullyConnectedOracle, self).__init__(
            dataset, noise_std=noise_std, is_batched=True,
            internal_batch_size=32, internal_measurements=1,
            expect_normalized_y=True,
            expect_normalized_x=not isinstance(dataset, DiscreteDataset),
            expect_logits=False if isinstance(
                dataset, DiscreteDataset) else None, **kwargs)

    @staticmethod
    def check_input_format(dataset):
        """a function that accepts a model-based optimization dataset as input
        and determines whether the provided dataset is compatible with this
        oracle score function (is this oracle a correct one)

        Arguments:

        dataset: DatasetBuilder
            an instance of a subclass of the DatasetBuilder class which has
            a set of design values 'x' and prediction values 'y', and defines
            batching and sampling methods for those attributes

        Returns:

        is_compatible: bool
            a boolean indicator that is true when the specified dataset is
            compatible with this ground truth score function

        """

        return True  # any data set is always supported with this model

    def save_model_to_zip(self, model, zip_archive):
        """a function that serializes a machine learning model and stores
        that model in a compressed zip file using the python ZipFile interface
        for sharing and future loading by an ApproximateOracle

        Arguments:

        model: Any
            any format of of machine learning model that will be stored
            in the self.model attribute for later use

        zip_archive: ZipFile
            an instance of the python ZipFile interface that has loaded
            the file path specified by self.resource.disk_target

        """

        with tempfile.NamedTemporaryFile() as file:
            model.save(file.name, save_format='h5')
            model_bytes = file.read()
        with zip_archive.open('fully_connected.h5', "w") as file:
            file.write(model_bytes)  # save model bytes in the h5 format

    def load_model_from_zip(self, zip_archive):
        """a function that loads components of a serialized model from a zip
        given zip file using the python ZipFile interface and returns an
        instance of the model

        Arguments:

        zip_archive: ZipFile
            an instance of the python ZipFile interface that has loaded
            the file path specified by self.resource.disk_target

        Returns:

        model: Any
            any format of of machine learning model that will be stored
            in the self.model attribute for later use

        """

        with zip_archive.open('fully_connected.h5', "r") as file:
            model_bytes = file.read()  # read model bytes in the h5 format
        with tempfile.NamedTemporaryFile() as file:
            file.write(model_bytes)
            return keras.models.load_model(file.name)

    def create_tensorflow_dataset(self, dataset, batch_size=32,
                                  shuffle_buffer=1000, repeat=10):
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
        if isinstance(dataset, DiscreteDataset) and dataset.is_logits:
            input_shape = input_shape[:-1]

        # map from the dataset format to the oracle format using numpy
        def process(x, y):
            return self.dataset_to_oracle_x(x), self.dataset_to_oracle_y(y)

        # map from the dataset format to the oracle format using tensorflow
        def process_tf(x, y):
            dtype = tf.int32 if isinstance(dataset, DiscreteDataset) \
                             else tf.float32
            x, y = tf.numpy_function(process, (x, y), (dtype, tf.float32))
            x.set_shape([None, *input_shape])
            y.set_shape([None, 1])
            return x, y

        # create a dataset from individual samples
        tf_dataset = tf.data.Dataset.from_generator(
            dataset.iterate_samples,
            (dataset.input_dtype, dataset.output_dtype),
            (tf.TensorShape(dataset.input_shape),
             tf.TensorShape(dataset.output_shape)))

        # batch and repeat the dataset
        tf_dataset = tf_dataset.shuffle(shuffle_buffer)
        tf_dataset = tf_dataset.batch(batch_size)
        tf_dataset = tf_dataset.repeat(repeat)
        auto = tf.data.experimental.AUTOTUNE
        tf_dataset = tf_dataset.map(process_tf, num_parallel_calls=auto)
        return tf_dataset.prefetch(auto)

    def fit(self, dataset, hidden_size=64,
            activation="relu", hidden_layers=2, **kwargs):
        """a function that accepts a set of design values 'x' and prediction
        values 'y' and fits an approximate oracle to serve as the ground
        truth function f(x) in a model-based optimization problem

        Arguments:

        dataset: DatasetBuilder
            an instance of a subclass of the DatasetBuilder class which has
            a set of design values 'x' and prediction values 'y', and defines
            batching and sampling methods for those attributes

        Returns:

        model: Any
            any format of of machine learning model that will be stored
            in the self.model attribute for later use

        """

        input_shape = dataset.input_shape
        if isinstance(dataset, DiscreteDataset) and dataset.is_logits:
            input_shape = input_shape[:-1]

        tf_dataset = self.create_tensorflow_dataset(
            dataset, batch_size=32, shuffle_buffer=1000, repeat=10)

        model_layers = [keras.Input(shape=input_shape)]
        if isinstance(dataset, DiscreteDataset):
            model_layers.append(
                layers.Embedding(dataset.num_classes, hidden_size))

        model_layers.append(layers.Flatten())
        for i in range(hidden_layers):
            model_layers.append(
                layers.Dense(hidden_size, activation=activation))
        model_layers.append(layers.Dense(1))

        model = keras.Sequential(model_layers)
        model.compile(optimizer='adam', loss='mse')
        model.fit(tf_dataset, **kwargs)

        # cleanup the dataset and return the trained model
        return model

    def protected_score(self, x):
        """Score function to be implemented by oracle subclasses, where x is
        either a batch of designs if self.is_batched is True or is a
        single design when self._is_batched is False

        Arguments:

        x_batch: np.ndarray
            a batch or single design 'x' that will be given as input to the
            oracle model in order to obtain a prediction value 'y' for
            each 'x' which is then returned

        Returns:

        y_batch: np.ndarray
            a batch or single prediction 'y' made by the oracle model,
            corresponding to the ground truth score for each design
            value 'x' in a model-based optimization problem

        """

        # call the model's predict function to generate predictions
        return self.model.predict(x)
