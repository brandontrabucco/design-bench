from design_bench.oracles.tensorflow.tensorflow_oracle import TensorflowOracle
from design_bench.datasets.discrete_dataset import DiscreteDataset
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tempfile


class FullyConnectedOracle(TensorflowOracle):
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

    name = "tensorflow_fully_connected"

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

    @classmethod
    def check_input_format(cls, dataset):
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

    def fit(self, dataset, hidden_size=64, activation="relu",
            hidden_layers=2, epochs=10, shuffle_buffer=1000,
            learning_rate=0.0003, **kwargs):
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

        # obtain the expected shape of inputs to the model
        input_shape = dataset.input_shape
        if isinstance(dataset, DiscreteDataset) and dataset.is_logits:
            input_shape = input_shape[:-1]

        # build a model with an input layer and option embedding
        model_layers = [keras.Input(shape=input_shape)]
        if isinstance(dataset, DiscreteDataset):
            model_layers.append(
                layers.Embedding(dataset.num_classes, hidden_size))

        # add several fully connected layers and a final output layer
        model_layers.append(layers.Flatten())
        for i in range(hidden_layers):
            model_layers.append(
                layers.Dense(hidden_size, activation=activation))
            model_layers.append(layers.LayerNormalization())
        model_layers.append(layers.Dense(1))

        # build a sequential model and fit to a data generator
        model = keras.Sequential(model_layers)
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mse')
        model.fit(self.create_tensorflow_dataset(
            dataset, batch_size=self.internal_batch_size,
            shuffle_buffer=shuffle_buffer, repeat=epochs), **kwargs)

        # return the trained model
        return model

    def protected_predict(self, x):
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
