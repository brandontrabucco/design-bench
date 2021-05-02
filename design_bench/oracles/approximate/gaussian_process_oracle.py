from design_bench.oracles.approximate_oracle import ApproximateOracle
from design_bench.datasets.discrete_dataset import DiscreteDataset
from design_bench.datasets.dataset_builder import DatasetBuilder
from sklearn.gaussian_process import GaussianProcessRegressor
import numpy as np
import pickle as pkl


class GaussianProcessOracle(ApproximateOracle):
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

    name = "gaussian_process"

    def __init__(self, dataset: DatasetBuilder, noise_std=0.0, **kwargs):
        """Initialize the ground truth score function f(x) for a model-based
        optimization problem, which involves loading the parameters of an
        oracle model and estimating its computational cost

        Arguments:

        dataset: DatasetBuilder
            an instance of a subclass of the DatasetBuilder class which has
            a set of design values 'x' and prediction values 'y', and defines
            batching and sampling methods for those attributes
        noise_std: float
            the standard deviation of gaussian noise added to the prediction
            values 'y' coming out of the ground truth score function f(x)
            in order to make the optimization problem difficult

        """

        # initialize the oracle using the super class
        super(GaussianProcessOracle, self).__init__(
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

        with zip_archive.open('gaussian_process.pkl', "w") as file:
            return pkl.dump(model, file)  # save the model using pickle

    def load_model_from_zip(self, zip_archive):
        """a function that loads components of a serialized model from a zip
        given zip file using the python ZipFile interface and returns an
        instance of the model

        Arguments:

        zip_archive: ZipFile
            an instance of the python ZipFile interface that has loaded
            the file path specified by self.resource.disk_targetteh

        Returns:

        model: Any
            any format of of machine learning model that will be stored
            in the self.model attribute for later use

        """

        with zip_archive.open('gaussian_process.pkl', "r") as file:
            return pkl.load(file)  # load the random forest using pickle

    def fit(self, dataset, max_samples=1000,
            min_percentile=0.0, max_percentile=100.0, **kwargs):
        """a function that accepts a set of design values 'x' and prediction
        values 'y' and fits an approximate oracle to serve as the ground
        truth function f(x) in a model-based optimization problem

        Arguments:

        dataset: DatasetBuilder
            an instance of a subclass of the DatasetBuilder class which has
            a set of design values 'x' and prediction values 'y', and defines
            batching and sampling methods for those attributes
        kernel: Kernel or np.ndarray
            an instance of an sklearn Kernel if the dataset is continuous or
            an instance of a numpy array if the dataset is discrete, which
            will be passed to an instance of DiscreteSequenceKernel
        max_samples: int
            the maximum number of samples to be used when fitting a gaussian
            process, where the dataset is uniformly randomly sub sampled
            if the dataset is larger than max_samples

        Returns:

        model: Any
            any format of of machine learning model that will be stored
            in the self.model attribute for later use

        """

        # build the model class and assign hyper parameters
        model = GaussianProcessRegressor(**kwargs)

        # sample the entire dataset without transformations
        # note this requires the dataset to be loaded in memory all at once
        dataset._disable_subsample = True
        x = dataset.x
        y = dataset.y
        dataset._disable_subsample = False

        # select training examples using percentile sub sampling
        # necessary when the training set is too large for the model to fit
        indices = self.get_indices(y, max_samples=max_samples,
                                   min_percentile=min_percentile,
                                   max_percentile=max_percentile)
        x = x[indices]
        y = y[indices]

        # convert samples into the expected format of the oracle
        x = self.dataset_to_oracle_x(x)
        y = self.dataset_to_oracle_y(y)

        # fit the random forest model to the dataset
        model.fit(x.reshape((x.shape[0], np.prod(x.shape[1:]))),
                  y.reshape((y.shape[0],)))

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
        return self.model.predict(
            x.reshape((x.shape[0], np.prod(x.shape[1:]))))[:, np.newaxis]
