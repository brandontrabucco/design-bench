from design_bench.oracles.oracle_builder import OracleBuilder
from design_bench.datasets.dataset_builder import DatasetBuilder
from design_bench.disk_resource import DiskResource
import abc
import zipfile


class ApproximateOracle(OracleBuilder, abc.ABC):
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

    @abc.abstractmethod
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

        raise NotImplementedError

    @abc.abstractmethod
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

        raise NotImplementedError

    @abc.abstractmethod
    def fit(self, dataset, **kwargs):
        """a function that accepts a set of design values 'x' and prediction
        values 'y' and fits an approximate oracle to serve as the ground
        truth function f(x) in a model-based optimization problem

        Arguments:

        dataset: DatasetBuilder
            an instance of a subclass of the DatasetBuilder class which has
            a set of design values 'x' and prediction values 'y', and defines
            batching and sampling methods for those attributes

        """

        raise NotImplementedError

    def __init__(self, dataset: DatasetBuilder,
                 file=None, is_absolute=False, is_batched=True,
                 internal_batch_size=32, internal_measurements=1,
                 noise_std=0.0, expect_normalized_y=False,
                 expect_normalized_x=False, expect_logits=None, **kwargs):
        """Initialize the ground truth score function f(x) for a model-based
        optimization problem, which involves loading the parameters of an
        oracle model and estimating its computational cost

        Arguments:

        dataset: DatasetBuilder
            an instance of a subclass of the DatasetBuilder class which has
            a set of design values 'x' and prediction values 'y', and defines
            batching and sampling methods for those attributes
        file: str
            a path to a zip file that would contain a serialized model, and is
            useful when there are multiple versions of the same model
        is_absolute: bool
            a boolean that indicates whether the provided disk_target path is
            an absolute path or relative to the data folder
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
            a boolean indicator that specifies whether the inputs to the
            oracle score function are expected to be normalized
        expect_normalized_x: bool
            a boolean indicator that specifies whether the outputs of the
            oracle score function are expected to be normalized
        expect_logits: bool
            a boolean that specifies whether the oracle score function
            is expecting logits when the dataset is discrete

        """

        # initialize the oracle using the super class
        super(ApproximateOracle, self).__init__(
            dataset, is_batched=is_batched,
            internal_batch_size=internal_batch_size,
            internal_measurements=internal_measurements,
            noise_std=noise_std,
            expect_normalized_y=expect_normalized_y,
            expect_normalized_x=expect_normalized_x,
            expect_logits=expect_logits)

        # download the model parameters from s3
        self.resource = self.get_disk_resource(
            dataset, file=file, is_absolute=is_absolute)
        if not self.resource.is_downloaded \
                and not self.resource.download(unzip=False):
            self.save_model(self.resource.disk_target,
                            self.fit(dataset, **kwargs))

        # load the model from disk once its downloaded
        self.model = self.load_model(self.resource.disk_target)

    def get_disk_resource(self, dataset, file=None, is_absolute=False):
        """a function that returns a zip file containing all the files and
        meta data required to re-build an oracle model such as a neural
        network or random forest regression model

        Arguments:

        dataset: DatasetBuilder
            an instance of a subclass of the DatasetBuilder class which has
            a set of design values 'x' and prediction values 'y', and defines
            batching and sampling methods for those attributes
        file: str
            a path to a zip file that would contain a serialized model, and is
            useful when there are multiple versions of the same model
        is_absolute: bool
            a boolean that indicates whether the provided disk_target path is
            an absolute path or relative to the data folder

        Returns:

        model_resource: DiskResource
            a DiskResource instance containing the expected download link of the
            model and the target location on disk where the model is
            expected to be located or downloaded to

        """

        default = f"{dataset.name}/{self.name}.zip"
        return DiskResource(
            file if file is not None else default,
            is_absolute=is_absolute,
            download_method=None if file is not None else "direct",
            download_target=None if file is not None else
            f"https://design-bench.s3-us-west-1.amazonaws.com/{default}")

    def save_model(self, file, model):
        """a function that serializes a machine learning model and stores
        that model in a compressed zip file using the python ZipFile interface
        for sharing and future loading by an ApproximateOracle

        Arguments:

        file: str
            a path to a zip file that would contain a serialized model, and is
            useful when there are multiple versions of the same model
        model: Any
            any format of of machine learning model that will be stored
            in the self.model attribute for later use

        """

        with zipfile.ZipFile(file, mode="w") as file:
            self.save_model_to_zip(model, file)

    def load_model(self, file):
        """a function that loads components of a serialized model from a zip
        given zip file using the python ZipFile interface and returns an
        instance of the model

        Arguments:

        file: str
            a path to a zip file that would contain a serialized model, and is
            useful when there are multiple versions of the same model

        Returns:

        model: Any
            any format of of machine learning model that will be stored
            in the self.model attribute for later use

        """

        with zipfile.ZipFile(file, mode="r") as file:
            return self.load_model_from_zip(file)
