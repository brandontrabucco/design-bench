import abc


class FeatureExtractor(abc.ABC):
    """An abstract class for managing transformations applied to model-based
    optimization datasets when constructing the oracle; for example, if the
    oracle is intended to learn from molecule fingerprints

    max_x { y = f(x) }

    Public Methods:

    dataset_to_oracle_x(np.ndarray) -> np.ndarray
        Helper function for converting from designs contained in the
        dataset format into a format the oracle is expecting to process,
        such as from integers to logits of a categorical distribution

    dataset_to_oracle_y(np.ndarray) -> np.ndarray
        Helper function for converting from predictions contained in the
        dataset format into a format the oracle is expecting to process,
        such as from normalized to denormalized predictions

    oracle_to_dataset_x(np.ndarray) -> np.ndarray
        Helper function for converting from designs in the format of the
        oracle into the design format the dataset contains, such as
        from categorical logits to integers

    oracle_to_dataset_y(np.ndarray) -> np.ndarray
        Helper function for converting from predictions in the
        format of the oracle into a format the dataset contains,
        such as from normalized to denormalized predictions

    """

    name = "feature_extractor"

    @abc.abstractmethod
    def dataset_to_oracle_x(self, x_batch, dataset):
        """Helper function for converting from designs contained in the
        dataset format into a format the oracle is expecting to process,
        such as from integers to logits of a categorical distribution

        Arguments:

        x_batch: np.ndarray
            a batch of design values 'x' that will be given as input to the
            oracle model in order to obtain a prediction value 'y' for
            each 'x' which is then returned
        dataset: DatasetBuilder
            an instance of a subclass of the DatasetBuilder class representing
            the source of the batch, must be provided

        Returns:

        x_batch: np.ndarray
            a batch of design values 'x' that have been converted from the
            format of designs contained in the dataset to the format
            expected by the oracle score function

        """

        raise NotImplementedError("cannot run base class")

    @abc.abstractmethod
    def dataset_to_oracle_y(self, y_batch, dataset):
        """Helper function for converting from predictions contained in the
        dataset format into a format the oracle is expecting to process,
        such as from normalized to denormalized predictions

        Arguments:

        y_batch: np.ndarray
            a batch of prediction values 'y' that are from the dataset and
            will be processed into a format expected by the oracle score
            function, which is useful when training the oracle
        dataset: DatasetBuilder
            an instance of a subclass of the DatasetBuilder class representing
            the source of the batch, must be provided

        Returns:

        y_batch: np.ndarray
            a batch of prediction values 'y' that have been converted from
            the format of predictions contained in the dataset to the
            format expected by the oracle score function

        """

        raise NotImplementedError("cannot run base class")

    @abc.abstractmethod
    def oracle_to_dataset_x(self, x_batch, dataset):
        """Helper function for converting from designs in the format of the
        oracle into the design format the dataset contains, such as
        from categorical logits to integers

        Arguments:

        x_batch: np.ndarray
            a batch of design values 'x' that have been converted from
            the format of designs contained in the dataset to the
            format expected by the oracle score function
        dataset: DatasetBuilder
            an instance of a subclass of the DatasetBuilder class representing
            the source of the batch, must be provided

        Returns:

        x_batch: np.ndarray
            a batch of design values 'x' that have been converted from
            the format of the oracle to the format of designs
            contained in the dataset

        """

        raise NotImplementedError("cannot run base class")

    @abc.abstractmethod
    def oracle_to_dataset_y(self, y_batch, dataset):
        """Helper function for converting from predictions in the format
        of the oracle into a format the dataset contains, such as
        from normalized to denormalized predictions

        Arguments:

        y_batch: np.ndarray
            a batch of prediction values 'y' that have been converted from
            the format of predictions contained in the dataset to the
            format expected by the oracle score function
        dataset: DatasetBuilder
            an instance of a subclass of the DatasetBuilder class representing
            the source of the batch, must be provided

        Returns:

        y_batch: np.ndarray
            a batch of prediction values 'y' that have been converted from
            the format of the oracle to the format of predictions
            contained in the dataset

        """

        raise NotImplementedError("cannot run base class")

    @abc.abstractmethod
    def input_shape(self, dataset):
        """Helper function for converting from predictions in the format
        of the oracle into a format the dataset contains, such as
        from normalized to denormalized predictions

        Arguments:

        dataset: DatasetBuilder
            an instance of a subclass of the DatasetBuilder class representing
            the source of the batch, must be provided

        Returns:

        input_shape: List[int]
            the shape of input tensors that were sampled from the dataset and
            are transformed into features using subclasses of this class

        """

        raise NotImplementedError("cannot run base class")

    @abc.abstractmethod
    def input_dtype(self, dataset):
        """Helper function that returns the data type of the features returned
        by running the feature extractor from dataset to oracle

        Arguments:

        dataset: DatasetBuilder
            an instance of a subclass of the DatasetBuilder class representing
            the source of the batch, must be provided

        Returns:

        input_dtype: List[int]
            the type of input tensors that were sampled from the dataset and
            are transformed into features using subclasses of this class

        """

        raise NotImplementedError("cannot run base class")

    @abc.abstractmethod
    def is_discrete(self, dataset):
        """Helper function that specifies whether the transformation applied
        by the feature extractor returns a discrete or continuous set of
        features, which is required for building predictive models

        Arguments:

        dataset: DatasetBuilder
            an instance of a subclass of the DatasetBuilder class representing
            the source of the batch, must be provided

        Returns:

        is_discrete: bool
            a boolean that indicates whether the dataset has been transformed
            into a discrete or continuous representation

        """

        raise NotImplementedError("cannot run base class")

    @abc.abstractmethod
    def num_classes(self, dataset):
        """Helper function for determining the number of classes in the discrete
        representation intended for the oracle, if it is discrete, otherwise
        this function may not be implemented and will raise an error

        Arguments:

        dataset: DatasetBuilder
            an instance of a subclass of the DatasetBuilder class representing
            the source of the batch, must be provided

        Returns:

        num_classes: int
            the number of classes in the discrete representation for the model
            based optimization dataset used for training the oracle

        """

        raise NotImplementedError("cannot run base class")
