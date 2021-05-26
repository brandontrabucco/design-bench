from design_bench.oracles.tensorflow.tensorflow_oracle import TensorflowOracle
from design_bench.datasets.discrete_dataset import DiscreteDataset
import tensorflow as tf
import tensorflow.keras as keras
import tempfile
import math
import numpy as np
import shutil


import transformers
from transformers import TFBertForSequenceClassification as TFBert


class TransformerOracle(TensorflowOracle):
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

    name = "tensorflow_transformer"
    default_model_kwargs = dict(hidden_size=256, feed_forward_size=256,
                                activation='relu', num_heads=8,
                                num_blocks=4, epochs=20,
                                shuffle_buffer=5000, learning_rate=0.0001,
                                warm_up_steps=4000, dropout_rate=0.1)

    def __init__(self, dataset, **kwargs):
        """Initialize the ground truth score function f(x) for a model-based
        optimization problem, which involves loading the parameters of an
        oracle model and estimating its computational cost

        Arguments:

        dataset: DiscreteDataset
            an instance of a subclass of the DatasetBuilder class which has
            a set of design values 'x' and prediction values 'y', and defines
            batching and sampling methods for those attributes

        """

        # initialize the oracle using the super class
        super(TransformerOracle, self).__init__(
            dataset, is_batched=True, internal_measurements=1,
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

        # ensure that the data has exactly one sequence dimension
        if isinstance(dataset, DiscreteDataset) and not dataset.is_logits:
            return len(dataset.input_shape) == 1
        return len(dataset.input_shape) == 2

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

        # extract the bytes of the hugging face save path as a zip
        with tempfile.TemporaryDirectory() as directory:
            with tempfile.NamedTemporaryFile(suffix=".zip") as archive:
                model.save_pretrained(directory)
                shutil.make_archive(archive.name[:-4], 'zip', directory)
                model_bytes = archive.read()

        # write the h5 bytes ot the zip file
        with zip_archive.open('transformer.zip', "w") as file:
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

        # read the h5 bytes from the zip file
        with zip_archive.open('transformer.zip', "r") as file:
            model_bytes = file.read()  # read model bytes in the h5 format

        # load the bytes of the hugging face save path from a zip
        with tempfile.TemporaryDirectory() as directory:
            with tempfile.NamedTemporaryFile(suffix=".zip") as archive:
                archive.write(model_bytes)
                shutil.unpack_archive(archive.name, directory)
                return TFBert.from_pretrained(directory)

    def protected_fit(self, training, validation, model_kwargs=None):
        """a function that accepts a training dataset and a validation dataset
        containing design values 'x' and prediction values 'y' in a model-based
        optimization problem and fits an approximate model

        Arguments:

        training: DatasetBuilder
            an instance of a subclass of the DatasetBuilder class which has
            a set of design values 'x' and prediction values 'y', and defines
            batching and sampling methods for those attributes
        validation: DatasetBuilder
            an instance of a subclass of the DatasetBuilder class which has
            a set of design values 'x' and prediction values 'y', and defines
            batching and sampling methods for those attributes
        model_kwargs: dict
            a dictionary of keyword arguments that parameterize the
            architecture and learning algorithm of the model

        Returns:

        model: Any
            any format of of machine learning model that will be stored
            in the self.params["model"] attribute for later use

        """

        # these parameters control the neural network architecture
        hidden_size = model_kwargs["hidden_size"]
        num_heads = model_kwargs["num_heads"]
        dropout_rate = model_kwargs["dropout_rate"]
        feed_forward_size = model_kwargs["feed_forward_size"]
        activation = model_kwargs["activation"]
        num_blocks = model_kwargs["num_blocks"]

        # these parameters control the model training
        epochs = model_kwargs["epochs"]
        shuffle_buffer = model_kwargs["shuffle_buffer"]
        learning_rate = model_kwargs["learning_rate"]

        # build the hugging face model from a configuration
        model = TFBert(transformers.BertConfig(
            vocab_size=training.num_classes,
            num_labels=1,
            hidden_size=hidden_size,
            num_hidden_layers=num_blocks,
            num_attention_heads=num_heads,
            intermediate_size=feed_forward_size,
            hidden_act=activation,
            hidden_dropout_prob=dropout_rate,
            attention_probs_dropout_prob=dropout_rate,
            max_position_embeddings=training.input_shape[0],
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            position_embedding_type='absolute'))

        # estimate the number of training steps per epoch
        steps = int(math.ceil(training.dataset_size
                              / self.internal_batch_size))

        # compile the tensorflow model for training
        lr = keras.experimental.CosineDecay(
            learning_rate, steps * epochs, alpha=0.0)
        optimizer = keras.optimizers.Adam(learning_rate=lr)
        model.compile(optimizer=optimizer,
                      loss=tf.keras.losses.MeanSquaredError())

        # an input key for the huggingface transformer api
        input_key = "input_ids" if isinstance(
            training, DiscreteDataset) else "inputs_embeds"

        # create a tensorflow dataset generator for training
        training = self.create_tensorflow_dataset(
            training, batch_size=self.internal_batch_size,
            shuffle_buffer=shuffle_buffer, repeat=epochs)

        # create a tensorflow dataset generator for validation
        validation = self.create_tensorflow_dataset(
            validation, batch_size=self.internal_batch_size,
            shuffle_buffer=self.internal_batch_size, repeat=1)

        # convert to the huggingface transformer input format
        training = training.map(
            lambda x, y: ({input_key: x}, y),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)

        validation = validation.map(
            lambda x, y: ({input_key: x}, y),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # fit the model to a tensorflow dataset
        model.fit(training, steps_per_epoch=steps,
                  epochs=epochs, validation_data=validation)

        # return the trained model and rank correlation
        return model

    def protected_predict(self, x, model=None):
        """Score function to be implemented by oracle subclasses, where x is
        either a batch of designs if self.is_batched is True or is a
        single design when self._is_batched is False

        Arguments:

        x_batch: np.ndarray
            a batch or single design 'x' that will be given as input to the
            oracle model in order to obtain a prediction value 'y' for
            each 'x' which is then returned
        model: Any
            any format of of machine learning model that will be stored
            in the self.params["model"] attribute for later use

        Returns:

        y_batch: np.ndarray
            a batch or single prediction 'y' made by the oracle model,
            corresponding to the ground truth score for each design
            value 'x' in a model-based optimization problem

        """

        input_key = "input_ids" if isinstance(
            self.internal_dataset, DiscreteDataset) else "inputs_embeds"

        # call the model's predict function to generate predictions
        return (model if model else self.params["model"])\
            .predict({input_key: x})[0].astype(np.float32)
