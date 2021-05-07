from design_bench.task import Task
import re
import importlib


# these are the default task name match criterion and error messages
TASK_PATTERN = re.compile(r'(\w+)-(\w+)-v(\d+)$')
MISMATCH_MESSAGE = 'Attempted to register malformed task name: {}. (' \
                   'Currently all names must be of the form {}.)'
DEPRECATED_MESSAGE = 'Task {} not found (valid versions include {})'
UNKNOWN_MESSAGE = 'No registered task with name: {}'
REREGISTRATION_MESSAGE = 'Cannot re-register id: {}'


# this is used to import data set classes dynamically
def import_name(name):
    mod_name, attr_name = name.split(":")
    return getattr(importlib.import_module(mod_name), attr_name)


class TaskSpecification(object):

    def __init__(self, name, dataset, oracle,
                 dataset_kwargs=None, oracle_kwargs=None):
        """Create a specification for a model-based optimization task that
        dynamically imports that task when self.make is called.

        Arguments:

        name: str
            the name of the model-based optimization task which must match
            with the regex expression given by TASK_PATTERN
        dataset: str or callable
            the import path to the target dataset class or a callable that
            returns the target dataset class when called
        oracle: str or callable
            the import path to the target oracle class or a callable that
            returns the target oracle class when called
        dataset_kwargs: dict
            additional keyword arguments that are provided to the dataset
            class when it is initialized for the first time
        oracle_kwargs: dict
            additional keyword arguments that are provided to the oracle
            class when it is initialized for the first time

        """

        # store the init arguments
        self.name = name
        self.dataset = dataset
        self.oracle = oracle
        self.dataset_kwargs = {} if dataset_kwargs is None else dataset_kwargs
        self.oracle_kwargs = {} if oracle_kwargs is None else oracle_kwargs

        # check if the name matches with the regex template
        match = TASK_PATTERN.search(name)

        # if there is no match raise a ValueError
        if not match:
            raise ValueError(
                MISMATCH_MESSAGE.format(name, TASK_PATTERN.pattern))

        # otherwise select the task name from the regex match
        self.task_name = match.group(0)

    def make(self, dataset_kwargs=None, oracle_kwargs=None):
        """Instantiates the intended task using the additional
        keyword arguments provided in this function.

        Arguments:

        dataset_kwargs: dict
            additional keyword arguments that are provided to the dataset
            class when it is initialized for the first time
        oracle_kwargs: dict
            additional keyword arguments that are provided to the oracle
            class when it is initialized for the first time

        Returns:

        task: Task
            an instantiation of the task specified by task name which
            conforms to the standard task api

        """

        # use additional_kwargs to override self.kwargs
        kwargs = self.dataset_kwargs.copy()
        if dataset_kwargs is not None:
            kwargs.update(dataset_kwargs)

        # if self.entry_point is a function call it
        if callable(self.dataset):
            dataset = self.dataset(**kwargs)

        # if self.entry_point is a string import it first
        elif isinstance(self.dataset, str):
            dataset = import_name(self.dataset)(**kwargs)

        # return if the dataset could not be loaded
        else:
            return

        # use additional_kwargs to override self.kwargs
        kwargs = self.oracle_kwargs.copy()
        if oracle_kwargs is not None:
            kwargs.update(oracle_kwargs)

        # if self.entry_point is a function call it
        if callable(self.oracle):
            oracle = self.oracle(dataset, **kwargs)

        # if self.entry_point is a string import it first
        elif isinstance(self.dataset, str):
            oracle = import_name(self.oracle)(dataset, **kwargs)

        # return if the oracle could not be loaded
        else:
            return

        # potentially subsample the dataset
        dataset.subsample(
            max_samples=dataset_kwargs.get("max_samples", None),
            min_percentile=dataset_kwargs.get("min_percentile", 0),
            max_percentile=dataset_kwargs.get("max_percentile", 100))

        # relabel the dataset using the new oracle model
        dataset = dataset.clone(to_disk=True,
                                disk_target=f"{dataset.name}-{oracle.name}",
                                is_absolute=False)
        dataset.relabel(lambda x, y: oracle.predict(x))

        # return a task composing this oracle and dataset
        return Task(dataset, oracle)

    def __repr__(self):
        return "TaskSpecification({}, {}, {})".format(
            self.name, self.dataset, self.oracle)


class TaskRegistry(object):

    def __init__(self):
        """Provide a global interface for registering model-based
        optimization tasks that remain stable over time

        """

        self.task_specs = {}

    def make(self, name, dataset_kwargs=None, oracle_kwargs=None):
        """Instantiates the intended task using the additional
        keyword arguments provided in this function.

        Args:

        name: str
            the name of the model-based optimization task which must match
            with the regex expression given by TASK_PATTERN
        dataset_kwargs: dict
            additional keyword arguments that are provided to the dataset
            class when it is initialized for the first time
        oracle_kwargs: dict
            additional keyword arguments that are provided to the oracle
            class when it is initialized for the first time

        Returns:

        task: Task
            an instantiation of the task specified by task name which
            conforms to the standard task api

        """

        return self.spec(name).make(dataset_kwargs=dataset_kwargs,
                                    oracle_kwargs=oracle_kwargs)

    def all(self):
        """Generate a list of the names of all currently registered
        model-based optimization tasks

        Returns:

        names: list
            a list of names that corresponds to all currently registered
            tasks where names[i] is suitable for self.make(names[i])

        """

        return self.task_specs.values()

    def spec(self, name):
        """Looks up the task specification identifed by the task
        name argument provided here

        Args:

        name: str
            the name of the model-based optimization task which must match
            with the regex expression given by TASK_PATTERN

        Returns:

        spec: TaskSpecification
            a specification whose make function will dynamically import
            and create the task specified by 'name'

        """

        # check if the name matches with the regex template
        match = TASK_PATTERN.search(name)

        # if there is no match raise a ValueError
        if not match:
            raise ValueError(
                MISMATCH_MESSAGE.format(name, TASK_PATTERN.pattern))

        # try to locate the task specification
        try:
            return self.task_specs[name]

        # if it does not exist try to find out why
        except KeyError:

            # make a list of all similar registered tasks
            task_name = match.group(0)
            matching_tasks = [valid_name for valid_name, valid_spec
                              in self.task_specs.items()
                              if task_name == valid_spec.task_name]

            # there is a similar matching task
            if matching_tasks:
                raise ValueError(
                    DEPRECATED_MESSAGE.format(name, matching_tasks))

            # there are no similar matching tasks
            else:
                raise ValueError(
                    UNKNOWN_MESSAGE.format(name))

    def register(self, name, dataset, oracle,
                 dataset_kwargs=None, oracle_kwargs=None):
        """Register a specification for a model-based optimization task that
        dynamically imports that task when self.make is called.

        Args:

        name: str
            the name of the model-based optimization task which must match
            with the regex expression given by TASK_PATTERN
        dataset: str or callable
            the import path to the target dataset class or a callable that
            returns the target dataset class when called
        oracle: str or callable
            the import path to the target oracle class or a callable that
            returns the target oracle class when called
        dataset_kwargs: dict
            additional keyword arguments that are provided to the dataset
            class when it is initialized for the first time
        oracle_kwargs: dict
            additional keyword arguments that are provided to the oracle
            class when it is initialized for the first time

        """

        # raise an error if that task is already registered
        if name in self.task_specs:
            raise ValueError(REREGISTRATION_MESSAGE.format(name))

        # otherwise add that task to the collection
        self.task_specs[name] = TaskSpecification(
            name, dataset, oracle,
            dataset_kwargs=dataset_kwargs, oracle_kwargs=oracle_kwargs)


# create a global task registry
registry = TaskRegistry()


# wrap the TaskRegistry.register function globally
def register(name, dataset, oracle,
             dataset_kwargs=None, oracle_kwargs=None):
    return registry.register(
        name, dataset, oracle,
        dataset_kwargs=dataset_kwargs, oracle_kwargs=oracle_kwargs)


# wrap the TaskRegistry.make function globally
def make(name, dataset_kwargs=None, oracle_kwargs=None):
    return registry.make(name, dataset_kwargs=dataset_kwargs,
                         oracle_kwargs=oracle_kwargs)


# wrap the TaskRegistry.spec function globally
def spec(name):
    return registry.spec(name)
