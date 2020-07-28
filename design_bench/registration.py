"""Design for the registry is inspired significantly by:
https://github.com/openai/gym/blob/master/gym/envs/registration.py
"""

import re
import importlib


# these are the default task name match criterion and error messages
TASK_PATTERN = re.compile(r'([\w:.-]+)-v(\d+)$')
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

    def __init__(self,
                 name,
                 entry_point,
                 kwargs=None):
        """Create a specification for a model-based optimization task that
        dynamically imports that task when self.make is called.

        Args:

        name: str
            the name of the model-based optimization task which must match
            with the regex expression given by TASK_PATTERN
        entry_point: str or callable
            the import path to the target task class or a callable that
            returns the target task class when called
        kwargs: dict
            additional keyword arguments that are provided to the task
            class when it is initialized for the first time
        """

        # store the init arguments
        self.name = name
        self.entry_point = entry_point
        self.kwargs = {} if kwargs is None else kwargs

        # check if the name matches with the regex template
        match = TASK_PATTERN.search(name)

        # if there is no match raise a ValueError
        if not match:
            raise ValueError(
                MISMATCH_MESSAGE.format(name, TASK_PATTERN.pattern))

        # otherwise select the task name from the regex match
        self.task_name = match.group(0)

    def make(self, **additional_kwargs):
        """Instantiates the intended task using the additional
        keyword arguments provided in this function.

        Args:

        additional_kwargs: dict
            additional keyword arguments that are provided to the task
            class when it is initialized for the first time

        Returns:

        task: Task
            an instantiation of the task specified by task name which
            conforms to the standard task api
        """

        # use additional_kwargs to override self.kwargs
        kwargs = self.kwargs.copy()
        kwargs.update(additional_kwargs)

        # if self.entry_point is a function call it
        if callable(self.entry_point):
            return self.entry_point(**kwargs)

        # if self.entry_point is a string import it first
        elif isinstance(self.entry_point, str):
            return import_name(self.entry_point)(**kwargs)

    def __repr__(self):
        return "TaskSpecification({}, {})".format(self.name, self.entry_point)


class TaskRegistry(object):

    def __init__(self):
        """Provide a global interface for registering model-based
        optimization tasks that remain stable over time
        """

        self.task_specs = {}

    def make(self, name, **kwargs):
        """Instantiates the intended task using the additional
        keyword arguments provided in this function.

        Args:

        name: str
            the name of the model-based optimization task which must match
            with the regex expression given by TASK_PATTERN
        additional_kwargs: dict
            additional keyword arguments that are provided to the task
            class when it is initialized for the first time

        Returns:

        task: Task
            an instantiation of the task specified by task name which
            conforms to the standard task api
        """

        return self.spec(name).make(**kwargs)

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

    def register(self, name, entry_point, **kwargs):
        """Register a specification for a model-based optimization task that
        dynamically imports that task when self.make is called.

        Args:

        name: str
            the name of the model-based optimization task which must match
            with the regex expression given by TASK_PATTERN
        entry_point: str or callable
            the import path to the target task class or a callable that
            returns the target task class when called
        kwargs: dict
            additional keyword arguments that are provided to the task
            class when it is initialized for the first time
        """

        # raise an error if that task is already registered
        if name in self.task_specs:
            raise ValueError(REREGISTRATION_MESSAGE.format(name))

        # otherwise add that task to the collection
        self.task_specs[name] = TaskSpecification(name, entry_point, **kwargs)


# create a global task registry
registry = TaskRegistry()


# wrap the TaskRegistry.register function globally
def register(name, entry_point, **kwargs):
    return registry.register(name, entry_point, **kwargs)


# wrap the TaskRegistry.make function globally
def make(name, **kwargs):
    return registry.make(name, **kwargs)


# wrap the TaskRegistry.spec function globally
def spec(name):
    return registry.spec(name)
