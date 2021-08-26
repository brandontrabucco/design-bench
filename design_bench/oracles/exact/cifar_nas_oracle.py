from design_bench.oracles.exact_oracle import ExactOracle
from design_bench.datasets.discrete_dataset import DiscreteDataset
from design_bench.datasets.discrete.cifar_nas_dataset import CIFARNASDataset
import numpy as np


"""Large portions of this code are taken directly from:
https://github.com/young-geng/cifar_nas/
blob/d5a08299e3550f4ab946644c35d1834fb8c8fa49/nas.py

Original Author: Young Geng
Design-Bench Maintainer: Brandon Trabucco
"""

import functools
import operator

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms


class Block(nn.Module):

    def __init__(self, config):

        super().__init__()
        kernel_sizes = [2, 3, 4, 5, 6]

        activations = [
            nn.ReLU,
            nn.ELU,
            nn.LeakyReLU,
            nn.SELU,
            nn.SiLU,
        ]

        kernel_size1 = kernel_sizes[config[0]]
        activation1 = activations[config[1]]

        kernel_size2 = kernel_sizes[config[2]]
        activation2 = activations[config[3]]

        # the next lines are to compensate for how older versions of
        # pytorch do not natively support padding=same
        padding1 = (kernel_size1 // 2 + (kernel_size1 - 2 * (
            kernel_size1 // 2)) - 1, kernel_size1 // 2,
                kernel_size1 // 2 + (kernel_size1 - 2 * (
                    kernel_size1 // 2)) - 1, kernel_size1 // 2)

        padding2 = (kernel_size2 // 2 + (kernel_size2 - 2 * (
            kernel_size2 // 2)) - 1, kernel_size2 // 2,
                kernel_size2 // 2 + (kernel_size2 - 2 * (
                    kernel_size2 // 2)) - 1, kernel_size2 // 2)

        self.network = nn.Sequential(
            nn.ZeroPad2d(padding1),
            nn.Conv2d(32, 32, kernel_size1),
            activation1(),
            nn.ZeroPad2d(padding2),
            nn.Conv2d(32, 32, kernel_size2),
            activation2(),
            nn.MaxPool2d(3, stride=1, padding=1)
        )

    def forward(self, x):
        return self.network(x) + x


class ParameterizedCNN(nn.Module):

    def __init__(self, config):
        super().__init__()

        n_params = 4
        assert len(config) % n_params == 0
        configs = []
        for i in range(0, len(config), n_params):
            configs.append(config[i:i + n_params])

        layers = [nn.ZeroPad2d((1, 1, 1, 1)),
                  nn.Conv2d(3, 32, 3)]

        for config in configs:
            layers.append(Block(config))

        layers.append(nn.Conv2d(32, 256, 1))

        self.convs = nn.Sequential(*layers)
        self.linear = nn.Linear(256, 10)

    def forward(self, x):
        x = self.convs(x)
        x = torch.mean(x, (2, 3))
        x = self.linear(x)
        return x


def search(arch_config):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True,
        transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=256,
        shuffle=True, num_workers=3)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True,
        transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=200,
        shuffle=False, num_workers=3)

    net = ParameterizedCNN(arch_config)

    net = net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=1e-3,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=20)

    def train():
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    def test():
        nonlocal best_acc
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        acc = 100.*correct/total
        if acc > best_acc:
            best_acc = acc
        return acc

    test_acc = 0.0
    for epoch in range(start_epoch, start_epoch + 20):
        train()
        test_acc = test()
        scheduler.step()

    return test_acc


class CIFARNASOracle(ExactOracle):
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

    """

    name = "exact_test_accuracy"

    @classmethod
    def supported_datasets(cls):
        """An attribute the defines the set of dataset classes which this
        oracle can be applied to forming a valid ground truth score
        function for a model-based optimization problem

        """

        return {CIFARNASDataset}

    @classmethod
    def fully_characterized(cls):
        """An attribute the defines whether all possible inputs to the
        model-based optimization problem have been evaluated and
        are are returned via lookup in self.predict

        """

        return True

    @classmethod
    def is_simulated(cls):
        """An attribute the defines whether the values returned by the oracle
         were obtained by running a computer simulation rather than
         performing physical experiments with real data

        """

        return False

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

        return np.array([search(x.tolist())]).astype(np.float32)

    def __init__(self, dataset: DiscreteDataset, **kwargs):
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
        super(CIFARNASOracle, self).__init__(
            dataset, is_batched=False,
            internal_batch_size=1, internal_measurements=1,
            expect_normalized_y=False,
            expect_normalized_x=False, expect_logits=False, **kwargs)
