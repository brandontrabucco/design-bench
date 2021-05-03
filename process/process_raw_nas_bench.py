from nasbench import api
from nasbench.lib import graph_util
import numpy as np
import argparse
import os
import itertools


# tokens representing markup
START = '<start>'
STOP = '<stop>'
PAD = '<pad>'
SEPARATOR = '<separator>'


# tokens representing layers
INPUT = 'input'
OUTPUT = 'output'
CONV1X1 = 'conv1x1-bn-relu'
CONV3X3 = 'conv3x3-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'


# tokens representing adjacency
ADJACENCY_ZERO = '0'
ADJACENCY_ONE = '1'


# list mapping ids to token names
ID_TO_NODE = [START, STOP, PAD, SEPARATOR,
              INPUT, OUTPUT, CONV1X1, CONV3X3, MAXPOOL3X3,
              ADJACENCY_ZERO, ADJACENCY_ONE]


# dict mapping token names to ids
NODE_TO_ID = {name: idx for
              idx, name in enumerate(ID_TO_NODE)}


def generate_graphs(nasbench):
    """A function that generates all possible graphs that could have been
    processed via NAS Bench, and yields tuples of x values and y values,
    where y is zero when x is not contained in NASBench-101

    Arguments:

    nasbench: NASBench
        an instantiation of the NASBench class provided in the official
        release of nas bench source code

    Returns:

    generator: Iterator
        a generator tha yields tuples of x values and y values, where
        y is zero when x is not contained in NASBench-101

    """

    # these settings were used in the NASBench-101 paper
    max_vertices = 7
    max_edges = 9
    max_epochs = 108
    max_adjacency_size = max_vertices * (max_vertices - 1) // 2

    # a helper function that maps a model architecture to a metric
    def model_to_metric(_ops, _matrix):
        model_spec = api.ModelSpec(matrix=_matrix,
                                   ops=[ID_TO_NODE[t] for t in _ops])
        computed_metrics = nasbench.get_metrics_from_spec(model_spec)[1]
        return np.mean([d["final_test_accuracy"] for d in
                        computed_metrics[max_epochs]])\
            .astype(np.float32).reshape([1])

    # generate all possible graphs and labellings
    for vertices in range(2, max_vertices + 1):
        for bits in range(2 ** (vertices * (vertices - 1) // 2)):

            # generate an adjacency matrix for the graph
            matrix = np.fromfunction(graph_util.gen_is_edge_fn(bits),
                                     (vertices, vertices),
                                     dtype=np.int8)

            # discard graphs which can be pruned or exceed constraints
            if (not graph_util.is_full_dag(matrix) or
                    graph_util.num_edges(matrix) > max_edges):
                continue

            # convert the binary adjacency matrix to a vector
            vector = matrix[np.triu_indices(matrix.shape[0], k=1)]

            # Iterate through all possible labellings
            for labelling in itertools.product(
                    *[[CONV1X1, CONV3X3, MAXPOOL3X3]
                      for _ in range(vertices - 2)]):

                # convert the graph and labelling to numpy arrays
                ops = [INPUT] + list(labelling) + [OUTPUT]
                ops = np.array([NODE_TO_ID[t] for t in ops]).astype(np.int32)

                # yield samples encoded in a standard sequence format
                yield np.concatenate([
                    [NODE_TO_ID[START]], ops, [NODE_TO_ID[SEPARATOR]],
                    vector + NODE_TO_ID[ADJACENCY_ZERO],
                    [NODE_TO_ID[STOP]], [NODE_TO_ID[PAD]] * (
                        max_vertices - ops.size +
                        max_adjacency_size - vector.size)],
                    axis=0), model_to_metric(ops, matrix)


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Process Raw NASBench")
    parser.add_argument("--tfrecord",
                        type=str, default="./data/nasbench_full.tfrecord")
    parser.add_argument("--shard-folder",
                        type=str, default="./")
    parser.add_argument("--samples-per-shard",
                        type=int, default=50000)
    args = parser.parse_args()
    files_list = []
    os.makedirs(os.path.join(
        args.shard_folder, f"nas_bench/"), exist_ok=True)

    # loop through all possible architectures
    x_list = []
    y_list = []
    shard_id = 0
    for x, y in generate_graphs(api.NASBench(args.tfrecord)):

        # save the x and y values and potentially write a shard
        x_list.append(x)
        y_list.append(y)
        if len(y_list) == args.samples_per_shard:

            x_shard = np.stack(x_list, axis=0).astype(np.int32)
            x_shard_path = os.path.join(
                args.shard_folder, f"nas_bench/nas_bench-x-{shard_id}.npy")
            np.save(x_shard_path, x_shard)

            y_shard = np.stack(y_list, axis=0).astype(np.float32)
            y_shard_path = os.path.join(
                args.shard_folder, f"nas_bench/nas_bench-y-{shard_id}.npy")
            np.save(y_shard_path, y_shard)
            files_list.append(f"nas_bench/nas_bench-x-{shard_id}.npy")

            x_list = []
            y_list = []
            shard_id += 1

    # serialize another shard if there are any remaining
    if len(x_list) > 0:

        x_shard = np.stack(x_list, axis=0).astype(np.int32)
        x_shard_path = os.path.join(
            args.shard_folder, f"nas_bench/nas_bench-x-{shard_id}.npy")
        np.save(x_shard_path, x_shard)

        y_shard = np.stack(y_list, axis=0).astype(np.float32)
        y_shard_path = os.path.join(
            args.shard_folder, f"nas_bench/nas_bench-y-{shard_id}.npy")
        np.save(y_shard_path, y_shard)
        files_list.append(f"nas_bench/nas_bench-x-{shard_id}.npy")

    print(files_list)
