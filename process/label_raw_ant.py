import cma
import numpy as np
import multiprocessing
import os
import argparse
import itertools
from design_bench.oracles.exact.ant_morphology_oracle import AntMorphologyOracle
from design_bench.datasets.continuous.ant_morphology_dataset import AntMorphologyDataset
from morphing_agents.mujoco.ant.designs import DEFAULT_DESIGN
from morphing_agents.mujoco.ant.designs import normalize_design_vector
from morphing_agents.mujoco.ant.designs import denormalize_design_vector


def single_evaluate(design):
    placeholder_dataset = AntMorphologyDataset()
    oracle = AntMorphologyOracle(placeholder_dataset)
    return oracle.predict(design[np.newaxis].astype(np.float32))[0]


pool = multiprocessing.Pool()


def many_evaluate(designs):
    return pool.map(single_evaluate, designs)


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Label Raw Ant Morphologies")
    parser.add_argument("--shard-folder", type=str, default="./")
    args = parser.parse_args()

    os.makedirs(os.path.join(
        args.shard_folder, f"ant_morphology/"), exist_ok=True)

    dataset = AntMorphologyDataset()
    designs = [x for x in dataset.iterate_samples(return_x=True, return_y=False)]
    predictions = many_evaluate(designs)

    np.save(os.path.join(
        args.shard_folder,
        f"ant_morphology/ant_morphology-x-0.npy"),
        np.array(designs).astype(np.float32))

    np.save(os.path.join(
        args.shard_folder,
        f"ant_morphology/ant_morphology-y-0.npy"),
        np.array(predictions).astype(np.float32).reshape([-1, 1]))
