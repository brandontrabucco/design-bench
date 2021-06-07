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
    return oracle.predict(
        denormalize_design_vector(design)[np.newaxis].astype(np.float32))[0]


pool = multiprocessing.Pool()


def many_evaluate(designs):
    return pool.map(single_evaluate, designs)


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Process Raw Ant Morphologies")
    parser.add_argument("--shard-folder", type=str, default="./")
    parser.add_argument("--samples", type=int, default=25000)
    args = parser.parse_args()

    os.makedirs(os.path.join(
        args.shard_folder, f"ant_morphology/"), exist_ok=True)

    golden_design = normalize_design_vector(
        np.concatenate(DEFAULT_DESIGN, axis=0))

    sigma = 0.02
    max_iterations = 250
    save_every = 1

    designs = [denormalize_design_vector(golden_design)]
    predictions = [single_evaluate(golden_design)]

    for i in itertools.count():

        initial_design = golden_design + \
                         np.random.normal(0, 0.075, golden_design.shape)

        es = cma.CMAEvolutionStrategy(initial_design, sigma)

        step = 0
        while not es.stop() and step < max_iterations:

            if len(designs) >= args.samples:
                break

            xs = es.ask()
            ys = many_evaluate(xs)

            es.tell(xs, [-yi[0] for yi in ys])

            step += 1

            if step % save_every == 0:
                designs.extend([denormalize_design_vector(xi) for xi in xs])
                predictions.extend(ys)

            print(f"CMA-ES ({len(designs)} samples)"
                  f" - Restart {i+1}"
                  f" - Step {step+1}/{max_iterations}"
                  f" - Current Objective Value = {np.mean(ys)}")

        np.save(os.path.join(
            args.shard_folder,
            f"ant_morphology/ant_morphology-x-0.npy"),
            np.array(designs).astype(np.float32))

        np.save(os.path.join(
            args.shard_folder,
            f"ant_morphology/ant_morphology-y-0.npy"),
            np.array(predictions).astype(np.float32).reshape([-1, 1]))

        if len(designs) >= args.samples:
            break

