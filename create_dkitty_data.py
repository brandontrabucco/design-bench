from morphing_agents.mujoco.dkitty.designs import DEFAULT_DESIGN
from morphing_agents.mujoco.dkitty.elements import LEG_LOWER_BOUND
from morphing_agents.mujoco.dkitty.elements import LEG_UPPER_BOUND
from multiprocessing import Pool
import design_bench
import numpy as np
import argparse


def score(design):
    task = design_bench.make('DKittyMorphology-v0')
    return design, task.score(design[np.newaxis, :])[0, :]


def log_result(map_return):
    inner_x, inner_y = map_return
    final_xs.append(inner_x)
    final_ys.append(inner_y)

    # only save every percent of completion
    if len(final_xs) % (len(x) // 100) == 0:
        xs = np.stack(final_xs, axis=0)
        ys = np.stack(final_ys, axis=0)
        np.save(f'{args.out}_X.npy', xs)
        np.save(f'{args.out}_y.npy', ys)

        mean = np.mean(ys)
        stdv = np.std(ys - mean)
        print('{:.0%} done: {} {}'.format(
            len(final_ys) / len(x),
            mean,
            stdv))


if __name__ == '__main__':

    parser = argparse.ArgumentParser('DKittyData')
    parser.add_argument('--cores', type=int, default=1)
    parser.add_argument('--num-samples', type=int, default=20)
    parser.add_argument('--noise', type=float, default=.02)
    parser.add_argument('--out', type=str, default='dkitty_morphology')
    args = parser.parse_args()

    np.random.seed(np.random.randint(999999))
    lb = np.concatenate([LEG_LOWER_BOUND] * 4, axis=0)[np.newaxis, :]
    ub = np.concatenate([LEG_UPPER_BOUND] * 4, axis=0)[np.newaxis, :]
    size = (ub - lb) / 2.0

    # perturb the gold standard morphology
    x = np.concatenate(DEFAULT_DESIGN, axis=0)[np.newaxis, :]
    x = np.tile(x, (args.num_samples, 1))
    x = np.clip(x + np.random.normal(0, args.noise, x.shape) * size, lb, ub)

    final_xs = []
    final_ys = []

    # apply the scoring function asynchronously
    pool = Pool(args.cores)
    for x_i in x:
        pool.apply_async(score, args=[x_i], callback=log_result)

    # save the final results to the disk
    pool.close()
    pool.join()
    np.save(f'{args.out}_X.npy', np.stack(final_xs, axis=0))
    np.save(f'{args.out}_y.npy', np.stack(final_ys, axis=0))
