from morphing_agents.mujoco.ant.designs import DEFAULT_DESIGN
from morphing_agents.mujoco.ant.elements import LEG_LOWER_BOUND
from morphing_agents.mujoco.ant.elements import LEG_UPPER_BOUND
from multiprocessing import Pool
import design_bench
import numpy as np
import tqdm
import matplotlib.pyplot as plt


def score(z):
    t = design_bench.make('AntMorphology-v0')
    s0 = t.score(z[np.newaxis, :])[0, :]
    s1 = t.score(z[np.newaxis, :])[0, :]
    s2 = t.score(z[np.newaxis, :])[0, :]
    s3 = t.score(z[np.newaxis, :])[0, :]
    s4 = t.score(z[np.newaxis, :])[0, :]
    return (s0 + s1 + s2 + s3 + s4) / 5.


if __name__ == '__main__':

    lb = np.concatenate([LEG_LOWER_BOUND] * 4, axis=0)[np.newaxis, :]
    ub = np.concatenate([LEG_UPPER_BOUND] * 4, axis=0)[np.newaxis, :]
    size = (ub - lb) / 2.0

    x = np.concatenate(DEFAULT_DESIGN, axis=0)[np.newaxis, :]
    x = np.tile(x, (5000, 1))
    x = np.clip(x + np.random.normal(0, 0.015, x.shape) * size, lb, ub)

    pool = Pool(12)
    *xs, = x  # unstack the dataset
    ys = pool.map(score, tqdm.tqdm(xs))

    y = np.stack(ys, axis=0)
    np.save('ant_morphology_X.npy', x)
    np.save('ant_morphology_y.npy', y)

    plt.hist(y, 1000)
    plt.title('Coverage Of Ant Morphology Data')
    plt.xlabel('Average Return')
    plt.ylabel('Number Of Examples')
    plt.show()
