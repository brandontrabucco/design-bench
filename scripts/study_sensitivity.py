from morphing_agents.mujoco.ant.designs import DEFAULT_DESIGN as DEFAULT_ANT
from morphing_agents.mujoco.dkitty.designs import DEFAULT_DESIGN as DEFAULT_DKITTY
import design_bench
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import tqdm


if __name__ == '__main__':

    plt.rcParams['text.usetex'] = True
    matplotlib.rc('font', family='serif', serif='cm10')
    matplotlib.rc('mathtext', fontset='cm')
    color_palette = ['#EE7733',
                     '#0077BB',
                     '#33BBEE',
                     '#009988',
                     '#CC3311',
                     '#EE3377',
                     '#BBBBBB',
                     '#000000']

    gold_ant = np.concatenate(DEFAULT_ANT, axis=0)
    gold_dkitty = np.concatenate(DEFAULT_DKITTY, axis=0)

    task = design_bench.make('DKittyMorphology-v0')

    perturb_channel = 4
    lower_bound = 3.14 - np.pi / 4
    upper_bound = 3.14
    bins = 100

    ys = []
    xs = []
    start_img = None
    final_img = None

    for b in tqdm.tqdm(range(bins + 1)):
        gold_dkitty[perturb_channel] = lower_bound + b * (upper_bound -
                                                          lower_bound) / bins
        xs.append(gold_dkitty[perturb_channel])

        # create a policy forward pass in numpy
        def mlp_policy(h):
            h = np.maximum(0.0, h @ task.weights[0] + task.weights[1])
            h = np.maximum(0.0, h @ task.weights[2] + task.weights[3])
            return np.tanh(np.split(
                h @ task.weights[4] + task.weights[5], 2)[0])

        # convert vectors to morphologies
        env = task.env_class(expose_design=False, fixed_design=[
            task.env_element(*np.clip(np.array(xi), task.lb, task.ub))
            for xi in np.split(gold_dkitty, task.elements)])

        if b == 0:
            final_img = env.render(mode='rgb_array')

        # do many rollouts using a pretrained agent
        average_returns = []
        for i in range(task.num_rollouts):
            obs = env.reset()
            average_returns.append(np.zeros([], dtype=np.float32))
            for t in range(task.rollout_horizon):
                obs, rew, done, info = env.step(mlp_policy(obs))
                average_returns[-1] += rew.astype(np.float32)
                if done:
                    break

        if b == bins - 1:
            start_img = env.render(mode='rgb_array')

        ys.append(
            np.mean(average_returns))

    fig, axes = plt.subplots(ncols=3,
                             nrows=1,
                             figsize=(25.0, 5.0))

    axes[0].imshow(start_img)
    axes[0].spines['right'].set_visible(False)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['bottom'].set_visible(False)
    axes[0].spines['left'].set_visible(False)
    axes[0].xaxis.set_ticks([])
    axes[0].xaxis.set_ticklabels([])
    axes[0].yaxis.set_ticks([])
    axes[0].yaxis.set_ticklabels([])
    axes[0].set_xlabel(r'$\textbf{Succeeds}: \; \theta = \pi$', fontsize=24)

    axes[2].imshow(final_img)
    axes[2].spines['right'].set_visible(False)
    axes[2].spines['top'].set_visible(False)
    axes[2].spines['bottom'].set_visible(False)
    axes[2].spines['left'].set_visible(False)
    axes[2].xaxis.set_ticks([])
    axes[2].xaxis.set_ticklabels([])
    axes[2].yaxis.set_ticks([])
    axes[2].yaxis.set_ticklabels([])
    axes[2].set_xlabel(r'$\textbf{Fails}: \; \theta = \frac{3}{4} \pi$', fontsize=24)

    axes[1].plot(xs, ys, color=color_palette[0], linewidth=4)
    axes[1].yaxis.set_tick_params(labelsize=16)
    axes[1].xaxis.set_tick_params(labelsize=16)

    axes[1].spines['right'].set_visible(False)
    axes[1].spines['top'].set_visible(False)

    axes[1].yaxis.set_ticks_position('left')
    axes[1].xaxis.set_ticks_position('bottom')

    axes[1].set_xlabel(r'$\textbf{Leg orientation} \; \theta$', fontsize=24)
    axes[1].set_ylabel(r'$\textbf{Average return}$', fontsize=24)
    axes[1].set_title(r'$\textbf{DKittyMorphology-v0}$', fontsize=24)
    axes[1].grid(color='grey',
                 linestyle='dotted',
                 linewidth=2)

    plt.tight_layout()
    plt.savefig(f'sensitivity.pdf')
