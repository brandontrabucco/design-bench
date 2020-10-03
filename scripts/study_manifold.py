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

    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(25.0, 5.0))

    task = design_bench.make('GFP-v0')
    axis = axes[0]

    x = task.x
    y = task.y

    scores = []
    for i in tqdm.tqdm(range(x.shape[0])):
        design = np.random.multinomial(1, [1 / x.shape[-1]] * x.shape[-1], size=(1, *x.shape[1:-1]))
        score = task.score(design).mean()
        scores.append(score)

    axis.hist(scores, 20, color=color_palette[0], histtype='step', linewidth=4)
    axis.hist(y[:, 0], 20, color=color_palette[1], histtype='step', linewidth=4)

    axis.yaxis.set_tick_params(labelsize=16)
    axis.xaxis.set_tick_params(labelsize=16)

    axis.spines['right'].set_visible(False)
    axis.spines['top'].set_visible(False)

    axis.yaxis.set_ticks_position('left')
    axis.xaxis.set_ticks_position('bottom')

    axis.set_xlabel(r'$\textbf{Protein fluorescence}$', fontsize=24)
    axis.set_ylabel(r'$\textbf{Number of samples}$', fontsize=24)
    axis.set_title(r'$\textbf{GFP-v0}$', fontsize=24)
    axis.grid(color='grey',
              linestyle='dotted',
              linewidth=2)

    task = design_bench.make('MoleculeActivity-v0', split_percentile=100)
    axis = axes[1]

    x = task.x
    y = task.y

    scores = []
    for i in tqdm.tqdm(range(x.shape[0])):
        design = np.random.multinomial(1, [1 / x.shape[-1]] * x.shape[-1], size=(1, *x.shape[1:-1]))
        score = task.score(design).mean()
        scores.append(score)

    axis.hist(scores, 20, color=color_palette[0], histtype='step', linewidth=4)
    axis.hist(y[:, 0], 20, color=color_palette[1], histtype='step', linewidth=4)

    axis.yaxis.set_tick_params(labelsize=16)
    axis.xaxis.set_tick_params(labelsize=16)

    axis.spines['right'].set_visible(False)
    axis.spines['top'].set_visible(False)

    axis.yaxis.set_ticks_position('left')
    axis.xaxis.set_ticks_position('bottom')

    axis.set_xlabel(r'$\textbf{Drug activity}$', fontsize=24)
    axis.set_ylabel(r'$\textbf{Number of samples}$', fontsize=24)
    axis.set_title(r'$\textbf{MoleculeActivity-v0}$', fontsize=24)
    axis.grid(color='grey',
              linestyle='dotted',
              linewidth=2)

    task = design_bench.make('Superconductor-v0', split_percentile=100)
    axis = axes[2]

    x = task.x
    y = task.y

    upper_bound = x.max(axis=0, keepdims=True)
    lower_bound = x.min(axis=0, keepdims=True)

    scores = []
    for i in tqdm.tqdm(range(x.shape[0])):
        design = np.random.uniform(lower_bound, upper_bound)
        score = task.score(design).mean()
        scores.append(score)

    axis.hist(scores, 20, color=color_palette[0], histtype='step', linewidth=4)
    axis.hist(y[:, 0], 20, color=color_palette[1], histtype='step', linewidth=4)

    axis.yaxis.set_tick_params(labelsize=16)
    axis.xaxis.set_tick_params(labelsize=16)

    axis.spines['right'].set_visible(False)
    axis.spines['top'].set_visible(False)

    axis.yaxis.set_ticks_position('left')
    axis.xaxis.set_ticks_position('bottom')

    axis.set_xlabel(r'$\textbf{Critical temperature}$', fontsize=24)
    axis.set_ylabel(r'$\textbf{Number of samples}$', fontsize=24)
    axis.set_title(r'$\textbf{Superconductor-v0}$', fontsize=24)
    axis.grid(color='grey',
              linestyle='dotted',
              linewidth=2)

    task = design_bench.make('HopperController-v0')
    axis = axes[3]

    x = task.x
    y = task.y

    upper_bound = x.max(axis=0, keepdims=True)
    lower_bound = x.min(axis=0, keepdims=True)

    scores = []
    for i in tqdm.tqdm(range(x.shape[0])):
        design = np.random.uniform(lower_bound, upper_bound)
        score = task.score(design).mean()
        scores.append(score)

    axis.hist(scores, 20, color=color_palette[0], histtype='step', linewidth=4)
    axis.hist(y[:, 0], 20, color=color_palette[1], histtype='step', linewidth=4)

    axis.yaxis.set_tick_params(labelsize=16)
    axis.xaxis.set_tick_params(labelsize=16)

    axis.spines['right'].set_visible(False)
    axis.spines['top'].set_visible(False)

    axis.yaxis.set_ticks_position('left')
    axis.xaxis.set_ticks_position('bottom')

    axis.set_xlabel(r'$\textbf{Average return}$', fontsize=24)
    axis.set_ylabel(r'$\textbf{Number of samples}$', fontsize=24)
    axis.set_title(r'$\textbf{HopperController-v0}$', fontsize=24)
    axis.grid(color='grey',
              linestyle='dotted',
              linewidth=2)

    plt.legend([r'$\textbf{Sampled uniformly}$',
                r'$\textbf{Original}$'],
               ncol=2,
               loc='lower center',
               bbox_to_anchor=(-1.5, -0.5),
               fontsize=20,
               fancybox=True)
    fig.subplots_adjust(bottom=0.3, wspace=0.3)
    plt.tight_layout()
    plt.savefig(f'manifold.pdf')
