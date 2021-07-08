import design_bench
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


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

    def pretty(s):
        return s.replace('_', ' ').title()

    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(25.0, 15.0))

    for i, task_name in enumerate([
            "TFBind8-Exact-v0",
            "GFP-Transformer-v0",
            "UTR-Transformer-v0",
            "ChEMBL-ResNet-v0",
            "Superconductor-RandomForest-v0",
            "HopperController-Exact-v0",
            "AntMorphology-Exact-v0",
            "DKittyMorphology-Exact-v0"]):

        print(f"Evaluating: {task_name}")

        task = design_bench.make(task_name)
        task.dataset.subsample(max_samples=1000)
        axis = axes[i // 4][i % 4]

        x = task.x
        y = task.y

        if task.is_discrete:

            designs = np.random.multinomial(
                1, [1 / x.shape[-1]] * x.shape[-1], size=x.shape[:-1])

        else:

            upper_bound = x.max(axis=0, keepdims=True)
            lower_bound = x.min(axis=0, keepdims=True)

            designs = np.random.uniform(
                lower_bound, upper_bound, size=x.shape)

        scores = task.predict(designs)

        axis.hist(scores, 20, color=color_palette[0],
                  histtype='step', linewidth=4)
        axis.hist(y[:, 0], 20, color=color_palette[1],
                  histtype='step', linewidth=4)

        axis.yaxis.set_tick_params(labelsize=16)
        axis.xaxis.set_tick_params(labelsize=16)

        axis.spines['right'].set_visible(False)
        axis.spines['top'].set_visible(False)

        axis.yaxis.set_ticks_position('left')
        axis.xaxis.set_ticks_position('bottom')

        axis.set_xlabel(r'$\textbf{' + pretty(task.y_name) + '}$', fontsize=24)
        axis.set_ylabel(r'$\textbf{Number of samples}$', fontsize=24)
        axis.set_title(r'$\textbf{' + task_name + '}$', fontsize=24)
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
    fig.subplots_adjust(bottom=0.3, wspace=0.3, hspace=0.3)
    plt.tight_layout()
    plt.savefig(f'manifold.pdf')
