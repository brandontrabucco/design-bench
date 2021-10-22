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

    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(25.0, 5.0))

    task_name_to_title = {
        "TFBind10-Exact-v0": "TF Bind 10",
        "ChEMBL_MCHC_CHEMBL3885882-RandomForest-v0": "ChEMBL",
        "Superconductor-RandomForest-v0": "Superconductor",
        "AntMorphology-Exact-v0": "Ant Morphology"
    }

    for i, task_name in enumerate([
            "TFBind10-Exact-v0",
            "ChEMBL_MCHC_CHEMBL3885882-RandomForest-v0",
            "Superconductor-RandomForest-v0",
            "AntMorphology-Exact-v0"]):

        print(f"Evaluating: {task_name}")

        task = design_bench.make(task_name)
        task.dataset.subsample(max_samples=1000)
        axis = axes[i]

        x = task.x
        y = task.y

        if task.is_discrete:

            num_classes = x.max()
            designs = np.random.randint(num_classes, size=x.shape)

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

        task_title = task_name_to_title[task_name]

        axis.set_xlabel(r'$\textbf{' + pretty(task.y_name) + '}$', fontsize=24)
        axis.set_ylabel(r'$\textbf{Number of samples}$', fontsize=24)
        axis.set_title(r'$\textbf{' + task_title + '}$', fontsize=24, pad=20)
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
    plt.savefig(f'manifold_half.pdf')
