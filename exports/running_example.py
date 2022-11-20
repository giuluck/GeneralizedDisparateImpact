import numpy as np
import pandas as pd
import seaborn as sns
import seaborn.categorical
from matplotlib import pyplot as plt

from moving_targets.learners import LinearRegression
from src.models import CausalExclusionMaster


# noinspection PyProtectedMember
class CustomViolinPlotter(seaborn.categorical._ViolinPlotter):
    def __init__(self, *args, **kwargs):
        super(CustomViolinPlotter, self).__init__(*args, **kwargs)
        self.gray = 'black'


seaborn.categorical._ViolinPlotter = CustomViolinPlotter
sns.set_context('poster')
sns.set_style('whitegrid')

s = 30
linewidth = 2
continuous = 'cnt'
categorical = 'ctg'
scatter_alpha = 1.0
model_alpha = 0.6
violin_color = 'white'
scatter_color = 'black'
model_colors = ['tab:red', 'tab:green', 'tab:blue']
plot_models = False
categorical_violin = False
zorder = 1
folder = '../temp'

save_data = True
save_plot = True
show_plot = True

if __name__ == '__main__':
    # LOAD COMMON DATA
    np.random.seed(0)
    space = np.random.uniform(-np.pi, np.pi, size=100)
    inp = pd.DataFrame.from_dict({'cnt': space, 'ctg': (space > 0.5 * np.pi).astype(int)})
    out = 4 * np.sin(space) + np.square(space) + np.random.normal(size=100)

    # RUNNING EXAMPLE WITH CATEGORICAL FEATURE (kernel 1 is enough)
    z = inp['ctg'].values
    data = {'x': z}
    plt.figure(figsize=(12, 5), tight_layout=True)
    ax = None
    # show two equal plots one for the original targets and one for the processed ones
    for i, name in enumerate(['pre', 'post']):
        # if 'pre' use original targets, otherwise compute processed targets leveraging the causal exclusion master
        if name == 'pre':
            y = out
            title = 'Original Targets'
        else:
            master = CausalExclusionMaster(classification=False, degrees=1, features='ctg', thresholds=0.0)
            y = master.adjust_targets(x=inp, y=out, p=None)
            title = 'Constrained Targets'
        data[f'y_{name}'] = y
        # create a linspace on the x axis and compute the respective y as predicted by the shadow model
        sx = np.linspace(z.min(), z.max(), num=1000)
        sy = LinearRegression(polynomial=1).fit(z.reshape((-1, 1)), y).predict(sx.reshape((-1, 1)))
        # plot either the scattered data points or the violin plots and the shadow model predictions
        ax = plt.subplot(1, 2, i + 1, sharex=ax, sharey=ax)
        ax.set_title(title)
        if categorical_violin:
            sns.violinplot(x=z, y=y, color=violin_color, zorder=zorder)
        else:
            sns.stripplot(x=z, y=y, alpha=scatter_alpha, color=scatter_color, zorder=zorder)
        if plot_models:
            sns.lineplot(x=sx, y=sy, linewidth=linewidth, color=model_colors[0], label=f'lr')
    # show plot and save csv data according to boolean variables
    if save_data:
        pd.DataFrame.from_dict(data).to_csv(f'{folder}/categorical.csv', index=False)
    if save_plot:
        plt.savefig(f'{folder}/categorical.png', format='png')
        plt.savefig(f'{folder}/categorical.svg', format='svg')
        plt.savefig(f'{folder}/categorical.eps', format='eps')
    if show_plot:
        plt.show()

    # RUNNING EXAMPLE WITH CONTINUOUS FEATURE
    z = inp['cnt'].values
    data = {'x': z}
    plt.figure(figsize=(12, 10), tight_layout=True)
    ax = None
    # show two equal plots one for the original targets and one for the processed ones
    for i, kernel in enumerate([None, 1, 2, 3]):
        # if no kernel use original targets, otherwise compute processed targets leveraging the causal exclusion master
        if kernel is None:
            y = out
            name = 'pre'
            title = 'Original Targets'
        else:
            master = CausalExclusionMaster(classification=False, degrees=kernel, features='cnt', thresholds=0.0)
            y = master.adjust_targets(x=inp, y=out, p=None)
            title = f'Constrained Targets up to Order {kernel}'
            name = f'k{kernel}'
        data[f'y_{name}'] = y
        # create a linspace on the x axis and compute the respective y as predicted by the shadow models
        sx = np.linspace(z.min(), z.max(), num=1000)
        sy1 = LinearRegression(polynomial=1).fit(z.reshape((-1, 1)), y).predict(sx.reshape((-1, 1)))
        sy2 = LinearRegression(polynomial=2).fit(z.reshape((-1, 1)), y).predict(sx.reshape((-1, 1)))
        sy3 = LinearRegression(polynomial=3).fit(z.reshape((-1, 1)), y).predict(sx.reshape((-1, 1)))
        # plot the scattered data points and the shadow model predictions
        ax = plt.subplot(2, 2, i + 1, sharex=ax, sharey=ax)
        ax.set_title(title)
        sns.scatterplot(x=z, y=y, s=s, alpha=scatter_alpha, color=scatter_color, zorder=zorder)
        if plot_models:
            sns.lineplot(x=sx, y=sy1, linewidth=linewidth, alpha=model_alpha, color=model_colors[0], label=f'lr 1')
            sns.lineplot(x=sx, y=sy2, linewidth=linewidth, alpha=model_alpha, color=model_colors[1], label=f'lr 2')
            sns.lineplot(x=sx, y=sy3, linewidth=linewidth, alpha=model_alpha, color=model_colors[2], label=f'lr 3')
    # show plot and save csv data according to boolean variables
    if save_data:
        pd.DataFrame.from_dict(data).to_csv(f'{folder}/continuous.csv', index=False)
    if save_plot:
        plt.savefig(f'{folder}/continuous.png', format='png')
        plt.savefig(f'{folder}/continuous.svg', format='svg')
        plt.savefig(f'{folder}/continuous.eps', format='eps')
    if show_plot:
        plt.show()
