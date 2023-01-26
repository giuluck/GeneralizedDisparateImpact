import numpy as np
import pandas as pd
import seaborn as sns
import seaborn.categorical
from matplotlib import pyplot as plt

from moving_targets.learners import LinearRegression
from src.models import FirstOrderMaster


# noinspection PyProtectedMember
class CustomViolinPlotter(seaborn.categorical._ViolinPlotter):
    def __init__(self, *args, **kwargs):
        super(CustomViolinPlotter, self).__init__(*args, **kwargs)
        self.gray = 'black'


seaborn.categorical._ViolinPlotter = CustomViolinPlotter
sns.set_context('poster')
sns.set_style('whitegrid')

s = 80
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
    n = 100
    np.random.seed(0)
    space = np.random.uniform(-np.pi, np.pi, size=n)
    inp = pd.DataFrame.from_dict({'x': space})
    out = 4 * np.sin(space) + np.square(space) + np.random.normal(size=n)

    data = inp.copy()
    for i, kernel in enumerate(['no', 1, 2, 3]):
        # if no kernel use original targets, otherwise compute processed targets leveraging the causal exclusion master
        if kernel == 'no':
            y = out
            name = 'pre'
            title = 'No Constraint'
        else:
            master = FirstOrderMaster(classification=False, degree=kernel, excluded='x', threshold=0.0, relative=0)
            y = master.adjust_targets(x=inp, y=out, p=None)
            title = '$\\operatorname{GeDI}(x, y; V^' + str(kernel) + ') = 0$'
            name = f'k{kernel}'
        data[f'y_{name}'] = y
        # create a linspace on the x axis and compute the respective y as predicted by the shadow models
        sx = np.linspace(np.min(space), np.max(space), num=1000)
        sy1 = LinearRegression(polynomial=1).fit(space.reshape((-1, 1)), y).predict(sx.reshape((-1, 1)))
        sy2 = LinearRegression(polynomial=2).fit(space.reshape((-1, 1)), y).predict(sx.reshape((-1, 1)))
        sy3 = LinearRegression(polynomial=3).fit(space.reshape((-1, 1)), y).predict(sx.reshape((-1, 1)))
        # plot the scattered data points and the shadow model predictions
        plt.figure(figsize=(4, 3.5), tight_layout=True)
        plt.ylim(bottom=np.min(out) - 2, top=np.max(out) + 2)
        plt.title(title)
        sns.scatterplot(x=space, y=y, s=s, alpha=scatter_alpha, color=scatter_color, zorder=zorder)
        if plot_models:
            sns.lineplot(x=sx, y=sy1, linewidth=linewidth, alpha=model_alpha, color=model_colors[0], label=f'lr 1')
            sns.lineplot(x=sx, y=sy2, linewidth=linewidth, alpha=model_alpha, color=model_colors[1], label=f'lr 2')
            sns.lineplot(x=sx, y=sy3, linewidth=linewidth, alpha=model_alpha, color=model_colors[2], label=f'lr 3')
        # show plot and save csv data according to boolean variables
        if save_data:
            # noinspection PyTypeChecker
            pd.DataFrame.from_dict(data).to_csv(f'{folder}/running_{kernel}.csv', index=False)
        if save_plot:
            plt.savefig(f'{folder}/running_{kernel}.eps', format='eps')
        if show_plot:
            plt.show()
