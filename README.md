# Generalized Disparate Impact for Configurable Fairness Solutions in ML

This repository provides the code for reproducing the results obtained in the paper *Generalized Disparate Impact for Configurable Fairness Solutions in ML*, published at ICML 2023.

## Paper Citation


If you use this codebase, please cite:

```
@inproceedings{10.5555/3618408.3618867,
author = {Giuliani, Luca and Misino, Eleonora and Lombardi, Michele},
title = {Generalized disparate impact for configurable fairness solutions in ML},
year = {2023},
publisher = {JMLR.org},
abstract = {We make two contributions in the field of AI fairness over continuous protected attributes. First, we show that the Hirschfeld-Gebelein-Renyi (HGR) indicator (the only one currently available for such a case) is valuable but subject to a few crucial limitations regarding semantics, interpretability, and robustness. Second, we introduce a family of indicators that are: 1) complementary to HGR in terms of semantics; 2) fully interpretable and transparent; 3) robust over finite samples; 4) configurable to suit specific applications. Our approach also allows us to define fine-grained constraints to permit certain types of dependence and forbid others selectively. By expanding the available options for continuous protected attributes, our approach represents a significant contribution to the area of fair artificial intelligence.},
booktitle = {Proceedings of the 40th International Conference on Machine Learning},
articleno = {459},
numpages = {16},
location = {Honolulu, Hawaii, USA},
series = {ICML'23}
}
```

## Installation Setup

Before running any script, please make sure that you installed all the necessary requirements.

We suggest the use of a virtual environment in order to avoid package conflicts due to specific versions.
The virtual environment can be created via the command:
```
python -m venv <environment-name>
```
and the requirements installed via:
```
pip install -r requirements.txt
```

The code is written for `Python 3.7`, and it needs `Gurobi 10.0` to be installed as a standalone software.
Free academic licenses can be requested on their [website](https://www.gurobi.com/free-trial/).
All the models were trained on a machine with an Intel Core I9 10920X 3.5G and 64GB of RAM but no operation involved a time limit, therefore the results should vary only regarding to the time dimension.

Finally, we copied the code of the [Moving Targets](https://github.com/moving-targets/moving-targets) library in order to introduce a few custom changes.
The library can be also installed as an external package from [PyPI](https://pypi.org/project/moving-targets/).

## Experimental Results

The experimental results can be replicated by running the scripts in the `experiments` folder.

Before that, it is advisable to create a personal configuration for [Weights & Biases](https://wandb.ai/site) which is used to log the results.
This can be done by editing the `config.py` file in order to set a custom W&B entity and project names.

The folder contains six different scripts:
* `inspect_baselines.py` and `inspect_mt.py` allow, respectively, to manually inspect all the implemented approaches and (specifically) the ones based on Moving Targets;
* `gedi_calibration.py` and `gedi_mt.py` run the hyperparameter tuning experiments for Neural Networks and Moving Targets, respectively;
* `gedi_experiments.py` run the final experiments presented in the paper.
The remaining file (`config.py`) is used to setup the configuration for Weights & Biases, which is used to log the results of the two tuning and the final experimental script.

The original results presented in the paper can be found at [this link](https://wandb.ai/shape-constraints/gedi-experiments).

## Tables and Plots

The tables and plots presented in the paper can be obtained by running the scripts in the `export` folder.

The folder contains:
* a script to produce the plots of the running example (`running_example.py`) showed in Figure 3;
* a script to plot the value of the GeDI indicator with respect to the Binned DIDI (`gedi_vs_didi.py`) as in Figure 3;
* a script to plot the value of the GeDI indicator with respect to the HGR one (`gedi_vs_hgr.py`) as in Figure 4;
* a script to retrieve all the values obtained from the experiments and collect them in a _.csv_ file (`results_file.py`);
* a script to retrieve all the values obtained from the experiments and collect them in latex tables (`results_tables.py`).

## Contacts

In case of questions about the code or the paper, do not hesitate to contact the corresponding authors:

* **Luca Giuliani** ([luca.giuliani13@unibo.it](mailto:luca.giuliani13@unibo.it))
* **Eleonora Misino** ([eleonora.misino2@unibo.it](mailto:eleonora.misino2@unibo.it))
