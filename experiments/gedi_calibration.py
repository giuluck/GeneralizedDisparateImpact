"""This script runs the experiments on neural networks calibration and stores the results on Weights & Biases."""

import os

os.environ['WANDB_SILENT'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import shutil
import time

from experiments.config import WandBConfig
from src.models import NeuralNetwork
from src.experiments import get

datasets = ['communities', 'adult']
units = [[32] * 2, [32] * 3, [128] * 2, [128] * 3, [256] * 2, [256] * 3, [1024] * 2, [1024] * 3]
batch = [16, 128, -1]
tot = len(datasets) * len(units) * len(batch)

if __name__ == '__main__':
    i = 0
    for ds in datasets:
        for hu in units:
            for bs in batch:
                i += 1
                start = time.time()
                print(f'Trial {i:0{len(str(tot))}}/{tot}: dataset="{ds}", units={hu}, batch={bs}', end='')
                exp = get(f'{ds} categorical')
                fold = exp.get_folds(folds=1)
                x, y = fold['train']
                log = NeuralNetwork.WandbLogger(
                    fold=fold,
                    metrics=exp.metrics,
                    dataset=ds,
                    project=WandBConfig.gedi_calibration,
                    entity=WandBConfig.entity,
                    run=f'[{ds}] - {hu} - [{bs}]'
                )
                mdl = NeuralNetwork(
                    classification=exp.classification,
                    hidden_units=hu,
                    epochs=500,
                    logger=log,
                    verbose=False,
                    batch_size=len(x) if bs == -1 else bs
                )
                exp.run_instance(model=mdl, x=x, y=y, fold=fold, index=None, log=None, show=False)
                print(f' -- elapsed time = {time.time() - start:.2f}s')
    shutil.rmtree('wandb')
