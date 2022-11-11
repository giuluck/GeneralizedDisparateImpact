import os

os.environ['WANDB_SILENT'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import shutil
import logging
import time

import tensorflow as tf
from src.models import KerasWandbLogger, MLP
from src.experiments import get

logger = tf.get_logger()
logger.setLevel(logging.ERROR)

datasets = ['communities categorical', 'adult categorical']
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
                exp = get(ds)
                fold = exp.get_folds(folds=1)
                x, y = fold['train']
                cb = KerasWandbLogger(
                    metrics=exp.metrics,
                    dataset=ds,
                    fold=fold,
                    batch=bs,
                    units=hu,
                    run=f'[{ds.replace(" categorical", "")}] - {hu} - [{bs}]'
                )
                mdl = MLP(
                    classification=exp.classification,
                    epochs=500,
                    units=hu,
                    verbose=False,
                    callbacks=[cb],
                    batch_size=len(x) if bs == -1 else bs
                )
                exp.run_instance(model=mdl, x=x, y=y, fold=fold, index=None, log=None, show=False)
                print(f' -- elapsed time = {time.time() - start:.2f}s')
    shutil.rmtree('wandb')
