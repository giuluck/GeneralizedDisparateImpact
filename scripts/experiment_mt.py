import os

os.environ['WANDB_SILENT'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import shutil
import time

from moving_targets.callbacks import WandBLogger
from src.experiments import get
from src.metrics import RegressionWeight

datasets = {
    'communities categorical': [1],
    # 'communities continuous': [1, 2, 3, 5],
    'adult categorical': [1],
    # 'adult continuous': [1, 2, 3, 5]
}

if __name__ == '__main__':
    print('-------------------------------------------------')
    for ds, degrees in datasets.items():
        print(f' * DATASET: {ds}')
        for i, dg in enumerate(degrees):
            if i != 0:
                print()
            print(f'    - DEGREE: {dg}')
            exp = get(ds)
            folds = exp.get_folds(folds=5)
            for idx, fold in enumerate(folds):
                print(f'      > FOLD: {idx}', end='')
                start = time.time()
                x, y = fold['train']
                # build the model without callback and add it later in order to use the model config
                mdl = exp.get_model(
                    model='mt',
                    fold=fold,
                    degrees=dg,
                    metrics=exp.metrics + [RegressionWeight(feature=f, degree=dg, name=f) for f in exp.excluded],
                )
                # noinspection PyUnresolvedReferences
                mdl.add_callback(WandBLogger(project='nci_mt', entity='giuluck', run_name=ds, fold=idx, **mdl.config))
                exp.run_instance(model=mdl, x=x, y=y, fold=fold, index=None, log=None, show=False)
                print(f' -- elapsed time = {time.time() - start:.2f}s')
        print('-------------------------------------------------')
    shutil.rmtree('wandb')
