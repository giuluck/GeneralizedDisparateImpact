import seaborn as sns
from tensorflow.python.keras.callbacks import EarlyStopping

from src.experiments import get

sns.set_context('notebook')
sns.set_style('whitegrid')

if __name__ == '__main__':
    exp = get('communities continuous')
    fold = exp.get_folds(folds=1)
    model = exp.get_model(
        model='sbr',
        alpha=0.0,
        epochs=1000,
        val_split=0.2,
        callbacks=[EarlyStopping(monitor='val_loss', patience=10)],
        verbose=True)
    print('MODEL CONFIGURATION:')
    for k, v in model.config.items():
        print(f'  > {k} --> {v}')
    print('-------------------------------------------------')
    x, y = fold['train']
    exp.run_instance(model=model, x=x, y=y, fold=fold, index=None, log=None, show=True)
