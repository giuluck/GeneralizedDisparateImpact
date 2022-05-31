import importlib.resources
import os
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score

dataset = 0
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == '__main__':
    # Handle data
    with importlib.resources.path('data','marconi') as filepath:
        files = [filepath.joinpath(gzip) for gzip in os.listdir(filepath)]
    df = pd.read_parquet(files[dataset]).rename(columns={'timestamp': 'index'}).set_index('index')

    # TODO: keeping avg information only for the moment, may need to change it
    X = df.drop(columns=['label', 'New_label']).astype('float32')
    X = X[[f for f in X.columns if 'avg:' in f]].rename(columns=lambda f: f[4:])
    y = df['New_label'].astype('category').cat.codes.values


    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    folder = os.path.join(PROJECT_DIR, 'marconi_single_feature')
    save = True
    metrics ={}
    # TODO: which features to exclude?
    #   > 'cpu_idle' and 'cpu_aidle' (and btw what is the difference?)
    #   > 'load_one', 'load_five' and 'load_fifteen' (where are the others?)
    #   > 'all' to consider all the features
    # features = [f for f in X.columns if 'load' in f or 'idle' in f] + ['all']
    features = [f for f in X.columns] + ['all']
    for feature in features:
        print(f'Testing feature "{feature}"...')
        # Classify
        if feature == 'all':
            model = LogisticRegression(class_weight='balanced', random_state=42, max_iter=500, n_jobs = -1).fit(X_train ,y_train)
            y_pred = model.predict(X_test)
        else:
            model = LogisticRegression(class_weight='balanced', random_state=42, max_iter=500, n_jobs=-1).fit(
                X_train[feature].to_numpy().reshape(-1, 1), y_train)
            y_pred = model.predict(X_test[feature].to_numpy().reshape(-1, 1))
        # Metrics
        metrics[feature] = [f1_score(y_test, y_pred, average='macro')]
        conf_matrix = confusion_matrix(y_test, y_pred)
        print(f"     f1 score: {metrics[feature]}")
        if save:
            df_cm = pd.DataFrame(conf_matrix, index=[i for i in range(2)],
                                 columns=[i for i in range(2)])
            plt.figure(figsize=(10, 7))
            sns.heatmap(df_cm, annot=True)
            Path(folder).mkdir(parents=True, exist_ok=True)
            path = os.path.join(folder, f'conf_matrix_{feature}.png')
            plt.savefig(path)
            plt.close('all')

    if save:
        df = pd.DataFrame.from_dict(metrics).T
        df = df.rename({0: "f1_score"}, axis=1)
        df.to_json(os.path.join(folder, f'single_feature_metrics.json'))
