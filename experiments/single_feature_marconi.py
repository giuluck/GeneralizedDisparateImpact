import importlib.resources
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import confusion_matrix, f1_score, make_scorer

dataset = 0
DEBUG = False
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == '__main__':

    # Handle data
    with importlib.resources.path('data','marconi') as filepath:
        files = [filepath.joinpath(gzip) for gzip in os.listdir(filepath)]
    df = pd.read_parquet(files[dataset]).rename(columns={'timestamp': 'index'}).set_index('index')

    scaling = 'raw'
    if scaling =='stdScaled':
        X = df.drop(columns=['label', 'New_label']).astype('float32')
        scaler = StandardScaler()
        scaler =scaler.fit(X)
        X_scaled = scaler.transform(X=X)
        X = pd.DataFrame(data=X_scaled,  # values
                    index = X.index,  # 1st column as index
                     columns = X.columns)  #
    elif scaling == 'normalized':
        X = df.drop(columns=['label', 'New_label']).astype('float32')
        scaler = Normalizer()
        scaler =scaler.fit(X)
        X_scaled = scaler.transform(X=X)
        X = pd.DataFrame(data=X_scaled,  # values
                    index = X.index,  # 1st column as index
                    columns = X.columns)  #
    elif scaling == 'raw':
        X = df.drop(columns=['label', 'New_label']).astype('float32')

    #X = X[[f for f in X.columns if 'avg:' in f]].rename(columns=lambda f: f[4:])
    y = df['New_label'].astype('category').cat.codes.values


    # Train-test split
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    kf = KFold(n_splits=5, random_state=42, shuffle=True)
    folder = os.path.join(PROJECT_DIR, 'marconi_single_feature')
    save = True
    metrics ={}

    # features = [f for f in X.columns if 'load' in f or 'idle' in f] + ['all']
    models = {'LogReg': LogisticRegression(class_weight='balanced', max_iter=500, n_jobs = -1),
              'SVC':  SVC(C = 1., class_weight='balanced', gamma='scale', max_iter=500),
              'DecTree': DecisionTreeClassifier(class_weight='balanced')}
    features = [f for f in X.columns if 'avg' in f] + ['all']

    verbose = False

    # Loop over features
    for feature in tqdm(features):
        if verbose:
            print(f'Testing feature "{feature}"...')
        metrics[feature] = dict()
        # Loop over models
        for model_name in models:
            model = models[model_name]
            f1_train = []
            f1_test = []
            if verbose:
                print(f'  ...with model {model_name}')

            # Loop over 10-folds
            for train_index, test_index in kf.split(X):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y[train_index], y[test_index]
                # Classify
                if feature == 'all':
                    model = model.fit(X_train,y_train)
                    y_pred_train = model.predict(X_train)
                    y_pred_test = model.predict(X_test)
                    f1_train.append(f1_score(y_train, y_pred_train, average='macro'))
                    f1_test.append(f1_score(y_test, y_pred_test,average='macro'))
                else:
                    model = model.fit(X_train[feature].to_numpy().reshape(-1,1),y_train)
                    y_pred_train = model.predict(X_train[feature].to_numpy().reshape(-1,1))
                    y_pred_test = model.predict(X_test[feature].to_numpy().reshape(-1,1))
                    f1_train.append(f1_score(y_train, y_pred_train, average='macro'))
                    f1_test.append(f1_score(y_test, y_pred_test,average='macro'))
                    if DEBUG:
                        break
            # Metrics
            f1_mean_train = np.mean(f1_train)
            f1_std_train = np.std(f1_train)
            f1_mean_test = np.mean(f1_test)
            f1_std_test = np.std(f1_test)
            metrics[feature][f'TRAIN_f1_mean_{model_name}'] = f1_mean_train
            metrics[feature][f'TRAIN_f1_std_{model_name}'] = f1_std_train
            metrics[feature][f'TEST_f1_mean_{model_name}'] = f1_mean_test
            metrics[feature][f'TEST_f1_std_{model_name}'] = f1_std_test
            metrics[feature][f'DIFF_f1_{model_name}'] = f1_mean_test - f1_std_train
            if verbose:
                print(f"     f1 score: {metrics[feature]}")

            if DEBUG:
                break

        if DEBUG:
            break

    if save:
        df = pd.DataFrame.from_dict(metrics).T
        df.to_json(os.path.join(folder, f'single_feature_{scaling}_metrics.json'))
