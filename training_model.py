from typing import Tuple

import joblib
# import matplotlib.pyplot as plt
import pandas as pd
# import seaborn as sns
import typer
import umap
from clearml import Logger, Task
from sklearn.datasets import load_digits
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    return load_digits(as_frame=True, return_X_y=True)


def main(min_neighbors: int, max_neighbors: int) -> None:
    Task.set_random_seed(42)
    task = Task.init(
        project_name="digits-training",
        task_name="Best training",
        # output_uri='s3://s3.timeweb.com:443/20ec772d-mlops'
    )
    logger = Logger.current_logger()
    X, y = load_data()
    task.upload_artifact(name='Training data', artifact_object=pd.concat([X, y], axis=1))
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    scaler.fit(train_X)
    train_X = scaler.transform(train_X)
    test_X = scaler.transform(test_X)
    joblib.dump(scaler, 'models/scaler.pkl')
    
    reducer = umap.UMAP(n_components=5, random_state=42, n_neighbors=15)
    reducer.fit(train_X, train_y)
    train_X = reducer.transform(train_X)
    test_X = reducer.transform(test_X)
    joblib.dump(reducer, 'models/reducer.pkl')

    best_model = {'best_model': None, 'best_roc': 0}
    for n in range(min_neighbors, max_neighbors + 1):
        print(f'Training model with {n} neighbors')
        model = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=n, weights='distance'))
        model.fit(train_X, train_y)

        acc = model.score(test_X, test_y)
        roc = roc_auc_score(test_y, model.predict_proba(test_X), multi_class='ovr')
        if roc > best_model['best_roc']:
            best_model['best_model'] = model
            best_model['best_roc'] = roc
        # можно итеративно записывать текущие метрики при подборе гиперпараметров
        logger.report_scalar('ROC_AUC', 'Test', iteration=n, value=roc)
        logger.report_scalar('Accuracy', 'Test', iteration=n, value=acc)

    joblib.dump(best_model['best_model'], 'models/model_cls.pkl')
    task.close()


if __name__ == '__main__':
    typer.run(main)