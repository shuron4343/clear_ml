# from typing import Tuple
# from pathlib import Path

# import joblib
# # import matplotlib.pyplot as plt
import pandas as pd
# # import seaborn as sns
import numpy as np
import requests
# import typer
import umap
from clearml import TaskTypes
from clearml.automation.controller import PipelineDecorator
# from sklearn.datasets import load_digits
# from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
# from sklearn.multiclass import OneVsRestClassifier
# from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

@PipelineDecorator.component(return_values=['X, y'], cache=True, task_type=TaskTypes.data_processing)
def load_data() -> pd.DataFrame:
    print('Step one')

    from pathlib import Path 
    from clearml import Dataset
    import pandas as pd

    dataset_path = Dataset.get(
        dataset_name="DigitsData",
        dataset_project="digits-training",
    ).get_local_copy()
    print('dataset_path: ', dataset_path)
    df = pd.read_csv(Path(dataset_path, 'digits_data.csv'))
    df = df.drop(columns=df.filter(regex='Unnamed', axis=1).columns)
    y = df.pop('target')
    return df, y


@PipelineDecorator.component(return_values=['model'], cache=True, task_type=TaskTypes.training)
def train_model(min_neighbors, max_neighbors, train_X, test_X, train_y, test_y):
    print('step four')

    from clearml import Logger
    import joblib
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.metrics import roc_auc_score

    logger = Logger.current_logger()
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
    model = best_model['best_model']
    joblib.dump(model, 'models/model_cls.pkl')
    return model


@PipelineDecorator.component(return_values=['reducer'], cache=True, task_type=TaskTypes.training)
def fit_reducer(train_X, train_y, *args, **kwargs) -> umap.UMAP:
    print('step three')

    import joblib
    import umap

    reducer = umap.UMAP(*args, **kwargs)
    reducer.fit(train_X, train_y)
    joblib.dump(reducer, 'models/reducer.pkl')
    return reducer


@PipelineDecorator.component(return_values=['scaler'], cache=True, task_type=TaskTypes.training)
def fit_scaler(train_X, *args, **kwargs) -> StandardScaler:
    print('step two')

    import joblib
    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler(*args, **kwargs)
    scaler.fit(train_X)
    joblib.dump(scaler, 'models/scaler.pkl')
    return scaler


@PipelineDecorator.pipeline(
        name='training_pipeline', project='digits-training', version='0.1',
        args_map={'min': ['min_neighbors'], 'max': ['max_neighbors']},
        # pipeline_execution_queue=None,
)
def execute_pipeline(min_neighbors: int, max_neighbors: int) -> None:
    
    print('launch step one')
    X, y = load_data()
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=42)

    print('launch step two')
    scaler = fit_scaler(train_X)
    train_X = scaler.transform(train_X)
    test_X = scaler.transform(test_X)
    
    print('launch step three')
    reducer = fit_reducer(train_X, train_y, n_components=5, random_state=42, n_neighbors=15)
    train_X = reducer.transform(train_X)
    test_X = reducer.transform(test_X)

    print('launch step four')
    train_model(min_neighbors, max_neighbors, train_X, test_X, train_y, test_y)


if __name__ == '__main__':
    PipelineDecorator.run_locally()
    execute_pipeline(min_neighbors=2, max_neighbors=100)
    print('Precessing done')