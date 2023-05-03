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
# from clearml.automation.controller import PipelineDecorator
# from sklearn.datasets import load_digits
# from sklearn.metrics import roc_auc_score
# from sklearn.model_selection import train_test_split
# from sklearn.multiclass import OneVsRestClassifier
# from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from clearml import PipelineController


def load_data() -> pd.DataFrame:

    from pathlib import Path 
    from clearml import Dataset
    import pandas as pd

    dataset_path = Dataset.get(
        dataset_name="digits",
        dataset_project="Datasets",
    ).get_local_copy()
    print('dataset_path: ', dataset_path)
    df = pd.read_csv(Path(dataset_path, 'digits_data.csv'))
    df = df.drop(columns=df.filter(regex='Unnamed', axis=1).columns)
    y = df.pop('target')
    return df, y


def train_model(min_neighbors, max_neighbors, train_X, test_X, train_y, test_y):

    from clearml import Logger
    import joblib
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.metrics import roc_auc_score

    logger = Logger.current_logger()
    best_model = {'best_model': None, 'best_roc': 0}
    for n in range(int(min_neighbors), int(max_neighbors) + 1):
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


def fit_reducer(train_X, train_y, n_components=5, random_state=42, n_neighbors=15) -> umap.UMAP:

    import joblib
    import umap

    reducer = umap.UMAP(n_components=n_components, random_state=random_state, n_neighbors=n_neighbors)
    reducer.fit(train_X, train_y)
    joblib.dump(reducer, 'models/reducer.pkl')
    return reducer


def fit_scaler(train_X) -> StandardScaler:

    import joblib
    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler()
    scaler.fit(train_X)
    joblib.dump(scaler, 'models/scaler.pkl')
    return scaler


def reduce_data(train_X, test_X, reducer):

    train_X = reducer.transform(train_X)
    test_X = reducer.transform(test_X)
    return train_X, test_X

def scale_data(train_X, test_X, scaler):

    train_X = scaler.transform(train_X)
    test_X = scaler.transform(test_X)
    return train_X, test_X

def split_data(X, y):

    from sklearn.model_selection import train_test_split

    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=42)
    return train_X, test_X, train_y, test_y


if __name__ == '__main__':

    # create the pipeline controller
    pipe = PipelineController(
        project='digits-training',
        name='Pipeline demo s3',
        version='0.1',
        add_pipeline_tags=False,
        auto_version_bump=True
    )

    # set the default execution queue to be used (per step we can override the execution)
    pipe.set_default_execution_queue('default')

    # add parameters to the pipeline
    pipe.add_parameter(
        name='min_neighbors',
        description='Minimum number of neighbors',
        default = 2,
        param_type=int,
    )
    pipe.add_parameter(
        name='max_neighbors',
        description='Maximum number of neighbors',
        default = 100,
        param_type=int,
    )

    # add pipeline components
    pipe.add_function_step(
        name='load_data',
        function=load_data,
        function_return=['X', 'y'],
        cache_executed_step=True,
    )
    pipe.add_function_step(
        name='split_data',
        function=split_data,
        parents=['load_data'],
        function_kwargs=dict(X='${load_data.X}', y='${load_data.y}'),
        function_return=['train_X', 'test_X', 'train_y', 'test_y'],
        cache_executed_step=True
    )
    pipe.add_function_step(
        name='scaler',
        function=fit_scaler,
        parents=['split_data'],
        function_kwargs=dict(train_X='${split_data.train_X}'),
        function_return=['scaler'],
        cache_executed_step=True,
    )
    pipe.add_function_step(
        name='norm_data',
        function=scale_data,
        parents=['scaler'],
        function_kwargs=dict(train_X='${split_data.train_X}', test_X='${split_data.test_X}', scaler='${scaler.scaler}'),
        function_return=['train_X', 'test_X'],
        cache_executed_step=True,
    )
    pipe.add_function_step(
        name='reducer',
        function=fit_reducer,
        parents=['norm_data'],
        function_kwargs=dict(train_X='${norm_data.train_X}', train_y='${split_data.train_y}'),
        function_return=['reducer'],
        cache_executed_step=True,
    )
    pipe.add_function_step(
        name='reduce_data',
        function=reduce_data,
        parents=['reducer'],
        function_kwargs=dict(train_X='${norm_data.train_X}', test_X='${norm_data.test_X}', reducer='${reducer.reducer}'),
        function_return=['train_X', 'test_X'],
        cache_executed_step=True
    )
    pipe.add_function_step(
        name='training_model',
        function=train_model,
        parents=['reduce_data'],
        function_kwargs=dict(
            min_neighbors='${pipeline.min_neighbors}', max_neighbors='${pipeline.max_neighbors}',
            train_X='${reduce_data.train_X}', test_X='${reduce_data.test_X}',
            train_y='${split_data.train_y}', test_y='${split_data.test_y}'
        ),
        function_return=['model'],
        cache_executed_step=True,
    )

    # execute the pipeline
    pipe.start_locally(run_pipeline_steps_locally=True)

    print('Pipeline execution completed')