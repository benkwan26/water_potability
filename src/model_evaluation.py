from dvclive import Live
import json
import os
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import yaml

PARAMS_PATH = 'params.yaml'

def load_data(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception as e:
        raise Exception(f"Error loading data from {path}: {e}")

def load_params(path: str) -> float:
    try:
        with open(path, 'r') as f:
            params = yaml.safe_load(f)
        
        return params
    except Exception as e:
        raise Exception(f"Error loading paramters from {path}: {e}")

def split_X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    try:
        X = df.drop(columns=["Potability"])
        y = df['Potability']

        return X, y
    except Exception as e:
        raise Exception(f"Error splitting data: {e}")
    
def load_model(path: str):
    try:
        with open(path, 'rb') as f:
            model = pickle.load(f)
        
        return model
    except Exception as e:
        raise Exception(f"Error loading model from {path}: {e}")

def test_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    try:
        params = load_params(PARAMS_PATH)
        test_size = params['data_collection']['test_size']
        n_estimators = params['model_building']['n_estimators']

        y_hat = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_hat)
        precision = precision_score(y_test, y_hat)
        recall = recall_score(y_test, y_hat)
        f1 = f1_score(y_test, y_hat)

        with Live(save_dvc_exp=True) as live:
            live.log_metric('accuracy', accuracy)
            live.log_metric('precision', precision)
            live.log_metric('recall', recall)
            live.log_metric('f1', f1)

            live.log_param('test_size', test_size)
            live.log_param('n_estimators', n_estimators)

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

        return metrics
    except Exception as e:
        raise Exception(f"Error evaluating model: {e}")

def save_metrics(metrics: dict, path: str) -> None:
    try:
        with open('metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)
    except Exception as e:
        raise Exception(f"Error saving metrics to {path}: {e}")

def main():
    PROCESSED_DATA_PATH = os.path.join('data', 'processed', 'test_processed.csv')
    MODEL_PATH = 'model.pkl'
    METRICS_PATH = 'metrics.json'

    try:
        test_df = load_data(PROCESSED_DATA_PATH)

        X_test, y_test = split_X_y(test_df)

        model = load_model(MODEL_PATH)

        metrics = test_model(model, X_test, y_test)

        save_metrics(metrics, METRICS_PATH)
    except Exception as e:
        raise Exception(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
    