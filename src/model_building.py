import os
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
import yaml

def load_data(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception as e:
        raise Exception(f"Error loading data from {path}: {e}")

def load_params(path: str) -> float:
    try:
        with open(path, 'r') as f:
            params = yaml.safe_load(f)
        
        return params['model_building']['n_estimators']
    except Exception as e:
        raise Exception(f"Error loading paramters from {path}: {e}")

def split_X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    try:
        X = df.drop(columns=["Potability"])
        y = df['Potability']

        return X, y
    except Exception as e:
        raise Exception(f"Error splitting data: {e}")

def train_model(X: pd.DataFrame, y: pd.Series, n_estimators: int) -> RandomForestClassifier:
    try:
        clf = RandomForestClassifier(n_estimators=n_estimators)
        clf.fit(X, y)

        return clf
    except Exception as e:
        raise Exception(f"Error training model: {e}")

def save_model(model: RandomForestClassifier, path: str) -> None:
    try:
        pickle.dump(model, open('model.pkl', 'wb'))
    except Exception as e:
        raise Exception(f"Error saving model to {path}: {e}")

def main():
    PROCESSED_DATA_PATH = os.path.join('data', 'processed', 'train_processed.csv')
    PARAMS_PATH = 'params.yaml'
    MODEL_PATH = 'model.pkl'

    try:
        train_df = load_data(PROCESSED_DATA_PATH)

        n_estimators = load_params(PARAMS_PATH)

        X_train, y_train = split_X_y(train_df)

        model = train_model(X_train, y_train, n_estimators)

        save_model(model, MODEL_PATH)
    except Exception as e:
        raise Exception(f"An error occurred: {e}")

if __name__ == '__main__':
    main()