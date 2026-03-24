import os
import pandas as pd
from sklearn.model_selection import train_test_split
import yaml

DATA_PATH = os.path.join('data', 'water_potability.csv')
PARAMS_PATH = 'params.yaml'
RAW_DATA_PATH = os.path.join('data', 'raw')

def load_data(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception as e:
        raise Exception(f"Error loading data from {path}: {e}")

def load_params(path: str) -> float:
    try:
        with open(path, 'r') as f:
            params = yaml.safe_load(f)
        
        return params['data_collection']['test_size']
    except Exception as e:
        raise Exception(f"Error loading paramters from {path}: {e}")

def split_data(df: pd.DataFrame, test_size: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        return train_test_split(df, test_size=test_size, random_state=42)
    except Exception as e:
        raise Exception(f"Error splitting data from {e}")

def save_data(df: pd.DataFrame, path: str) -> None:
    try:
        df.to_csv(path, index=False)
    except Exception as e:
        raise Exception(f"Error saving data to {path}: {e}")

def main():

    try:
        df = load_data(DATA_PATH)

        test_size = load_params(PARAMS_PATH)

        train_df, test_df = split_data(df, test_size)

        os.makedirs(RAW_DATA_PATH)

        save_data(train_df, os.path.join(RAW_DATA_PATH, 'train.csv'))
        save_data(test_df, os.path.join(RAW_DATA_PATH, 'test.csv'))
    except Exception as e:
        raise Exception(f"An error occurred: {e}")

if __name__ == '__main__':
    main()