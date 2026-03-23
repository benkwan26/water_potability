import os
import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception as e:
        raise Exception(f"Error loading data from {path}: {e}")

def fill_missing_with_median(df: pd.DataFrame) -> pd.DataFrame:
    try:
        for column in df.columns:
            if df[column].isnull().any():
                df[column].fillna(df[column].median(), inplace=True)
        
        return df
    except Exception as e:
        raise Exception(f"Error filling missing values: {e}")

def save_data(df: pd.DataFrame, path: str) -> None:
    try:
        df.to_csv(path, index=False)
    except Exception as e:
        raise Exception(f"Error saving data to {path}: {e}")

def main():
    RAW_DATA_PATH = os.path.join('data', 'raw')
    PROCESSED_DATA_PATH = os.path.join('data', 'processed')

    try:
        train_df = load_data(os.path.join(RAW_DATA_PATH, 'train.csv'))
        test_df = load_data(os.path.join(RAW_DATA_PATH, 'test.csv'))

        train_df_processed = fill_missing_with_median(train_df)
        test_df_processed = fill_missing_with_median(test_df)

        os.makedirs(PROCESSED_DATA_PATH)

        save_data(train_df_processed, os.path.join(PROCESSED_DATA_PATH, 'train_processed.csv'))
        save_data(test_df_processed, os.path.join(PROCESSED_DATA_PATH, 'test_processed.csv'))
    except Exception as e:
        raise Exception(f"An error occurred: {e}")

if __name__ == '__main__':
    main()