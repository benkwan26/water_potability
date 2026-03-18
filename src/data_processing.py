import pandas as pd
import numpy as np
import os

train_df = pd.read_csv("./data/raw/train.csv")
test_df = pd.read_csv("./data/raw/test.csv")

def fill_missing_with_median(df):
    for column in df.columns:
        if df[column].isnull().any():
            df[column].fillna(df[column].median(), inplace=True)
    
    return df

train_df_processed = fill_missing_with_median(train_df)
test_df_processed = fill_missing_with_median(test_df)

DATA_PATH = os.path.join('data', 'processed')
os.makedirs(DATA_PATH)
train_df_processed.to_csv(os.path.join(DATA_PATH, 'train_processed.csv'), index=False)
test_df_processed.to_csv(os.path.join(DATA_PATH, 'test_processed.csv'), index=False)