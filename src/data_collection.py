import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import yaml

TEST_SIZE = yaml.safe_load(open('params.yaml'))['data_collection']['test_size']

df = pd.read_csv("./data/water_potability.csv")

train_data, test_data = train_test_split(df, test_size=TEST_SIZE, random_state=42)

DATA_PATH = os.path.join('data', 'raw')
os.makedirs(DATA_PATH)
train_data.to_csv(os.path.join(DATA_PATH, 'train.csv'), index=False)
test_data.to_csv(os.path.join(DATA_PATH, 'test.csv'), index=False)