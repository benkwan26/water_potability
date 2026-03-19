import pandas as pd
import numpy as np
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

test_df = pd.read_csv("./data/processed/test_processed.csv")

X_test = test_df.iloc[:, 0: -1].values
y_test = test_df.iloc[:, -1].values

model = pickle.load(open('model.pkl', 'rb'))

y_hat = model.predict(X_test)

accuracy = accuracy_score(y_test, y_hat)
precision = precision_score(y_test, y_hat)
recall = recall_score(y_test, y_hat)
f1 = f1_score(y_test, y_hat)

metrics = {
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1': f1
}

with open('metrics.json', 'w') as f:
    json.dump(metrics, f, indent=4)