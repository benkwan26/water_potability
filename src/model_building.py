import numpy as np
import os
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
import yaml

N_ESTIMATORS = yaml.safe_load(open('params.yaml'))['model_building']['n_estimators']

train_df = pd.read_csv("./data/processed/train_processed.csv")

X_train = train_df.drop(columns=["Potability"])
y_train = train_df['Potability']

clf = RandomForestClassifier(n_estimators=N_ESTIMATORS)
clf.fit(X_train, y_train)

pickle.dump(clf, open('model.pkl', 'wb'))