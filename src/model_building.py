import pandas as pd
import numpy as np
import os
import pickle
from sklearn.ensemble import RandomForestClassifier

train_df = pd.read_csv("./data/processed/train_processed.csv")

X_train = train_df.drop(columns=["Potability"])
y_train = train_df['Potability']

clf = RandomForestClassifier()
clf.fit(X_train, y_train)

pickle.dump(clf, open('model.pkl', 'wb'))