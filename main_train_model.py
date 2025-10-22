import os
import joblib
import pandas as pd 
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

from preprocessing import Preprocessor

csv_path = 'loan_approval.csv'
output_model_filename = 'gradient_model.pkl'

df = Preprocessor(csv_path)
df = df.clean_df()

le = LabelEncoder()
df['loan_approved'] = le.fit_transform(df['loan_approved'])
X = df.drop(columns=['loan_approved'])
y = df['loan_approved']

print(df)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
clf = GradientBoostingClassifier(n_estimators=1000, learning_rate=0.05, max_depth=1, random_state=6).fit(X_train, y_train)
print(clf.score(X_test, y_test))

joblib.dump(clf, output_model_filename)