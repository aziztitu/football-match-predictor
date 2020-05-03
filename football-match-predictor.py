from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from time import time
from sklearn.metrics import f1_score

# Utility Functions


def train_classifier(clf, X_train, y_train):
    start = time()
    clf.fit(X_train, y_train)
    end = time()
    print("Model trained in {:2f} seconds".format(end-start))


def predict_labels(clf, features, target):
    start = time()
    y_pred = clf.predict(features)
    end = time()
    print("Made Predictions in {:2f} seconds".format(end-start))

    acc = sum(target == y_pred) / float(len(y_pred))

    return f1_score(target, y_pred, average='micro'), acc


def model(clf, X_train, y_train, X_test, y_test):
    train_classifier(clf, X_train, y_train)

    f1, acc = predict_labels(clf, X_train, y_train)
    print("Training Info:")
    print("-" * 20)
    print("F1 Score:{}".format(f1))
    print("Accuracy:{}".format(acc))

    f1, acc = predict_labels(clf, X_test, y_test)
    print("Test Metrics:")
    print("-" * 20)
    print("F1 Score:{}".format(f1))
    print("Accuracy:{}".format(acc))


def derive_clean_sheet(src):
    arr = []
    n_rows = src.shape[0]

    for data in range(n_rows):

        #[HTHG, HTAG]
        values = src.iloc[data].values
        cs = [0, 0]

        if values[0] == 0:
            cs[1] = 1

        if values[1] == 0:
            cs[0] = 1

        arr.append(cs)

    return arr


# Data gathering

data_files = [
    'data/spanish-la-liga_zip/data/season-0910_csv.csv',
    'data/spanish-la-liga_zip/data/season-1011_csv.csv',
    'data/spanish-la-liga_zip/data/season-1112_csv.csv',
    'data/spanish-la-liga_zip/data/season-1213_csv.csv',
    'data/spanish-la-liga_zip/data/season-1314_csv.csv',
    'data/spanish-la-liga_zip/data/season-1415_csv.csv',
    'data/spanish-la-liga_zip/data/season-1516_csv.csv',
    'data/spanish-la-liga_zip/data/season-1617_csv.csv',
    'data/spanish-la-liga_zip/data/season-1718_csv.csv',
    'data/spanish-la-liga_zip/data/season-1819_csv.csv',
]

data_frames = []

for data_file in data_files:
    data_frames.append(pd.read_csv(data_file))

data = pd.concat(data_frames).reset_index()
print(data)

# Pre processing

input_filter = ['home_encoded', 'away_encoded', 'HS',
                'AS', 'HST', 'AST', 'HTCT', 'ATCT', 'HR', 'AR']
output_filter = ['FTR']

cols_to_consider = []
cols_to_consider.extend(input_filter)
cols_to_consider.extend(output_filter)

encoder = LabelEncoder()
home_encoded = encoder.fit_transform(data['HomeTeam'])
away_encoded = encoder.fit_transform(data['AwayTeam'])
wins_encoded = encoder.fit_transform(data['FTR'])

data['home_encoded'] = home_encoded
data['away_encoded'] = away_encoded
data['wins_encoded'] = wins_encoded

htg_df = data[['HTHG', 'HTAG']]
cs_data = derive_clean_sheet(htg_df)
cs_df = pd.DataFrame(cs_data, columns=['HTCT', 'ATCT'])

data = pd.concat([data, cs_df], axis=1)

data = data[cols_to_consider]

print(data[data.isna().any(axis=1)])
data = data.dropna(axis=0)

# Training & Testing

X = data[input_filter]
Y = data['FTR']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

svc_classifier = SVC(random_state=100, kernel='rbf')
one_vs_all_clf = LogisticRegression(multi_class='ovr', max_iter=500)
nbClassifier = GaussianNB()
dtClassifier = DecisionTreeClassifier()
rfClassifier = RandomForestClassifier()

print("Support Vector Machine")
print("-" * 20)
model(svc_classifier, X_train, Y_train, X_test, Y_test)

print()
print("Logistic Regression one vs All Classifier")
print("-" * 20)
model(one_vs_all_clf, X_train, Y_train, X_test, Y_test)

print()
print("Gaussain Naive Bayes Classifier")
print("-" * 20)
model(nbClassifier, X_train, Y_train, X_test, Y_test)

print()
print("Decision Tree Classifier")
print("-" * 20)
model(dtClassifier, X_train, Y_train, X_test, Y_test)

print()
print("Random Forest Classifier")
print("-" * 20)
model(rfClassifier, X_train, Y_train, X_test, Y_test)
