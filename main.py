import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV

from preprocessing import process, undersample
from save_result import save
from vars import *

import warnings

warnings.filterwarnings("ignore")

titanic = pd.read_csv("train.csv")
titanic = process(titanic)


def do(data, job: str = "sr"):
    X = data[features]
    y = data["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    if job == "ens":
        _ensemble(X_train, X_test, y_train, y_test)

    if job == "comp":
        _compare(X_train, X_test, y_train, y_test)

    if job == "tune":
        _tune(X_train, y_train)

    if job == "sr":
        _solute(X, y)


def _compare(X_train, X_test, y_train, y_test):
    for name, model in models.items():
        print("-" * 32 + " " + name + " " + "-" * 32)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        print(classification_report(y_test, y_pred))


def _ensemble(X_train, X_test, y_train, y_test):
    vc.fit(X_train, y_train)
    y_pred = vc.predict(X_test)

    print(classification_report(y_test, y_pred))


def _solute(X, y):
    model = vc
    model.fit(X, y)
    save(model)


def _tune(X_train, y_train):
    test = RandomForestClassifier()
    GRID = tuner["rfc"]

    model = GridSearchCV(test, param_grid=GRID, verbose=2, n_jobs=1)
    model.fit(X_train, y_train)
    print(model.score(X_train, y_train))
    print(model.best_params_)


def charts(data):
    for label in features:
        print(data[label].nunique())
        print(set(data[label]))
        data = undersample(data, label)
        ax = sns.countplot(y=data[label][data["Survived"] == 0])
        plt.bar_label(ax.containers[0])
        plt.title(label, color="red")
        plt.show()


if __name__ == "__main__":
    # charts(titanic)
    do(titanic, job="tune")
