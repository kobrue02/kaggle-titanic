import pandas as pd

from aux import features
from preprocessing import process


def save(model):
    solutions = pd.read_csv("test.csv")
    proc = process(solutions)
    test = proc[features]

    solutions["Survived"] = model.predict(test)
    solutions = solutions[["PassengerId", "Survived"]]

    solutions.to_csv("solutions.csv", index=False)
    print("Success.")
