from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB, ComplementNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

import numpy as np

features = ["Age", "Sex", "Cabin", "Fare"]

mlp = MLPClassifier(alpha=0.1, hidden_layer_sizes=10, max_iter=1600, solver="lbfgs")
gnb = GaussianNB(var_smoothing=0.657933224657568)
lgr = LogisticRegression(C=0.1, penalty="l2")

models = {"MLP": mlp,
          "Bernoulli": BernoulliNB(),
          "Gaussian": gnb,
          "Complement": ComplementNB(),
          "Random Forest": RandomForestClassifier(),
          "Decision Tree": DecisionTreeClassifier(),
          "LogReg": lgr}

tuner = {"mlp":
             {'solver': ['lbfgs'], 'max_iter': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000],
              'alpha': 10.0 ** -np.arange(1, 10), 'hidden_layer_sizes': np.arange(10, 15)},
         "bnb":
             {'var_smoothing': np.logspace(0, -9, num=100)},
         "lgr":
             {"C": np.logspace(-3, 3, 7), "penalty": ["l1", "l2"]},
         "rfc":
             {'bootstrap': [True, False],
              'max_depth': [10, 50, 90],
              'max_features': ['auto', 'sqrt'],
              'min_samples_leaf': [1, 2, 4],
              'min_samples_split': [2, 5, 10],
              'n_estimators': [200, 1000]}
         }

fm = BernoulliNB()

votingC = VotingClassifier(estimators=[('lgr', lgr),
                                       ('gnb', gnb),
                                       ('mlp', mlp),
                                       ('rdf', RandomForestClassifier())])
