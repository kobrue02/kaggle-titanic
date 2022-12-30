from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB, ComplementNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

features = ["Age", "Sex", "Cabin", "Ticket"]

models = {"MLP": MLPClassifier(),
          "Bernoulli": BernoulliNB(),
          "Gaussian": GaussianNB(),
          "Complement": ComplementNB(),
          "Random Forest": RandomForestClassifier(),
          "Decision Tree": DecisionTreeClassifier(),
          "LogReg": LogisticRegression()}

vc = VotingClassifier(estimators=[('lgr', LogisticRegression()),
                                  ('bnb', BernoulliNB()),
                                  ('mlp', MLPClassifier()),
                                  ('rdf', RandomForestClassifier())])
