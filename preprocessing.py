import pandas as pd

from vars import features

seed = 42


def process(data):

    data = _sex(data)
    data = _age(data)
    data = _cabin(data)
    data = _embarked(data)
    data = _ticket(data)
    data = _fare(data)

    # remove useless column
    data = data.drop(["Name", "PassengerId", "SibSp", "Parch"], axis=1)

    data = data.astype({"Age": int, "Fare": int})

    return data


def _sex(data):

    # changing Sex column to numeric values
    data.Sex = data.Sex.replace("male", 0)
    data.Sex = data.Sex.replace("female", 1)

    return data


def _age(data):
    data["Age"] = data['Age'].fillna(data['Age'].mean())
    data["Age"] = data["Age"].astype('int')
    for i in range(0, max(data['Age']) + 1):
        if i <= data['Age'].median():
            data['Age'] = data['Age'].replace(i, 0)
        else:
            data['Age'] = data['Age'].replace(i, 1)
    return data


def _cabin(data):
    # removing the cabin number code
    data['Cabin'] = data['Cabin'].str.replace('\d+', '')

    # changing alpha values in cabin column to 1
    data["Cabin"] = data["Cabin"].replace(["A", "B", "C", "D", "E", "F", "G", "T"], "1", inplace=True)
    data["Cabin"] = data["Cabin"].str[0]

    # get rid of whitespaces and nan values
    data["Cabin"] = data["Cabin"].str.replace(" ", "0")
    data = data.fillna(0)
    data["Cabin"] = data["Cabin"].astype('int')
    return data


def _ticket(data):

    # turning the tickets into numeric values
    data['Ticket'] = data['Ticket'].str.replace('\d+', '')
    data['Ticket'] = data['Ticket'].str.replace('/', '')
    data['Ticket'] = data['Ticket'].str.replace('.', '')

    tickets = set(data.Ticket)

    tkt = {}
    for i, ticket in enumerate(tickets):
        tkt[ticket] = i

    for k, v in tkt.items():
        if v == 0:
            data['Ticket'] = data['Ticket'].replace(k, v)
        else:
            data['Ticket'] = data['Ticket'].replace(k, 1)

    return data


def _embarked(data):
    # replacing embarked with numerics
    data.Embarked = data.Embarked.replace("C", 1)
    data.Embarked = data.Embarked.replace("Q", 2)
    data.Embarked = data.Embarked.replace("S", 3)
    return data


def _fare(data):
    data["Fare"] = data["Fare"].fillna(data["Fare"].median())
    data["Fare"] = data["Fare"].astype('int')
    for i in range(0, max(data['Fare']) + 1):
        if i <= data['Fare'].median():
            data['Fare'] = data['Fare'].replace(i, 0)
        else:
            data['Fare'] = data['Fare'].replace(i, 1)
    return data


def undersample(data, label):

    s0 = len(data[data[label] == 0])
    s1 = len(data[data[label] == 1])
    cut = min(s0, s1)
    class0 = data[data[label] == 0].sample(n=cut, random_state=seed)
    class1 = data[data[label] == 1].sample(n=cut, random_state=seed)

    new_data = pd.concat([class0, class1])
    return new_data
