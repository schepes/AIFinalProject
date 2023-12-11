from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

class BaselineModel:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.models = {
            'GaussianNB': GaussianNB(),
            'LogisticRegression': LogisticRegression(max_iter=10000),
            'DecisionTree': DecisionTreeClassifier(),
            'RandomForest': RandomForestClassifier(),
            'AdaBoost': AdaBoostClassifier(),
            'SVC': SVC()
        }

    def train_and_evaluate(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        for name, model in self.models.items():
            # Convert sparse matrix to dense array
            X_train_dense = X_train.toarray()
            X_test_dense = X_test.toarray()

            model.fit(X_train_dense, y_train)
            predictions = model.predict(X_test_dense)
            accuracy = accuracy_score(y_test, predictions)
            report = classification_report(y_test, predictions)
            print(f"Model: {name}")
            print(f"Accuracy: {accuracy}")
            print(f"Report: {report}\n")
		
        print("Finished")
