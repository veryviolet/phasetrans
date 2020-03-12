import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


class PredictiveModel:

    def train(self, data, split, validation, with_flat=False):

        if with_flat:
            data = data[data['trend'] != 'FT']

        self.X_train, self.X_test, self.X_dev = self.split_data(data, split)
        if validation == 'OOS':
            self.accs = self.trend_forecasting()
        elif validation == 'CROSS_VAL':
            self.accs = self.trend_forecasting_cross_val(data)
        else:
            raise Exception("Error: Validation options are OOS or CROSS_VAL")

        max_acc = 0
        model = None
        for key, val in self.accs.items():
            if val > max_acc:
                max_acc = val
                model = key

        return model, max_acc

    def split_data(self, data, split):

        proportions = np.array(split)
        proportions = (len(data) * proportions).astype('int32')
        X_train = data.iloc[:proportions[0]]
        X_test = data.iloc[proportions[0]:proportions[0] + proportions[1]]
        X_dev = data.iloc[proportions[0] + proportions[1]:]

        return X_train, X_test, X_dev

    def trend_forecasting(self):

        cols = ['ft_1', 'ft_2', 'ft_3', 'ft_4', 'ft_5', 'ft_6']
        accs = {}

        model = GaussianNB()
        model.fit(self.X_train[cols], self.X_train['trend_int'])
        accs['Naive_Bayes'] = accuracy_score(self.X_test['trend_int'], model.predict(self.X_test[cols]))
        p =  model.predict(self.X_dev[cols])
        print("Näive Bayes", "Prediction", 'UP' if p[-1] == 0 else 'DOWN', "Accuracy on test:", round(accs['Naive_Bayes'], 4))

        model = DecisionTreeClassifier(random_state=101)
        model.fit(self.X_train[cols], self.X_train['trend_int'])
        accs['Decision_Tree'] = accuracy_score(self.X_test['trend_int'], model.predict(self.X_test[cols]))
        p = model.predict(self.X_dev[cols])
        print("Decision Tree", "Prediction", 'UP' if p[-1] == 0 else 'DOWN', "Accuracy on test:", round(accs['Decision_Tree'], 4))

        model = MLPClassifier(random_state=101)
        model.fit(self.X_train[cols], self.X_train['trend_int'])
        accs['MLP'] = accuracy_score(self.X_test['trend_int'], model.predict(self.X_test[cols]))
        p = model.predict(self.X_dev[cols])
        print("MLP", "Prediction", 'UP' if p[-1] == 0 else 'DOWN', "Accuracy on test:", round(accs['MLP'], 4))

        try:
            model = SVC(random_state=101, max_iter=1000)
            model.fit(self.X_train[cols], self.X_train['trend_int'])
            accs['SVM'] = accuracy_score(self.X_test['trend_int'], model.predict(self.X_test[cols]))
            p = model.predict(self.X_dev[cols])
            print("SVM", "Prediction", 'UP' if p[-1] == 0 else 'DOWN', "Accuracy on test:", round(accs['SVM'], 4))
        except Exception as e:
            print("Error with SVC:", str(e))
            accs['SVM'] = 0

        model = KNeighborsClassifier(2)
        model.fit(self.X_train[cols], self.X_train['trend_int'])
        accs['KNN'] = accuracy_score(self.X_test['trend_int'], model.predict(self.X_test[cols]))
        p = model.predict(self.X_dev[cols])
        print("KNN", "Prediction", 'UP' if p[-1] == 0 else 'DOWN', "Accuracy on test:", round(accs['KNN'], 4))

        return accs

    def trend_forecasting_cross_val(self, data):
        cols = ['ft_1', 'ft_2', 'ft_3', 'ft_4', 'ft_5', 'ft_6']
        accs = {}

        kf = KFold(n_splits=10)

        model_NB = GaussianNB()
        model_DT = DecisionTreeClassifier(random_state=101)
        model_MLP = MLPClassifier(random_state=101)
        model_SVC = SVC(random_state=101, max_iter=1000)
        model_KN = KNeighborsClassifier(2)

        for train_index, test_index in kf.split(self.X_train):
            data_train = data.iloc[train_index, :]
            model_NB.fit(data_train[cols], data_train['trend_int'])
            model_DT.fit(data_train[cols], data_train['trend_int'])
            model_MLP.fit(data_train[cols], data_train['trend_int'])
            try:
                model_SVC.fit(data_train[cols], data_train['trend_int'])
            except Exception as e:
                print("Error with SVC:", str(e))
            model_KN.fit(data_train[cols], data_train['trend_int'])


        accs['Naive_Bayes'] = np.mean(cross_val_score(model_NB, data[cols], data['trend_int'], cv=10))
        accs['Decision_Tree'] = np.mean(cross_val_score(model_DT, data[cols], data['trend_int'], cv=10))
        accs['MLP'] = np.mean(cross_val_score(model_MLP, data[cols], data['trend_int'], cv=10))
        accs['SVM'] = np.mean(cross_val_score(model_SVC, data[cols], data['trend_int'], cv=10))
        accs['KNN'] = np.mean(cross_val_score(model_KN, data[cols], data['trend_int'], cv=10))

        p = model_NB.predict(self.X_dev[cols])
        print("Näive Bayes", "Prediction", 'UP' if p[-1] == 0 else 'DOWN', "Accuracy on test:", round(accs['Naive_Bayes'], 4))
        p = model_NB.predict(self.X_dev[cols])
        print("Decision Tree", "Prediction", 'UP' if p[-1] == 0 else 'DOWN', "Accuracy on test:", round(accs['Decision_Tree'], 4))
        p = model_NB.predict(self.X_dev[cols])
        print("SVM", "Prediction", 'UP' if p[-1] == 0 else 'DOWN', "Accuracy on test:", round(accs['MLP'], 4))
        p = model_NB.predict(self.X_dev[cols])
        print("SVM", "Prediction", 'UP' if p[-1] == 0 else 'DOWN', "Accuracy on test:", round(accs['SVM'], 4))
        p = model_NB.predict(self.X_dev[cols])
        print("KNN", "Prediction", 'UP' if p[-1] == 0 else 'DOWN', "Accuracy on test:", round(accs['KNN'], 4))


        return accs
