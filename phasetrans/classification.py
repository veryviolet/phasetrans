import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


def trend_classification(data, column_name, with_flat=False):

    comm_cp = {i: [] for i in list(data['community'].drop_duplicates())}

    last_val = -1
    for ind, row in data.iloc[:-1].iterrows():
        if row['community'] != last_val:
            xi = row[column_name]
        elif row['community'] != data.loc[ind + 1, 'community']:
            xf = row[column_name]
            comm_cp[row['community']].append((xf - xi) / xi)

        last_val = row['community']

    for i, v in comm_cp.items():

        if with_flat:
            if not v:
                comm_cp[i] = 'FT'
            else:
                c_mean = np.mean(v)
                c_std = np.std(v)

                if np.abs(c_mean) > c_std:
                    comm_cp[i] = 'FT'
                elif c_mean > 0:
                    comm_cp[i] = 'UP'
                else:
                    comm_cp[i] = 'DW'
        else:
            avg = sum(v) / len(v) if v else 0

            if avg > 0:
                comm_cp[i] = 'UP'
            else:
                comm_cp[i] = 'DW'

    data['trend'] = data['community'].apply(lambda x: comm_cp[x])

    if with_flat:
        data['trend_int'] = 2
        data.loc[data['trend'] == 'UP', 'trend_int'] = 0
        data.loc[data['trend'] == 'DW', 'trend_int'] = 1
    else:
        data['trend_int'] = 0
        data.loc[data['trend'] == 'DW', 'trend_int'] = 1

    return data

    


