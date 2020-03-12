import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


def feature_extraction(data, short_window, long_window, column_name):

    data['MA_short'] = data[column_name].rolling(window=short_window).mean()
    data['MA_long'] = data[column_name].rolling(window=long_window).mean()

    data['ft_1'] = (data[column_name]-data['MA_short'])/data['MA_short']
    data['ft_2'] = (data[column_name] - data['MA_long']) / data['MA_long']

    data['ft_3'] = data['MA_short'].pct_change()
    data['ft_4'] = data['MA_long'].pct_change()

    data['ft_5'] = (data[column_name] - data[column_name].rolling(window=short_window).min()) / data[column_name].rolling(
        window=short_window).max()
    data['ft_6'] = (data[column_name] - data[column_name].rolling(window=long_window).min()) / data[column_name].rolling(
        window=long_window).max()

    data = data.dropna()
    cols = ['ft_1', 'ft_2', 'ft_3', 'ft_4', 'ft_5', 'ft_6']
    #data = data.loc[:, cols + [COL]]

    # Normalization and discretization
    data.loc[:, cols] = (data.loc[:, cols]-data.loc[:, cols].mean())/data.loc[:, cols].std()
    bins = [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]

    for c in cols:
        new_cols = []
        for i, b in enumerate(bins):
            new_cols.append(i)
            data.loc[:, new_cols[-1]] = (data.loc[:, c]-b).abs()
        data.loc[:, c] = data.loc[:, new_cols].idxmin(axis=1)
        data.loc[:, c] = data.loc[:, c].apply(lambda x: bins[int(x)])

    data = data.drop(columns=new_cols)

    data = data.reset_index()

    return data

#    data = data.iloc[5000:5050, :]

#    plt.figure()
#    plt.title('Feature Extraction')
#    x = range(len(data))
#    plt.plot(x, data['ft_1'])
#    plt.plot(x, data['ft_3'])
#    plt.plot(x, data['ft_5'])
#    plt.legend(['Ft 1', 'Ft 2', 'Ft 5'])

#    plt.show()

