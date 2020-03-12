import matplotlib.pyplot as plt
import numpy as np

def network_generation(data):

    data['V'] = data['ft_1'].astype('str') + '_' + data['ft_2'].astype('str') + '_' + data['ft_3'].astype('str') + '_' + \
                data['ft_4'].astype('str') + '_' + data['ft_5'].astype('str') + '_' + data['ft_6'].astype('str')

    data['W_'] = -1
    node_id = 0

    for t, row in data.iterrows():
        if data.loc[t, 'W_'] == -1:
            data.loc[data['V'] == row['V'], 'W_'] = node_id
            node_id += 1

    M = np.zeros((node_id, node_id))

    for t in range(len(data)-1):
        x = data.loc[t, 'W_']
        y = data.loc[t + 1, 'W_']
        M[x, y] += 1

    return data, M

