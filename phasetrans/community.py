import numpy as np
import pandas as pd 


def communities_detection(data, A):
    A[A > 0] = 1
    m = A.sum() / 2
    ki = A.sum(axis=1)
    ai = ki / (2 * m)

    Q = (1 / (2 * m) * A.sum(axis=1) - (ai) ** 2).sum()
    dQ = 1 / (2 * m) * A - ((A.T * ki).T * ki) / ((2 * m) ** 2)

    Q = Q + dQ.sum()
    max_Q = Q

    communities = [[i] for i in range(len(ai))]
    optimum_communities = list(communities)
    min_size = int(len(optimum_communities)*0.2)

    dQs = []
    Qs = []

    cont = 0
    while len(ai) > min_size:

        dQ_triu = dQ.copy()
        np.fill_diagonal(dQ_triu, -100)
        Hi = np.argmax(dQ_triu, axis=1)
        H = np.max(dQ_triu, axis=1)
        Hj = np.argmax(H)
        Hi = Hi[Hj]

        rule1 = np.intersect1d(np.argwhere(dQ[Hi, :] > 0), np.argwhere(dQ[Hj, :] > 0))
        dQ[rule1, Hj] = dQ[Hj, rule1] = dQ[Hi, rule1] + dQ[Hj, rule1]
        rule2 = np.setdiff1d(np.argwhere(dQ[Hi, :] > 0), np.argwhere(dQ[Hj, :] > 0))
        dQ[rule2, Hj] = dQ[Hj, rule2] = dQ[Hi, rule2] - 2 * ai[Hj] * ai[rule2]
        rule3 = np.setdiff1d(np.argwhere(dQ[Hj, :] > 0), np.argwhere(dQ[Hi, :] > 0))
        dQ[rule3, Hj] = dQ[Hj, rule3] = dQ[Hj, rule3] - 2 * ai[Hi] * ai[rule3]

        if Hi == Hj:
            print("Sigue pasando", cont, Hi, Hj)
            # print(dQ)
            break

        cont += 1
        #if cont % 500 == 0:
        #    print(cont, len(ai), dQ.sum(), max_Q, len(optimum_communities))

        dQ = np.delete(dQ, Hi, 0)
        dQ = np.delete(dQ, Hi, 1)

        ai[Hj] = ai[Hj] + ai[Hi]
        ai = np.delete(ai, Hi)

        Q = Q + dQ.sum()
        dQs.append(dQ.sum())
        Qs.append(Q)

        communities[Hj] = communities[Hj] + communities[Hi]
        del communities[Hi]

        if Q > max_Q:
            max_Q = Q
            optimum_communities = list(communities)

        out_dict = {}
        for c, i in enumerate(optimum_communities):
            for j in i:
                out_dict[j] = c

        data['community'] = data['W_'].apply(lambda x: out_dict[x])

    return data

