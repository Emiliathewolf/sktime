from scipy.spatial.distance import pdist
import numpy as np
import pandas as pd

def us(query, p):
    """
    Scales a query to a given length, p
    :param query: Time Series to be scaled
    :param p: Length to scale to
    :return: QP, a numpy array containing the scaled query
    """
    n = query.size
    QP = np.empty(shape=(n, p))
    # p / n = scaling factor
    for i in range(n):
        curQ = query.iloc[i][0]
        for j in range(p):
            try:
                QP[i][j] = (curQ[int(j * (len(curQ) / p))])
            except Exception as e:
                print(e)
    return QP

def euclidian_distances(q):
    ED = sum(pdist(np.array(q), 'sqeuclidean'))
    return ED

def compare_scaling(query, min = None, max = None):
    """
    Compares the euclidean distances of multiple scale lengths for an array of time series, and returns the scaled
    query with the lowest euclidean distance
    :param query: An array of time series to be scaled
    :param min: Minimum length to scale to
    :param max: Maximum length to scale to
    :return: The query scaled to the optimal length between the min and max
    """
    best_match_value = float('inf')
    best_match = None

    if max == None:
        max = 0
        for i in range(query.size):
            if query.iloc[i][0].size > max:
                max = query.iloc[i][0].size

    if min == None:
        min = 0
        for i in range(query.size):
            if query.iloc[i][0].size < min:
                min = query.iloc[i][0].size
    n = min
    m = max
    #Parallel probs best
    for p in range(n, m):
        QP = us(query, p)
        dist = euclidian_distances(QP)  # Compare like sizes
        if dist < best_match_value:
            best_match_value = dist
            best_match = QP
    #Reshuffle so it fits the required structure
    ret = []
    for i in range(query.size):
        ret.append([best_match[i]])
    return pd.DataFrame(ret)

def pad_zero(query, direction, scale_size = None):
    """
    Pads either the prefix or suffix of time series data with zeros, up to a length defined by scale_size
    :param query: An array of time series to be scaled
    :param direction: Either prefix or suffix, determines what part to pad
    :param scale_size: Size to scale up to
    :return: A scaled array of time series
    """
    #Set size if needed
    if scale_size == None:
        max = 0
        for i in range(query.size):
            if query.iloc[i][0].size > max:
                max = query.iloc[i][0].size
        scale_size = max
    else:
        for i in range(query.size):
            if query.iloc[i][0].size > scale_size:
                #This can't scale down
                raise ValueError("Scale size must be greater than the longest series")

    #Scale needed values
    scaled = []
    for i in range(query.size):
        curQ = query.iloc[i][0].tolist()
        length = query.iloc[i][0].size
        for j in range(scale_size - length):
            try:
                if direction == 'prefix':
                    # Insert 0 at pos 0
                    curQ.insert(0,0)
                elif direction == 'suffix':
                    curQ.append(0)
            except Exception as e:
                print(e)
        scaled.append(pd.Series(curQ))

    #Reshuffle so it fits the required structure
    ret = []
    for i in range(query.size):
        ret.append([scaled[i]])
    return pd.DataFrame(ret)

def pad_noise(query, direction, scale_size = None):
    """
    Pads either the prefix or suffix of time series data with random noise, up to a length defined by scale_size
    :param query: An array of time series to be scaled
    :param direction: Either prefix or suffix, determines what part to pad
    :param scale_size: Size to scale up to
    :return: A scaled array of time series
    """
    #Set size if needed
    if scale_size == None:
        max = 0
        for i in range(query.size):
            if query.iloc[i][0].size > max:
                max = query.iloc[i][0].size
        scale_size = max
    else:
        for i in range(query.size):
            if query.iloc[i][0].size > scale_size:
                #This can't scale down
                raise ValueError("Scale size must be greater than the longest series")

    #Scale needed values
    scaled = []
    for i in range(query.size):
        curQ = query.iloc[i][0].tolist()
        length = query.iloc[i][0].size

        # get np mean, np std
        mean = np.mean(curQ)
        std = np.std(curQ)
        noise = np.random.normal(mean, std, scale_size - length)
        noise = noise.tolist()
        noise = list(map(abs, noise))
        for j in range(scale_size - length):
            try:
                if direction == 'prefix':
                    # Insert 0 at pos 0
                    curQ.insert(0, noise[j])
                elif direction == 'suffix':
                    curQ.append(noise[j])
            except Exception as e:
                print(e)
        scaled.append(pd.Series(curQ))

    #Reshuffle so it fits the required structure
    ret = []
    for i in range(query.size):
        ret.append([scaled[i]])
    return pd.DataFrame(ret)