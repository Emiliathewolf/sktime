import pandas as pd
import numpy as np
from sktime.datasets.base import _load_dataset
import matplotlib.pylab as plt
import matplotlib
import tkinter

def test():
    matplotlib.use('TkAgg')
    df = _load_dataset("PLAID", None, False)
    # We need this later
    cols = df.columns

    # Convert all of our values into lists
    querys = df.values.tolist()
    querydata = []
    for row in querys:
        querydata.append(row[0].tolist())

    # Collect all the class values to a sepperate list
    # for usage later
    classes = []
    for row in querys:
        classes.append(row[1])

    # Get longest element to be our candidate
    candidate = []
    for row in querydata:
        if len(row) > len(candidate):
            candidate = row

    #print("Before scaling...")
    #for x in range(0, 15):
    #    print(len(querydata[x]))

    # Scale our data
    scaleddata = []
    for row in querydata:
        scaleddata.append(us(row, candidate))

    #print("After scaling...")
    #for x in range(0, 15):
    #    print(len(scaleddata[x]))

    # Crate the new dataframe from our scaled elements.
    result = pd.DataFrame(columns=cols)
    for row, classval in zip(scaleddata, classes):
        result.append(row, classval)


    make_graphs(querydata, scaleddata)

def make_graphs(querydata, scaleddata, saveloc = None, name = None):
    fig, (ax1, ax2) = plt.subplots(2, figsize=(20, 10), dpi=100)
    fig.suptitle('Scaled vs Unscaled Based on Longest Length')
    #BEFORE 5 aand 6 ideal
    ax1.set_title('Unscaled')
    ax1.plot(querydata[6])
    #AFTER
    ax2.set_title('Scaled')
    ax2.plot(scaleddata[6])

    if len(querydata) > len(scaleddata):
        ax2.sharex(ax1)
    else:
        ax1.sharex(ax2)
    #plt.figure(figsize=(8, 6), dpi=800)
    if name is None:
        name = "Figure.png"
    if saveloc is None:
        saveloc = "../../../Graphs/" + name
    else:
        saveloc = saveloc + name
    plt.savefig(saveloc)
    plt.show()



# Implemented algorithms found here
# https://www.semanticscholar.org/paper/Efficiently-Finding-Arbitrarily-Scaled-Patterns-in-Keogh/7bace05f90a0d5352a7789e14cfaf3f5442b17ac


# This doesn't always produce the best match, and since we want the highest
# quality we should check that
def us(query, candidate, p=None):
    n = len(query)
    m = len(candidate)

    QP = []
    #print(f"n: {n}\tm: {m}")
    if len(query) == len(candidate):
        return query
    if p is None:
        p = m
    # p / n = scaling factor
    for j in range(p):
        QP.append(query[int(j*(n/p))])
    return QP


# Yes I know sklearn has this, no I don't care
def euclidian_distances(q, c):
    # doesn't matter which one since they're the same length
    n = len(q)
    EC = []
    for i in range(n):
        EC.append((q[i] - c[i])**2)
    return EC


def compare_scaling(query, candidate):
    best_match_value = float('inf')
    best_scaling_factor = None
    best_match = None
    best_p_value = None

    n = len(query)
    m = len(candidate)
    for p in range(n, m):
        QP = us(query, candidate, p)
        # Compare like sizes
        dist = sum(euclidian_distances(QP, candidate[:p]))
        if dist < best_match_value:
            best_match_value = dist
            best_scaling_factor = p/n
            best_p_value = p
            best_match = QP
    return [best_match_value, best_scaling_factor, best_match, best_p_value]


test()
