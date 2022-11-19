import os, sys
import numpy as np

def read_input(filename):
    a = {}

    with open(filename) as f:
        for line in f:
            (k,v) = line.split(',')
            try:
                a[k] = np.float(v[:-1])
            except:
                a[k] = v[:-1]
    return a
