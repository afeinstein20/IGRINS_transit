import os, sys
import numpy as np

def read_input():
    a = {}

    with open('params.inp') as f:
        for line in f:
            (k,v) = line.split(',')
            if k != 'model':
                a[k] = float(v[:-2])
            else:
                a[k] = v[:-1]
    return a
