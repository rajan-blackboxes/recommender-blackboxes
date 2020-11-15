"""
Contains collaborative recommendation systems

"""
import numpy as np
import pandas as pd
import sklearn


def collab_testing(name=None):
    if name == "None":
        return "Collaborative filtering"
    else:
        print(np.__version__)
        print(pd.__version__)
        print(sklearn.__version__)
