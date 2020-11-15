"""
Contains content based recommendation systems

"""
import numpy as np
import pandas as pd
import sklearn


def content_testing(name=None):
    if name == "None":
        return "Content based filtering"
    else:
        print(np.__version__)
        print(pd.__version__)
        print(sklearn.__version__)
