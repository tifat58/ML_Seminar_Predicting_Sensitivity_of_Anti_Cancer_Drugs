import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import pyplot
import scipy

import pandas as pd
import numpy as np

# pandas dataframe, as.matrix() makes a numpy array
expr = pd.read_excel('../Downloads/expression_sn38_small.xlsx')
expr_t = expr.T

#print(expr_t[1:])

ex = expr_t[1:]


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.model_selection import train_test_split
x = ex.iloc[:, 0:369]
y = ex.iloc[:, 368:369]

selector = SelectKBest(f_classif, k = 50)

xbest = selector.fit_transform(x, y)

print(x.shape[1])
print(xbest.shape[1])

print(xbest)


print("worked")