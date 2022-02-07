
#######################
# Md Saifuzzaman
# McGill University
# Git@mdsaifuzzaman
#######################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import cross_val_predict
from matplotlib.offsetbox import AnchoredText
import seaborn as sns
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn import model_selection
from collections import OrderedDict

dataset = pd.read_csv (r'C:\All\Research\Code_hub\Optimization\all_variable_ld.csv')
dataset = dataset.set_index('Sample_ID')

X= dataset.drop(dataset.columns[16:], axis = 1)
X= X.drop(X.columns[:2], axis = 1)
#X=X.drop(['Slope','NDRE','PRP2'], axis = 1)# 'NDRE'

y = dataset.drop(dataset.columns[:16], axis = 1)
y =y.drop(dataset.columns[-4:0], axis = 1)
#y=dataset[['Zn']]
print(y.head())

y_list = list (y.columns)
X_list = list(X.columns)

#y = np.array(y)
#X = np.array(X)


ensemble_rgrs = []
rs = [1008, 1987,1669, 247,1805,919,2867,2092,2774]
for i in range(len(rs)):
    ensemble_rgrs.append((y_list[i],
            RandomForestRegressor(warm_start=True, max_features=None,
                                oob_score=True,
                                random_state=rs[i]))) 

        
        

# Map a Regressor name to a list of (<n_estimators>, <error rate>) pairs.
error_rate = OrderedDict((label,[]) for label, _ in ensemble_rgrs)

# Range of `n_estimators` values to explore.
min_estimators = 15
max_estimators = 1000
j=0
for label, rgr in ensemble_rgrs:
    z=y[[label]]
    X_train, X_test, y_train, y_test = train_test_split(X, z, test_size = .30, random_state = rs[j])
    for i in range(min_estimators, max_estimators + 1):
        rgr.set_params(n_estimators=i)
        rgr.fit(X_train, y_train)
        
        # Record the OOB error for each `n_estimators=i` setting.
        oob_error = 1 - rgr.oob_score_
        #error_rate[label].append((i, rgr.oob_score_))
        error_rate[label].append((i, oob_error))

# Generate the "OOB error rate" vs. "n_estimators" plot.
for label, rgr_err in error_rate.items():
    xs, ys = zip(*rgr_err)
    plt.plot(xs, ys, label=label)
    print(label,min(rgr_err, key = lambda t: t[1]))

plt.xlim(min_estimators, max_estimators)
plt.xlabel("n_estimators")
plt.ylabel("OOB error rate")
plt.legend(loc="upper right")
plt.show()