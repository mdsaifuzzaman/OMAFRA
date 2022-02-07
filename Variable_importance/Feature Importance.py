
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


dataset = pd.read_csv (r'C:\All\Research\Code_hub\Variable_importance\all_variable_ld.csv')
dataset = dataset.set_index('Sample_ID')


X= dataset.drop(dataset.columns[16:], axis = 1)
X= X.drop(X.columns[:2], axis = 1)

y = dataset.drop(dataset.columns[:16], axis = 1)
y= dataset.drop(dataset.columns[-4:], axis = 1)

print(y.head())

y_list = list (y.columns)
X_list = list(X.columns)

X = np.array(X)


for element in y_list:
    y= dataset.loc[:,[element]]
    y = np.array(y)

    if element=='P':
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .20, train_size = .80, random_state = 122)
        regr_rf = RandomForestRegressor(n_estimators=140, random_state=122)
    elif element== 'Mn':
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .20, train_size = .80, random_state = 187)
        regr_rf = RandomForestRegressor(n_estimators=140, random_state=187)
    elif element=='SOM':
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .20, train_size = .80, random_state = 2982)
        regr_rf = RandomForestRegressor(n_estimators=140, random_state=2982)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .20, train_size = .80, random_state = 100)
        regr_rf = RandomForestRegressor(n_estimators=140, random_state=100)
    
    regr_rf.fit(X_train, y_train)

    #plot Feature Importance
    feat_importances = pd.Series(regr_rf.feature_importances_, index=X_list)
    feat_importances.nsmallest(17).plot(kind='barh')
    plt.title('Feature Importances ')
    plt.xlabel('Relative Importance')
    #plt.show()
    plt.savefig('C:\\All\\Research\\Code_hub\\Variable_importance\\'+element+'_feature_Imp.png',dpi=1000)
    plt.close()
