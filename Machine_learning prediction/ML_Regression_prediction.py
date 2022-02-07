
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

from sklearn.tree import export_graphviz
import pydot

'''
rs=3140
#X=X.drop(['TWI','U238','K40','Slope','PRP1','Elevation',], axis = 1)# 'PRP1','NDVI'
'''
dataset = pd.read_csv (r'C:\\All\Research\\Code_hub\\Mechine_learning prediction\\all_variable_ld.csv')
dataset = dataset.set_index('Sample_ID')


X= dataset.drop(dataset.columns[16:], axis = 1)
X= X.drop(X.columns[:2], axis = 1)
X = X.drop(['TWI','U238','K40','Th232','PRP2','Elevation','PRP1','TC','Slope'], axis = 1)
y = dataset.drop(dataset.columns[:16], axis = 1)
y=dataset[['K']]

print(y.head())

y_list = list (y.columns)
X_list = list(X.columns)

y = np.array(y)
X = np.array(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 20, random_state = 247)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size = .50, random_state = 247)


regr_rf = RandomForestRegressor(n_estimators=511, 
                                random_state = 247,)
regr_rf.fit(X_train, y_train)
#fi = pd.DataFrame({'feature': X_list,
#                   'importance': regr_rf.feature_importances_}).\
#                    sort_values('importance', ascending = False)

# Display
#fi.head(17)
#plot Feature Importance
feat_importances = pd.Series(regr_rf.feature_importances_, index=X_list)
feat_importances.nsmallest(17).plot(kind='barh')
plt.title('Feature Importances ')
plt.xlabel('Relative Importance')
#plt.show()
#plt.savefig('C:\\Users\\MaHeE\\Desktop\\ML_Project\\code\\Feature Importance\\'+y_list[0]+'_feature_Imp.png',dpi=1000)
plt.savefig('C:\\All\\Research\\Code_hub\\Mechine_learning prediction\\'+y_list[0]+'_feature_Imp.png',dpi=1000)
plt.close()


# Predict on new data
y_rf = regr_rf.predict(X_test)
y_rv = regr_rf.predict(X_val)



print(feat_importances.nsmallest(17))
print(pd.DataFrame(y_test, columns=y_list).describe().T)
print(pd.DataFrame(np.round(y_rf,2), columns = y_list).describe().T)
print('Test Mean Absolute Error: %.2f' % metrics.mean_absolute_error(y_test, y_rf, multioutput= 'raw_values'))
print('Test Mean Squared Error: %.2f' %  metrics.mean_squared_error(y_test, y_rf, multioutput = 'raw_values'))
print('Test Root Mean Squared Error: %.2f' % np.sqrt(metrics.mean_squared_error(y_test, y_rf, multioutput = 'raw_values')))
print("Test Score(R^2): %.2f" % regr_rf.score(X_test, y_test))
print('Validation R^2 = %.2f' % metrics.r2_score(y_val, y_rv, multioutput = 'raw_values'))
#print(y_test,y_rf )

pd.DataFrame(y_test, columns = y_list).to_csv ('C:\\All\\Research\\Code_hub\\Mechine_learning prediction\\'+y_list[0]+'_test.csv', index = None, header=True)
pd.DataFrame(np.round(y_rf,2), columns = y_list).to_csv ('C:\\All\\Research\\Code_hub\\Mechine_learning prediction\\'+y_list[0]+'_pred.csv', index = None, header=True)



# Pull out one tree from the forest
tree = regr_rf.estimators_[5]
# Export the image to a dot file
export_graphviz(tree, out_file = 'C:\\All\\Research\\Code_hub\\Mechine_learning prediction\\' +y_list[0]+'_tree.dot', feature_names = X_list, rounded = True, precision = 1)
# Use dot file to create a grap
(graph, ) = pydot.graph_from_dot_file('C:\\All\\Research\\Code_hub\\Mechine_learning prediction\\' +y_list[0]+'_tree.dot')
# Write graph to a png file
graph.write_png('C:\\All\\Research\\Code_hub\\Mechine_learning prediction\\' +y_list[0]+'_tree.png')

# Limit depth of tree to 3 levels
regr_rf_small = RandomForestRegressor(n_estimators=10, max_depth = 3)
regr_rf_small.fit(X_train, y_train)
# Extract the small tree
tree_small = regr_rf_small.estimators_[5]
# Save the tree as a png image
export_graphviz(tree_small, out_file = 'C:\\All\\Research\\Code_hub\\Mechine_learning prediction\\' +y_list[0]+'_small_tree.dot', feature_names = X_list, rounded = True, precision = 1)
(graph, ) = pydot.graph_from_dot_file('C:\\All\\Research\\Code_hub\\Mechine_learning prediction\\' +y_list[0]+'_small_tree.dot')
graph.write_png('C:\\All\\Research\\Code_hub\\Mechine_learning prediction\\' +y_list[0]+'_small_tree.png')
pd.DataFrame(np.round(y_rf,2), columns = y_list).describe().to_csv ('C:\\All\\Research\\Code_hub\\Mechine_learning prediction\\' +y_list[0]+'_pred_statistics.csv', index = True, header=True)

for i in range(len(y_list)):
    y_test_single = pd.DataFrame(np.round(y_test,2), columns = y_list).iloc[:,[i]].as_matrix()
    y_rf_single = pd.DataFrame(np.round(y_rf,2), columns = y_list).iloc[:,[i]].as_matrix()
    fig, ax = plt.subplots()
    ax.scatter(y_test_single, y_rf_single, edgecolors=(0, 0, 0))
    ax.plot([y_test_single.min(), y_test_single.max()], [y_test_single.min(), y_test_single.max()], 'k--', lw=1)
    ax.add_artist(AnchoredText('RMSE = %.2f \n$R^2$ = %.2f' 
                                % ((np.sqrt(metrics.mean_squared_error(y_test_single, y_rf_single))),
                                (metrics.r2_score(y_test_single, y_rf_single))), prop=dict(size=12), 
                                loc = 2, frameon= False))  
    ax.set_xlabel('Measured '+ y_list[i], fontsize=12)
    ax.set_ylabel('Predicted '+ y_list[i], fontsize=12)
    ax.set_title("Measured vs Predicted", fontsize=12)
    ax.xaxis.set_tick_params(length=1.5, width=0.5)
    ax.yaxis.set_tick_params(length=1.5,width=0.5)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['top'].set_linewidth(0.5)
    ax.spines['right'].set_linewidth(0.5)
    fig.savefig('C:\\All\\Research\\Code_hub\\Mechine_learning prediction\\'+y_list[i]+'.png')
    #plt.show()
    plt.close(fig)