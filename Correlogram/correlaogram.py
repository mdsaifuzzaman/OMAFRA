
#######################
# Md Saifuzzaman
# McGill University
# Git@mdsaifuzzaman
#######################

import math
import pickle
import numpy as np
import scipy.signal
import pandas as pd
from tkinter import *
from tkinter import filedialog
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import warnings; warnings.filterwarnings(action='once')

df = pd.read_csv(r'C:\All\Research\Code_hub\Correlogram\all_variable_ld.csv')

dfSns = df.drop(df.columns[17:], axis = 1)
dfSns = dfSns.drop(df.columns[:3], axis = 1)
dfSoil = df.drop(df.columns[:17], axis = 1)
dfSoil = dfSoil.drop(dfSoil.columns[-4:], axis = 1)
correlation = df.corr()
print(correlation)
correlation = correlation.iloc[3:17,17:-4]
print(correlation)

    #correalaogram: 
# Plot
plt.figure(figsize=(6,8))
hmap = sns.heatmap(correlation, xticklabels=dfSoil.corr().columns, yticklabels=dfSns.corr().columns, 
                        cmap='RdYlGn', center=0,annot=True,square=True, annot_kws={"size": 10}, 
                        linewidths=.5, cbar_kws={"shrink": .94,"pad":.05},fmt='.2f')

# use matplotlib.colorbar.Colorbar object
cbar = hmap.collections[0].colorbar
# here set the labelsize by 8
cbar.ax.tick_params(labelsize=8)

plt.xticks(rotation=90,fontsize=9)
plt.yticks(rotation=0,fontsize=9)

b, t = plt.ylim() # discover the values for bottom and top
b += 0.5 # Add 0.5 to the bottom
t -= 0.5 # Subtract 0.5 from the top
plt.ylim(b, t) # update the ylim(bottom, top) values

plt.show()
#plt.savefig('C:\\Users\\MaHeE\\Desktop\\ML_Project\\Final\\Corlleo.png',dpi=1000)
plt.close()

    #correlation:

sns.set(style="ticks")
# Compute the correlation matrix
corr = dfSns.corr('pearson')
# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(8, 8))
# Generate a custom diverging colormap
#cmap = sns.diverging_palette(220, 20, as_cmap=True)
# Draw the heatmap with the mask and correct aspect ratio
hmap = sns.heatmap(corr,mask=mask, cmap='RdYlGn', center=0,annot=True,
            square=True, annot_kws={"size": 12}, linewidths=.5, cbar_kws={"shrink": .867,"pad":-0.01},fmt='.2f')

# use matplotlib.colorbar.Colorbar object
cbar = hmap.collections[0].colorbar
# here set the labelsize by 8
cbar.ax.tick_params(labelsize=8)

b, t = plt.ylim() # discover the values for bottom and top
b += 0.5 # Add 0.5 to the bottom
t -= 0.5 # Subtract 0.5 from the top
plt.ylim(b, t) # update the ylim(bottom, top) values

plt.xticks(rotation=90,fontsize=9.5)
plt.yticks(rotation=0,fontsize=10)

plt.show()
f.savefig('C:\\All\\Research\\Code_hub\\Correlogram\\corre.png',dpi=500)
plt.close()