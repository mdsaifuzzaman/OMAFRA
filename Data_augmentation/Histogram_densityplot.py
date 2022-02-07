
#######################
# Md Saifuzzaman
# McGill University
# Git@mdsaifuzzaman
#######################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from matplotlib.offsetbox import AnchoredText
import seaborn as sns
from sklearn.metrics import mean_squared_error
from math import sqrt
from matplotlib.ticker import FormatStrFormatter


dataset = pd.read_csv (r'C:\All\Research\Code_hub\Data_augmentation\all_variable_ld.csv')
dataset = dataset.set_index('Sample_ID')




X= dataset.drop(dataset.columns[16:], axis = 1)
X= X.drop(X.columns[:2], axis = 1)
y = dataset.drop(dataset.columns[:16], axis = 1)
y=y.iloc[:,:-4]
#y=y.iloc[:,5:6]
print(y.head())
y_list = list (y.columns)
for element in y_list:
    #plt.figure( dpi= 80)
    fig, ax = plt.subplots(figsize=(2,2))#figsize=(16,8)
    ax=sns.distplot(dataset[element], hist=True, kde=True, 
                color = 'darkblue', 
                hist_kws={},
                kde_kws={'shade': True,'linewidth': 0.3})
    
    ax.set_xlabel(element,fontsize=6)
    ax.set_ylabel('Density',labelpad=1.2,fontsize=6)
    ax.add_artist(AnchoredText('σ = %.2f \nμ = %.2f' % (dataset[element].std(), dataset[element].mean()), prop=dict(size=6), loc = 1, frameon= False))  
    ax.xaxis.set_tick_params(length=1.5, width=0.2)
    ax.yaxis.set_tick_params(length=1.5,width=0.2)
    #ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    if element=='Ca' or element== 'K' or element=='Mg' or element=='Mn':
        plt.yticks(ax.get_yticks(), np.round(ax.get_yticks() * 100,2),fontsize=6)
        
    else:
        plt.yticks(fontsize=6)
        
    #ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    xmin, xmax = ax.get_xlim()
    if element== 'Zn':
        ax.set_xticks(np.round(np.linspace(1, 21, 5), 0))
        ax.set_yticks((np.round(np.linspace(0.0, 0.2, 5), 2)))
    elif element== 'CEC':
        ax.set_yticks((np.round(np.linspace(0.0, 0.4, 5), 1)))
        ax.set_xticks(np.round(np.linspace(7, 22, 4), 0))
        #plt.yticks()
    elif element== 'P':
        ax.set_yticks((np.round(np.linspace(0.0, 0.05, 6), 2)))
        ax.set_xticks(np.round(np.linspace(2, 92, 4), 0))
    elif element== 'Ca':
        ax.set_xticks(np.round(np.linspace(900,3000, 4), 0))
    elif element== 'K':
        ax.set_xticks(np.round(np.linspace(20,440, 4), 0)) 
    elif element== 'Mg':
        ax.set_xticks(np.round(np.linspace(15,495, 4), 0)) 
    elif element== 'Mn':
        ax.set_xticks(np.round(np.linspace(10,30, 5), 0)) 
    elif element== 'pH':
        ax.set_xticks(np.round(np.linspace(6.5,8, 4), 1))
    else:
        ax.set_xticks(np.round(np.linspace(abs(xmin), xmax, 4), 1))
    plt.xticks(fontsize=6)
    
    ax.spines['bottom'].set_linewidth(0.3)
    ax.spines['left'].set_linewidth(0.3)
    ax.spines['top'].set_linewidth(0.3)
    ax.spines['right'].set_linewidth(0.3)
    #plt.box(False)
    plt.tight_layout()
    plt.subplots_adjust(left=0.184,right=0.91)
    #plt.show()
    fig.savefig('C:\\All\\Research\\Code_hub\\Data_augmentation\\'+element+'_density.png',dpi=500)
    plt.close(fig)
