#######################
# Md Saifuzzaman
# McGill University
# Git@mdsaifuzzaman
#######################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

xls = pd.ExcelFile(r'C:\All\Research\Code_hub\Box_plot_data\Sampling_sensor_subset.xlsx')
print(xls.sheet_names)
xfile = pd.read_excel(r'C:\All\Research\Code_hub\Box_plot_data\Sampling_sensor_subset.xlsx', sheet_name=None)
print(xfile.keys())
F25 = pd.read_excel(xls, 'F25').assign(Study_Sites=xls.sheet_names[0])
WH = pd.read_excel(xls, 'WH').assign(Study_Sites=xls.sheet_names[1])
KM = pd.read_excel(xls, 'KM').assign(Study_Sites=xls.sheet_names[2])
LP = pd.read_excel(xls, 'LP').assign(Study_Sites=xls.sheet_names[3])
LD = pd.read_excel(xls, 'LD').assign(Study_Sites=xls.sheet_names[4])
TE = pd.read_excel(xls, 'TE').assign(Study_Sites=xls.sheet_names[5])
SM = pd.read_excel(xls, 'SM').assign(Study_Sites=xls.sheet_names[6])
R50 = pd.read_excel(xls, 'R50').assign(Study_Sites=xls.sheet_names[7])
RB = pd.read_excel(xls, 'RB').assign(Study_Sites=xls.sheet_names[8])
RL = pd.read_excel(xls, 'RL').assign(Study_Sites=xls.sheet_names[9])
VN = pd.read_excel(xls, 'VN').assign(Study_Sites=xls.sheet_names[10])
NX = pd.read_excel(xls, 'NX').assign(Study_Sites=xls.sheet_names[11])
print(F25.head())

cdf = pd.concat([F25, WH, KM, LP, LD, TE, SM, R50, RB, RL, VN, NX]) 
mdf = pd.melt(cdf, id_vars=['Study_Sites'], var_name=['Letter'])
print(mdf.head())


for p in  ['pH',  'BpH',  'SOM',  'P (ppm)', 'K (ppm)',  'CEC (meq hg-1)']:
    fig, axes = plt.subplots()
    datam = cdf[[p,'Study_Sites']]
    mdf = pd.melt(datam, id_vars=['Study_Sites'], var_name=['Parameter'], value_name=p)
    bx = sns.boxplot(x="Study_Sites", y=p, data=mdf, showfliers = True, whis= 'range', boxprops = dict(linewidth=.5), whiskerprops = dict(linewidth=.5), medianprops = dict(linewidth=.5), capprops = dict(linewidth=.5)) 
    axes.set_xlabel( 'Study Sites', fontsize=12)
    axes.set_ylabel(p, fontsize=12)
    #axes.set_title("Box Plot", fontsize=12)
    #axes.xaxis.set_tick_params(length=1.5, width=0.5)
    #axes.yaxis.set_tick_params(length=1.5,width=0.5)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    axes.spines['bottom'].set_linewidth(0.5)
    axes.spines['left'].set_linewidth(0.5)
    axes.spines['top'].set_linewidth(0.5)
    axes.spines['right'].set_linewidth(0.5)
    for i in range(len(xls.sheet_names)):
        sigma=pd.read_excel(xls, xls.sheet_names[i])[[p]].std()
        #textstr= r'$\sigma=%.2f$' % (sigma, )
        #axes.text(i/12, 0.99,textstr , transform=axes.transAxes, fontsize=6,verticalalignment='top', multialignment= 'center',)
        #for line in bx['caps']:
        #    # get position data for caps line
        #    x, y = line.get_xydata()[1] # top of caps line
            # overlay std value
        #    axes.text(x, y, '$\sigma=%.2f$' % sigma,
        #                horizontalalignment='center', fontsize=8) # draw above, centered
    fig.savefig('C:\\All\\Research\\Code_hub\\Box_plot_data\\'+ p +'.png')
    #plt.show()
    plt.close(fig)
