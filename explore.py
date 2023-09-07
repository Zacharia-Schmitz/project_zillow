'''Explore Zillow data

Functions:
- pear
- plt_loc
- nova
- ap
- yp
- cp
'''

########## IMPORTS ##########
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest, RFE, f_regression, SequentialFeatureSelector
from scipy import stats


########## EXPLORE ##########

def pear(train, x, y, alt_hyp='two-sided'):
    '''Spearman's R test with a print'''
    r,p = stats.spearmanr(train[x], train[y], alternative=alt_hyp)
    print(f"Spearman's R: {x} and {y}\n", f'r = {r}, p = {p}')

def plt_loc(df):
    '''toss in train data and get a pretty plot on lat long and prop value per county'''
    # make xy label scale smaller
    p = df.copy()
    p = p.assign(lat=p.latitude/1000000)
    p = p.assign(long=p.longitude/1000000)
    p = p.sort_values('prop_value')
    # make the size
    plt.figure(figsize=[12,6])
    # generic legend example
    sns.scatterplot(data=p,y='lat',x='long',hue='prop_value',palette='Greys',alpha=1)
    # plot per county the lat long, hue on prop value
    sns.scatterplot(data=p[p.county=='LA'],y='lat',x='long',hue='prop_value',palette='Blues')
    sns.scatterplot(data=p[p.county=='Orange'],y='lat',x='long',hue='prop_value',palette='Reds')
    sns.scatterplot(data=p[p.county=='Ventura'],y='lat',x='long',hue='prop_value',palette='Greens')
    # plot county legend
    plt.text(y=34.05,x=-119.18,s='Ventura',fontsize=16,color='darkgreen')
    plt.text(y=33.9,x=-118.75,s='Los Angeles',fontsize=16,color='darkblue')
    plt.text(y=33.6,x=-118.15,s='Orange',fontsize=16,color='orangered')
    # label generic legend
    plt.legend(title='Prop Value',labels=['$400k','$800k','$1.2m','$1.6m','$2.0m'])
    # give it a name
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Property Value based on Location')
    plt.show()

def compare_means(train, target, quant_var, alt_hyp='two-sided'):
    """
    The function compares the means of two groups using the Mann-Whitney U test and returns the test
    statistic and p-value.
    
    :param train: a pandas DataFrame containing the training data
    :param target: The target variable is a binary variable that indicates the outcome of interest. In
    this function, it is used to split the data into two groups based on the value of the target
    variable (0 or 1)
    :param quant_var: The quantitative variable that we want to compare the means of between two groups
    :param alt_hyp: The alternative hypothesis for the Mann-Whitney U test. It specifies the direction
    of the test and can be either "two-sided" (default), "less" or "greater". "two-sided" means that the
    test is two-tailed, "less" means that the test is one, defaults to two-sided (optional)
    :return: the result of a Mann-Whitney U test comparing the means of two groups (x and y) based on a
    quantitative variable (quant_var) in a training dataset (train) with a binary target variable
    (target). The alternative hypothesis (alt_hyp) can be specified as either 'two-sided' (default),
    'less', or 'greater'.
    """
    x = train[train[quant_var]][target]
    y = train[target]
    # alt_hyp = ‘two-sided’, ‘less’, ‘greater’
    stat,p = stats.mannwhitneyu(x, y, use_continuity=True, alternative=alt_hyp)
    print("Mann-Whitney Test:\n", f'stat = {stat}, p = {p}')

def nova(s1,s2,s3):
    '''ANOVA test for 3 samples'''
    stat,p = stats.kruskal(s1,s2,s3)
    print("Kruskal-Wallis H-Test:\n", f'stat = {stat}, p = {p}')

def ap(train):
    # explore area and property value
    pear(train,'area','prop_value')
    sns.regplot(data=train,x='area',y='prop_value',marker='.',line_kws={'color':'orange'})
    plt.title('Area and Property Value Correlation')
    plt.xlabel("Area in Square Feet")
    plt.ylabel("Property Value ($ Millions)")
    plt.show()

def yp(train):
    # explore age and property value
    pear(train,'age','prop_value')
    sns.regplot(data=train,x='age',y='prop_value',marker='.',line_kws={'color':'orange'})
    plt.title('Age and Property Value Correlation')
    plt.xlabel("Age of Property")
    plt.ylabel("Property Value ($ Millions)")
    plt.show()

def cp(train):
    # explore room count and property value
    pear(train,'roomcnt','prop_value')
    sns.regplot(data=train[train.roomcnt>0],x='roomcnt',y='prop_value',marker='.',line_kws={'color':'orange'})
    plt.title('Room Count and Property Value Correlation')
    plt.xlabel("Room Count")
    plt.ylabel("Property Value ($ Millions)")
    plt.show()

def lp(train):
    # explore average property value and counties
    nova(train[train.county=='LA'].prop_value,train[train.county=='Orange'].prop_value,train[train.county=='Ventura'].prop_value)
    plt_loc(train)

def dist(train):
    plt.hist(train.prop_value,200)
    plt.title('Property Value Distribution')
    plt.ylabel('# of Properties')
    plt.xlabel('Property Value ($ Millions)')
    plt.show()