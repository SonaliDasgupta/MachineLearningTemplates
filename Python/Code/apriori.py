# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 18:38:37 2018

@author: Admin
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset= pd.read_csv('D:/Udemy_ML/Apriori/Market_Basket_Optimisation.csv', header=None )

transactions= []
for i in range(0,7501):
    transactions.append([str(dataset.values[i, j]) for j in range(0,20)])
    
#train apriori
from apyori import apriori
rules = apriori(transactions,min_support=0.003, min_confidence= 0.2, min_lift= 3, min_length=2 )

#visualize
results= list(rules)


