# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 10:39:56 2018

@author: Admin
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

dataset= pd.read_csv("D:\\Udemy_ML\\UCB\\Ads_CTR_Optimisation.csv")
N= 10000
d=10
numbers_of_selections = [0] * d
sums_of_rewards = [0] * d
ads_selected= []
total_reward = 0
for n in range(0, N):
    max_upperbound = 0
    ad = 0
    for i in range(0, d):
        if numbers_of_selections[i] > 0:
            average_reward = sums_of_rewards[i]/numbers_of_selections[i]
            delta_i = math.sqrt(3/2 * math.log(n+1)/numbers_of_selections[i])
            upper_bound= average_reward + delta_i
        else:
            upper_bound= 1e400
        if upper_bound > max_upperbound:
            max_upperbound = upper_bound
            ad = i
    ads_selected.append(ad)
    numbers_of_selections[ad] = numbers_of_selections[ad] + 1
    reward = dataset.values[n, ad]
    sums_of_rewards[ad] = sums_of_rewards[ad] + reward
    total_reward = total_reward + reward
    
plt.hist(ads_selected)
plt.title('Ads Selections')
plt.xlabel('Ad number')
plt.ylabel('Num selections')
plt.show()

    
       
            
        
