# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 14:53:57 2018

@author: Admin
"""

import pandas as pd
import matplotlib.pyplot as plt
import random

dataset= pd.read_csv("D:\\Udemy_ML\\UCB\\Ads_CTR_Optimisation.csv")
N= 10000
d=10
numbers_of_rewards_1 = [0] * d
numbers_of_rewards_0 = [0] * d
ads_selected= []
total_reward = 0

#Thompson Sampling
for n in range(0, N):
    max_random = 0
    ad = 0
    for i in range(0, d):
        random_beta = random.betavariate(numbers_of_rewards_1[i]+1, numbers_of_rewards_0[i]+1)
       
        if max_random < random_beta:
            max_random = random_beta
            ad = i
    ads_selected.append(ad)
    
    reward = dataset.values[n, ad]
    if reward==1:
        numbers_of_rewards_1[ad] = numbers_of_rewards_1[ad] + 1
    else:
        numbers_of_rewards_0[ad] = numbers_of_rewards_0[ad] + 1
    total_reward = total_reward + reward
    
plt.hist(ads_selected)
plt.title('Ads Selections')
plt.xlabel('Ad number')
plt.ylabel('Num selections')
plt.show()