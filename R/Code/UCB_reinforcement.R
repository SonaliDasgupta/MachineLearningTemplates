dataset = read.csv("Ads_CTR_Optimisation.csv")

#UCB
d = 10
N = 10000
ads_selected = integer(0)
number_of_selections =  integer(d)
number_of_rewards = integer(d)
total_reward = 0
for (n in 1:N){
  maxUpperBound = 0
  ad = 1
  for (i in 1:d){
    if(number_of_selections[i]>0){
      avg_Reward = number_of_rewards[i]/number_of_selections[i]
      delta_i = sqrt(3/2 * log(n)/number_of_selections[i])
      upperBound = avg_Reward + delta_i
    }
    else{
      upperBound=1e400
    }
    
    if(upperBound > maxUpperBound){
      maxUpperBound = upperBound
      ad = i
    }
  }
    ads_selected = append(ads_selected, ad)
    number_of_rewards[ad] = number_of_rewards[ad] + dataset[n, ad]
    number_of_selections[ad] = number_of_selections[ad] + 1
    total_reward = total_reward + dataset[n,ad]
  
}


hist(ads_selected, col = 'blue', main = 'Ads Selections', xlab= 'Ad', ylab = 'Num selections')