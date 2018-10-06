dataset = read.csv("Ads_CTR_Optimisation.csv")

#Thompson sampling
d = 10
N = 10000
ads_selected = integer(0)
number_of_rewards_1 =  integer(d)
number_of_rewards_0 = integer(d)
total_reward = 0
for (n in 1:N){
  maxRandom = 0
  ad = 1
  for (i in 1:d){
   
      random_beta = rbeta(n=1, shape1= number_of_rewards_1[i]+1, shape2= number_of_rewards_0[i]+1)
   
    
    if(random_beta > maxRandom){
      maxRandom = random_beta
      ad = i
    }
  }
  ads_selected = append(ads_selected, ad)
  reward= dataset[n, ad]
  if(reward==1){
  number_of_rewards_1[ad] = number_of_rewards_1[ad] + 1
  }
  else {
    number_of_rewards_0[ad] = number_of_rewards_0[ad] + 1
  }
  total_reward = total_reward + reward
  
}


hist(ads_selected, col = 'blue', main = 'Ads Selections', xlab= 'Ad', ylab = 'Num selections')