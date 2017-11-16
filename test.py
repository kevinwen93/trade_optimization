import numpy as np
# Data format:
# Open, Close, Low, High, Volume
data = np.genfromtxt('data/sorted_data.csv.1000', delimiter=",")[1:,2:]

prices = data[:,1]

diffsum = 0.0
for i in xrange(1, len(prices)):
  diff = prices[i] - prices[i - 1] 
  diffsum += diff
print diffsum

