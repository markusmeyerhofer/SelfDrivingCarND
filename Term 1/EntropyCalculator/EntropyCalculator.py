import math
#import scipy.stats
#print(scipy.stats.entropy(nodes,base=2))

def entropy(nodes):
    sum = 0.0
    for node in nodes:
        sum -= node * math.log(node, 2)
    return sum

nodes = [.5, .5]

childrenEntropy = entropy(nodes)

print("Entropy Children", childrenEntropy)

infoGain = 1 - 1/2*childrenEntropy-1/2*childrenEntropy

print("Information Gain", infoGain)

