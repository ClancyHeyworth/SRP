from collections import defaultdict
import gurobipy as gp
import random

random.seed(1)

num_fields = 20
# num_days = 15
num_days = 15
# cap = 2.0 #Harvest capacity per day of the farmer
cap = 2.0
D = range(num_days)
P = range(num_fields)
Probs = [random.random() for d in D]

ripe_days = []
for p in P:
    ripe_days.append(random.choice(D))


class Node:
    def __init__(self, rain, parent, stage, prob = 1.0):
        self.rain = rain
        self.parent = parent
        self.stage = stage
        self.prob = prob
    
    def __repr__(self):
        return f"Day {self.stage}, Rain {self.rain}"

N = []
children = defaultdict(list)
children_cdfs = defaultdict(list)

end_nodes = []

def extend_tree(node:Node):
    """
    Function that builds the tree of rain events
    root_node
     -rain_node
       -rain_node
       -no_rain_node
        ...
     -no_rain_node
       -rain_node
       -no_rain_node
        ...
    """
    N.append(node)

    if node.stage < num_days-1:
        rain_node = Node(True, node, node.stage+1)
        children[node].append(rain_node)
        children_cdfs[node].append(Probs[node.stage+1])
        rain_node.prob = Probs[node.stage+1] * node.prob
        extend_tree(rain_node)

        no_rain_node = Node(False, node, node.stage+1)
        children[node].append(no_rain_node)
        children_cdfs[node].append(1.0) 
        no_rain_node.prob = node.prob * (1 - Probs[node.stage+1])
        extend_tree(no_rain_node)
    else:
        end_nodes.append(node)

root_node = Node(None, None, -1)
extend_tree(root_node)

# print(N[0])
# quit()

def sample():
    """
    A function to sample a scenario from the tree.
    """
    sample = []
    node = N[0]
    while node.stage < num_days - 1:
        # Get the child node according to the probability of each outcome (rain vs. no rain)
        sample_number = random.random() # random number between 0 and 1 used to sample the next day's weather
        child_index = next(idx for idx in range(len(children[node])) if children_cdfs[node][idx] >= sample_number)
        node = children[node][child_index]
        sample.append(node)
    return sample

# Model
m = gp.Model()

X = { # Proportion of parcel p gathered on node n
    (p, n) : m.addVar()
    for p in P for n in N
    if not n.rain
}

Y = { # Proportion of parcel p left to harvest at end of node n
    (p, n) : m.addVar(lb=0)
    for p in P for n in N
}

"""
Objective Function
"""

NotHarvestedPenalty = num_days

m.setObjective(
    gp.quicksum(n.prob * abs(n.stage - ripe_days[p]) * X[p, n] for p, n in X)
    +\
    gp.quicksum(n.prob * Y[p, n] * NotHarvestedPenalty for p in P for n in end_nodes)
    # sum([n.prob for n in N]) *\
)

# Harvest capacity

HarvestCapacity = {
    n : m.addConstr(
        gp.quicksum(X[p, n] for p in P) <= cap
    )
    for n in N if not n.rain
}

# Connect Y

InitialHarvesting1 = {
    (p, n) : m.addConstr(
        Y[p, n] == 1 - X[p, n]
    )
    for p in P for n in N if n.stage == 1 and not n.rain
}

InitialHarvesting2 = {
    (p, n) : m.addConstr(
        Y[p, n] == 1
    )
    for p in P for n in N if n.stage == 1 and n.rain
}

InitialHarvesting3 = {
    (p, n) : m.addConstr(
        Y[p, n] == Y[p, n.parent] - X[p, n]
    )
    for p in P for n in N if n.stage > 1 and not n.rain
}

InitialHarvesting4 = {
    (p, n) : m.addConstr(
        Y[p, n] == Y[p, n.parent]
    )
    for p in P for n in N if n.stage > 1 and n.rain
}

# Force Zs

# Non-anticapivity

m.optimize()

print('obj:', m.ObjVal)

import numpy as np
print(np.mean([X[0, n].X for n in N if n.stage == ripe_days[0] and (0, n) in X]))

print(sum([n.prob for n in N if n.stage == 3]))

k = sum([n.prob * abs(n.stage - ripe_days[p]) * X[p, n].X for p, n in X])
print(k)
# s = set()
# for x in X:
#     s.add(round(X[x].X, ndigits=3))
#     # print(X[x].X)
# print(s)
i = 0
for n in N:
    for p in P:
        if (p, n) not in X:
            continue
        if n.stage == num_days - 2:
            no_harvest_penalty = sum([c.prob * NotHarvestedPenalty for c in children[n]])
            bad_harvest_penalty = n.prob * abs(n.stage - ripe_days[p])
            # print(no_harvest_penalty, bad_harvest_penalty, abs(n.stage - ripe_days[p]))
            assert no_harvest_penalty > bad_harvest_penalty
            i += 1
    # if i > 10:
    #     break
        