"""
Main file for executing optimisations.
"""
from collections import defaultdict
import gurobipy as gp
import numpy as np
import random

random.seed(1)

"""
Sets and Data
"""

num_fields = 20
num_days = 15
cap = 2.0 #Harvest capacity per day of the farmer
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

"""
Variables
"""

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

m.setObjective(
    gp.quicksum(n.prob * abs(n.stage - ripe_days[p]) * X[p, n] for p, n in X)
    +\
    gp.quicksum(n.prob * Y[p, n] * num_days for p in P for n in end_nodes)
    # sum([n.prob for n in N]) *\
)

"""
Constraints
"""

# Harvest capacity

One = {
    n : m.addConstr(
        gp.quicksum(X[p, n] for p in P) <= cap
    )
    for n in N if not n.rain
}

# Stage to stage constraints

TwoA = {
    (p, n) : m.addConstr(
        Y[p, n] == 1 - X[p, n]
    )
    for p in P for n in N if n.stage == 1 and not n.rain
}

TwoB = {
    (p, n) : m.addConstr(
        Y[p, n] == 1
    )
    for p in P for n in N if n.stage == 1 and n.rain
}

ThreeA = {
    (p, n) : m.addConstr(
        Y[p, n] == Y[p, n.parent] - X[p, n]
    )
    for p in P for n in N if n.stage > 1 and not n.rain
}

ThreeB = {
    (p, n) : m.addConstr(
        Y[p, n] == Y[p, n.parent]
    )
    for p in P for n in N if n.stage > 1 and n.rain
}

m.optimize()

print('obj:', m.ObjVal)

print('Average remaining field proportions across end nodes:', np.mean([Y[p, n].X for n in end_nodes for p in P]))


        