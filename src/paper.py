"""
Reconstruction of mixed-integer model described in paper
"""

import itertools
import random
import numpy as np
import gurobipy as gp
from tqdm import tqdm

"""
Sets
"""
# We will ignore farmer index for now

num_fields = 15
P = range(num_fields)

num_days = 10
J = range(num_days)

# All possible scenarios of availability
W = list(itertools.product([0, 1], repeat=num_days))
Epsilon = {(j, w) : w[j] for j in J for w in W} # very lazy


"""
Data
"""

# I assumed from interpreting the paper that each day had a unique
# probability of availability, and the probabilities of the sequences
# of days were thus the product of each step probability
random.seed(1)

day_probs = [random.random() for d in J]
w_probs = {
    w : np.prod([day_probs[j] if w[j] == 1 else 1 - day_probs[j] for j in J])
    for w in W
}

ripe_days = []
for p in P:
    ripe_days.append(random.choice(J))
d = ripe_days

cap = 2

# crop degredation indicators before and after harvest
r_a = 1
r_b = 1

alpha = 0.3

"""
Variables
"""

m = gp.Model()

X = { # Rate of parcel p gathered on jth day
    (p, j) : m.addVar(ub=1)
    for p in P for j in J
}

Z1 = { # 1 if parcel of farmer p gathered before its ripeness day
    (p, j) : m.addVar(vtype=gp.GRB.BINARY)
    for p in P for j in J
}

Z2 = { # 1 if parcel of farmer p gathered after its ripeness day
    (p, j) : m.addVar(vtype=gp.GRB.BINARY)
    for p in P for j in J
}

Lambda = { # 1 if chance constraints satisfied
    w : m.addVar(vtype=gp.GRB.BINARY)
    for w in W
}

"""
Objective
"""

# m.setObjective(
#     gp.quicksum(
#         r_a * Z2[p, j] * (j - d[p]) - r_b * Z1[p, j] * (j - d[p])
#         for p in P for j in J
#     ) 
#     +\
#     # secondary objective to minimize non-harvested fields
#     num_days * gp.quicksum(1 - gp.quicksum(X[p, j] for j in J) for p in P)
# )

m.setObjective(
    gp.quicksum(
        r_a * Z2[p, j] * (j - d[p]) - r_b * Z1[p, j] * (j - d[p])
        for p in P for j in J
    ) 
    +\
    # secondary objective to minimize non-harvested fields
    gp.quicksum(abs(J[-1] + 1 - ripe_days[p]) * (1 - gp.quicksum(X[p, j] for j in J)) for p in P)
)


"""
Constraints
"""

# Ignore this constraint
# Two = {
#     p : m.addConstr(
#         gp.quicksum(X[p, j] for j in J) == 1
#     )
#     for p in P
# }

# Ignored q data value
Three = {
    j : m.addConstr(
        gp.quicksum(X[p, j] for p in P) <= cap 
    )
    for j in J
}
            
Four1 = {
    (p, j, w) : m.addConstr(
        (Z1[p, j] + Z2[p, j]) - (Epsilon[j, w] - Lambda[w])
        <= 1
    )
    for p in tqdm(P) for j in J for w in W
}

Four2 = m.addConstr(
    gp.quicksum(w_probs[w] * Lambda[w] for w in W) >= 1 - alpha
)

Five = {
    (p, j) : m.addConstr(
        Z1[p, j] * (j - d[p]) <= 0
    )
    for p in P for j in J
}

Six = {
    (p, j) : m.addConstr(
        Z2[p, j] * (j - d[p]) >= 0
    )
    for p in P for j in J
}

Seven = {
    (p, j) : m.addConstr(
        Z1[p, j] + Z2[p, j] >= X[p, j]
    )
    for p in P for j in J
}

# Eight = {
#     (p, j) : m.addConstr(
#         X[p, j] <= 1
#     )
#     for p in P for j in J
# }

m.optimize()
    
total_harvested = sum([X[p, j].X for p in P for j in J])
print('Proportion of fields not harvested:', (num_fields - total_harvested)/num_fields)