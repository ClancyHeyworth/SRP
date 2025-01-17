import itertools
import random
import numpy as np
import gurobipy as gp
from tqdm import tqdm
from util import pertinent_filter, get_nodes, Node
from collections import defaultdict
    
def run_node_model(num_fields : int, num_days : int, cap : float):
    """
    Sets and Data
    """
    
    D = range(num_days)
    P = range(num_fields)
    random.seed(1)
    Probs = [random.random() for d in D]
    
    ripe_days = []
    for p in P:
        ripe_days.append(random.choice(D))
        
    N, children, children_cdfs = get_nodes(num_days, Probs)
    root_node = N[0]
    end_nodes = [n for n in N if n.stage == D[-1]]
    
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

    Z = {
        (p, n) : m.addVar(vtype=gp.GRB.BINARY)
        for p, n in X
    }
    
    """
    Objective Function
    """

    m.setObjective(
        gp.quicksum(n.prob * abs(n.stage - ripe_days[p]) * Z[p, n] for p, n in X)
        +\
        gp.quicksum(n.prob * Y[p, n] * abs(n.stage + 1 - ripe_days[p]) for p in P for n in end_nodes)
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
    
    Two = {
        p : m.addConstr(
            Y[p, root_node] == 1
        )
        for p in P
    }

    ThreeA = {
        (p, n) : m.addConstr(
            Y[p, n] == Y[p, n.parent] - X[p, n]
        )
        for p in P for n in N if n.stage >= 0 and not n.rain
    }

    ThreeB = {
        (p, n) : m.addConstr(
            Y[p, n] == Y[p, n.parent]
        )
        for p in P for n in N if n.stage >= 0 and n.rain
    }

    SetZ = {
        (p, n) : m.addConstr(
            Z[p, n] >= X[p, n]
        )
        for p, n in X
    }

    m.optimize()

    print('obj:', m.ObjVal)

    print('Expected proportion of fields not harvested:', np.mean([n.prob * Y[p, n].X for n in end_nodes for p in P]) / np.mean([n.prob for n in end_nodes]))

def run_paper_model(num_fields : int, num_days : int, alpha : float, cap : float):
    """
    Reconstruction of mixed-integer model described in paper
    """

    """
    Sets
    """
    # We will ignore farmer index for now

    P = range(num_fields)

    J = range(num_days)

    # All possible scenarios of availability
    W = list(itertools.product([0, 1], repeat=num_days))
    # W = np.array(np.meshgrid(*[[0, 1]] * num_days)).T.reshape(-1, num_days)
    
    Epsilon = {(j, tuple(w)) : w[j] for j in J for w in W} # very lazy

    """
    Data
    """
    
    # I assumed from interpreting the paper that each day had a unique
    # probability of rain, and the probabilities of the sequences
    # of days were thus the product of each step probability
    random.seed(1)

    day_probs = [random.random() for d in J]

    w_probs = {
        tuple(w) : np.prod([1 - day_probs[j] if w[j] == 1 else day_probs[j] for j in J])
        for w in W
    }

    from util import pertinent_filter2
    # W = pertinent_filter2(W, num_days, np.asarray(day_probs, dtype=np.float64), alpha)
    W = pertinent_filter(W, num_days, w_probs, alpha)

    ripe_days = []
    for p in P:
        ripe_days.append(random.choice(J))
    d = ripe_days

    # crop degredation indicators before and after harvest
    r_a = 1
    r_b = 1

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
    
    Test = {
        p : m.addConstr(
            gp.quicksum(X[p, j] for j in J) <= 1
        )
        for p in P
    }

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
        gp.quicksum(Lambda[w] for w in W) >= 1
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

    Eight = {
        (p, j) : m.addConstr(
            X[p, j] <= 1
        )
        for p in P for j in J
    }

    m.optimize()
        
    total_harvested = sum([X[p, j].X for p in P for j in J])
    print('Proportion of fields not harvested:', (num_fields - total_harvested)/num_fields)
    
    for j in J:
        break
        # print(j, sum([X[p, j].X for p in P]))
        print(f'Day {j}: {[(p, round(X[p, j].X, 2)) for p in P if round(X[p, j].X, 2) > 0]} Ripe: {[p for p in P if ripe_days[p] == j]} Rain Prob : {day_probs[j]}')
    return m.ObjVal

if __name__ == "__main__":
    import time
    
    t1 = time.time()
    run_paper_model(20, 12, 0.3, 2.0)
    t2 = time.time()
    
    print('Time taken:', t2-t1)
    