import itertools
import random
import numpy as np
import gurobipy as gp
from tqdm import tqdm
from util import pertinence_filter, get_nodes, Node, ModelParams
from collections import defaultdict
from tabulate import tabulate
    
def run_node_model(params : ModelParams):
    """
    Sets and Data
    """
    
    D = range(params.num_days)
    P = range(params.num_fields)
    random.seed(1)
    Probs = [random.random() for d in D]
    
    ### delete
    # np.random.seed(1)
    # day_probs = np.random.normal(0.1, 0.05, len(D))
    # day_probs = [abs(d) for d in day_probs]
    # Probs = day_probs
    ### delete
    
    ripe_days = []
    for p in P:
        ripe_days.append(random.choice(D))
        
    N, children, children_cdfs = get_nodes(params.num_days, Probs)
    root_node = N[0]
    end_nodes = [n for n in N if n.stage == D[-1]]
    
    cap = params.cap
    
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
    
    if not params.verbose:
        m.setParam('OutputFlag', 0)
        
    m.optimize()

    if params.verbose:
        print('obj:', m.ObjVal)

        print('Expected proportion of fields not harvested:', np.mean([n.prob * Y[p, n].X for n in end_nodes for p in P]) / np.mean([n.prob for n in end_nodes]))
    return m.ObjVal

def run_original_paper_model(params : ModelParams):
    """
    Reconstruction of mixed-integer model described in paper,
    without any scenario reduction
    """

    """
    Sets
    """
    # We will ignore farmer index for now

    P = range(params.num_fields)

    J = range(params.num_days)

    # All possible scenarios of availability
    W = list(itertools.product([0, 1], repeat=len(J)))
    
    Epsilon = {(j, tuple(w)) : w[j] for j in J for w in W} # very lazy

    """
    Data
    """
    
    # I assumed from interpreting the paper that each day had a unique
    # probability of rain, and the probabilities of the sequences
    # of days were thus the product of each step probability
    random.seed(1)

    day_probs = [random.random() for d in J]
    ### delete
    # np.random.seed(1)
    # day_probs = np.random.normal(0.1, 0.05, len(J))
    # day_probs = [abs(d) for d in day_probs]
    ### delete

    w_probs = {
        tuple(w) : np.prod([1 - day_probs[j] if w[j] == 1 else day_probs[j] for j in J])
        for w in W
    }

    # W = pertinent_filter2(W, num_days, np.asarray(day_probs, dtype=np.float64), alpha)
    alpha = params.alpha
    cap = params.cap

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
        for p in P for j in J for w in W
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

    Eight = {
        (p, j) : m.addConstr(
            X[p, j] <= 1
        )
        for p in P for j in J
    }
    
    if not params.verbose:
        m.setParam('OutputFlag', 0)

    m.optimize()
    
    if params.verbose:
        total_harvested = sum([X[p, j].X for p in P for j in J])
        print('Proportion of fields not harvested:', (len(P) - total_harvested)/len(P))
    
        print('Day Table:')
        table = []
        headers = ['Day', 'Rain Prob', 'Field, Harvest Prop', 'Ripe Fields']
        for j in J:
            # break
            # print(j, sum([X[p, j].X for p in P]))
            fields = ' '.join([str((p, round(X[p, j].X, 2))) for p in P if round(X[p, j].X, 2) > 0])
            ripe = ' '.join([str(p) for p in P if ripe_days[p] == j])
            table.append(
                [j, day_probs[j], fields, ripe]
            )
            # print(f'Day {j}: {[(p, round(X[p, j].X, 2)) for p in P if round(X[p, j].X, 2) > 0]} Ripe: {[p for p in P if ripe_days[p] == j]} Rain Prob : {day_probs[j]}')
        print(tabulate(table, headers=headers))
        print('Field Table:')
        table = []
        headers = ['', 'Field', 'Ripe Day', 'Harvest Day, Amount', 'Harvest Penalty', 'End Penalty']
        a_total = 0
        b_total = 0
        for p in P:
            a = sum([
                r_a * Z2[p, j].X * (j - d[p]) - r_b * Z1[p, j].X * (j - d[p]) for j in J
            ]) 
            
            b = sum([abs(J[-1] + 1 - ripe_days[p]) * (1 - sum([X[p, j].X for j in J]))])
            a_total += a
            b_total += b
            
            days_harvested = ' '.join([str((j, X[p, j].X)) for j in J if round(X[p, j].X, 2) > 0])
            table.append(['', p, ripe_days[p], days_harvested, round(a), round(b)])
        table.append(['Totals', '', '', '', a_total, b_total])
        print(tabulate(table, headers=headers))
    return m.ObjVal

def run_paper_model(params : ModelParams):
    """
    Reconstruction of mixed-integer model described in paper,
    uses (1 - alpha) scenario pertinence algorithm described in
    section 5 to improve performance time
    """

    """
    Sets
    """
    # We will ignore farmer index for now

    P = range(params.num_fields)

    J = range(params.num_days)

    # All possible scenarios of availability
    W = list(itertools.product([0, 1], repeat=len(J)))
    
    Epsilon = {(j, tuple(w)) : w[j] for j in J for w in W} # very lazy

    """
    Data
    """
    
    # I assumed from interpreting the paper that each day had a unique
    # probability of rain, and the probabilities of the sequences
    # of days were thus the product of each step probability
    random.seed(1)

    day_probs = [random.random() for d in J]
    ### delete
    # np.random.seed(1)
    # day_probs = np.random.normal(0.1, 0.05, len(J))
    # day_probs = [abs(d) for d in day_probs]
    ### delete

    w_probs = {
        tuple(w) : np.prod([1 - day_probs[j] if w[j] == 1 else day_probs[j] for j in J])
        for w in W
    }

    # W = pertinent_filter2(W, num_days, np.asarray(day_probs, dtype=np.float64), alpha)
    alpha = params.alpha
    cap = params.cap
    
    W = pertinence_filter(W, len(J), w_probs, alpha)

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
        for p in P for j in J for w in W
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
    
    if not params.verbose:
        m.setParam('OutputFlag', 0)

    m.optimize()
    
    if params.verbose:
        total_harvested = sum([X[p, j].X for p in P for j in J])
        print('Proportion of fields not harvested:', (len(P) - total_harvested)/len(P))
    
        print('Day Table:')
        table = []
        headers = ['Day', 'Rain Prob', 'Field, Harvest Prop', 'Ripe Fields']
        for j in J:
            # break
            # print(j, sum([X[p, j].X for p in P]))
            fields = ' '.join([str((p, round(X[p, j].X, 2))) for p in P if round(X[p, j].X, 2) > 0])
            ripe = ' '.join([str(p) for p in P if ripe_days[p] == j])
            table.append(
                [j, day_probs[j], fields, ripe]
            )
            # print(f'Day {j}: {[(p, round(X[p, j].X, 2)) for p in P if round(X[p, j].X, 2) > 0]} Ripe: {[p for p in P if ripe_days[p] == j]} Rain Prob : {day_probs[j]}')
        print(tabulate(table, headers=headers))
        print('Field Table:')
        table = []
        headers = ['', 'Field', 'Ripe Day', 'Harvest Day, Amount', 'Harvest Penalty', 'End Penalty']
        a_total = 0
        b_total = 0
        for p in P:
            a = sum([
                r_a * Z2[p, j].X * (j - d[p]) - r_b * Z1[p, j].X * (j - d[p]) for j in J
            ]) 
            
            b = sum([abs(J[-1] + 1 - ripe_days[p]) * (1 - sum([X[p, j].X for j in J]))])
            a_total += a
            b_total += b
            
            days_harvested = ' '.join([str((j, X[p, j].X)) for j in J if round(X[p, j].X, 2) > 0])
            table.append(['', p, ripe_days[p], days_harvested, round(a), round(b)])
        table.append(['Totals', '', '', '', a_total, b_total])
        print(tabulate(table, headers=headers))
    return m.ObjVal

if __name__ == "__main__":
    import time
    
    params = ModelParams(20, 10, 3, 0.99)
    t1 = time.time()
    print('Obj:', run_original_paper_model(params))
    t2 = time.time()
    print('Time taken:', t2-t1)
    
    t1 = time.time()
    print('Obj:', run_paper_model(params))
    t2 = time.time()
    print('Time taken:', t2-t1)
    
    t1 = time.time()
    print('Obj:', run_node_model(params))
    t2 = time.time()
    print('Time taken:', t2-t1)
    
    # t1 = time.time()
    # run_node_model(params)
    # t2 = time.time()
    
    # print('Time taken:', t2-t1)
    