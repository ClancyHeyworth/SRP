import gurobipy as gp
import numpy as np
from util2 import get_nodes, ModelParams, Node
import random
from collections import defaultdict
from tabulate import tabulate
from tqdm import tqdm

def run_window(params : ModelParams, window : int, seed : int, nodes_dict : dict[int, list[Node]]):
    """
    Sets & Data
    """
    P = range(params.num_fields)
    D = range(params.num_days)
    
    random.seed(seed)
    Probs = [random.random() for d in D]
    
    ripe_days = []
    for p in P:
        ripe_days.append(random.choice(D))
    # print([(p, ripe_days[p]) for p in P])
    # print([(d, Probs[d]) for d in D])
    
    params.num_days = window
    # N, children, children_cdfs = get_nodes(params.num_days, Probs)
    N = []
    for i in range(-1, params.num_days):
        N += nodes_dict[i]
    # N = [n for n in N if n.stage < params.num_days]
    # print(max([n.stage for n in N]), params.num_days)
    # quit()
    N_dash = N[1:]
    # N_dash = [nodes_dict[i]]
    root_node = nodes_dict[-1][0]
    
    cap = params.cap
    
    phs_days = 4
    phs_decline = 0.3
    
    _rain_count = dict()
    def rain_count(n : Node):
        if n not in _rain_count:
            if n is root_node:
                _rain_count[n] = 0
            elif n.parent is None:
                _rain_count[n] = n.rain
            elif n.rain == 1:
                _rain_count[n] = 1 + rain_count(n.parent)
            else:
                _rain_count[n] = 0
        return _rain_count[n]
    
    _phs = dict()
    def phs(p : int, n : Node):
        if (p, n) not in _phs:
            if n is None:
                return 0
            
            _phs[p, n] = 0
            if rain_count(n) >= max([1, phs_days - 
                max([0, (n.stage - ripe_days[p]) * phs_decline])]):
                _phs[p, n] = 1
            elif phs(p, n.parent) == 1:
                _phs[p, n] = 1
        return _phs[p, n]
    
    base_yield = 1
    yield_dec = 0.1
    yield_inc = 0.1
    yield_rain = 0.15
    lowest_yield = 0.1
    
    # def get_yield(p : int, n : Node):
    #     if n == root_node:
    #         return base_yield
    #     else:
    #         parent_yield = get_yield(p, n.parent)
    #         change = base_yield * yield_inc * (1 if n.stage <= ripe_days[p] else 0) \
    #             - yield_dec * (1 if n.stage > ripe_days[p] else 0) - yield_rain * n.rain
    #         return max(lowest_yield, parent_yield + change)
    max_yield = 1
    def get_yield(p : int, n : Node):
        return max(lowest_yield, max_yield - abs(n.stage - ripe_days[p]) * yield_dec - rain_count(n) * yield_rain)
    
    Yield = {(p, n) : get_yield(p, n) for p in P for n in N_dash}

    quality_price = 2
    phs_price = 1
    
    PHS = {(p, n) : phs(p, n) for p in P for n in N_dash}
    
    Lambda = 0.5
    alpha = params.alpha
    
    end_nodes = [n for n in N if n.stage == params.num_days - 1]
    
    """
    Variables
    """
    m = gp.Model()
    
    X = {
        (p, n) : m.addVar(ub=1)
        for p in P for n in N_dash if not n.rain
    }
    
    Y = {
        (p, n) : m.addVar(ub=1)
        for p in P for n in N
    }
    
    Z = {
        n : m.addVar()
        for n in N_dash if not n.rain
    }
    
    Beta = {
        n : m.addVar()
        for n in end_nodes
    }
    
    BetaMinus = {
        n : m.addVar()
        for n in end_nodes
    }
    
    Var = m.addVar()
    CVar = m.addVar()
    
    """
    Objective
    """
    
    m.setObjective(
        Lambda * gp.quicksum(n.prob * Beta[n] for n in end_nodes)
        + (1 - Lambda) * CVar,
        gp.GRB.MAXIMIZE
    )
    
    """
    Constraints
    """
    
    One = {
        n : m.addConstr(
            gp.quicksum(X[p, n] for p in P) <= cap
        )
        for n in N_dash if not n.rain
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
        for p in P for n in N_dash if not n.rain
    }

    ThreeB = {
        (p, n) : m.addConstr(
            Y[p, n] == Y[p, n.parent]
        )
        for p in P for n in N_dash if n.rain
    }
    
    Four = {
        n : m.addConstr(
            Z[n] == gp.quicksum(
                X[p, n] * Yield[p, n] * (quality_price + PHS[p, n] * (phs_price - quality_price))
                for p in P)
        )
        for n in Z
    }
    
    route = defaultdict(list)
    for n in end_nodes:
        nn = n
        while nn != root_node:
            if nn.rain == 0:
                route[n].append(nn)
            nn = nn.parent
            
    Five = {
        n : m.addConstr(
            Beta[n] == gp.quicksum(
                Z[nn] for nn in route[n]
            )
        )
        for n in end_nodes
    }
    
    Six = {
        n : m.addConstr(
           Beta[n] + BetaMinus[n] >= Var 
        )
        for n in end_nodes
    }
    
    Seven = m.addConstr(
        CVar == Var - (1 / alpha) * gp.quicksum(BetaMinus[n] * n.prob for n in end_nodes)
    )
    
    m.setParam('OutputFlag', 0)
    m.optimize()
    
    FirstNode = [n for n in N if n.stage == 0 and n.rain == 0][0]
    selected = [X[p, FirstNode].X for p in P]
    
    return selected

def similarity(a, b):
    if len(a) == 0:
        return 1 if len(b) == 0 else 0
    return len(set(a) & set(b)) / len(set(a) | set(b))

def weighted_jaccard(a, b):
    num = sum([min(a[i], b[i]) for i in range(len(a))]) 
    den = sum([max(a[i], b[i]) for i in range(len(a))])
    if den == 0:
        return 1
    return num / den

def main():
    num_days = 10
    num_fields = 10
    cap = 3

    similarity_map = {
        i : [] for i in range(1, num_days)
    }
        
    sample_num = 100
    for j in tqdm(range(sample_num)):
        params = ModelParams(num_fields, num_days, cap, 0.2, verbose = True)
        seed = j
        random.seed(seed)
        Probs = [random.random() for d in range(params.num_days)]
        nodes_dict = get_nodes(params.num_days, Probs)
        
        main_window = run_window(params, num_days, j, nodes_dict)

        for i in range(1, num_days):
            params = ModelParams(num_fields, num_days, cap, 0.2, verbose = True)
            selected = run_window(params, i, j, nodes_dict)
            
            similarity_map[i].append(weighted_jaccard(selected, main_window))

    # for i in range(1, num_days):
    #     print(i, sum(similarity_map[i]) / sample_num)
    print('[' + ','.join([str(sum(similarity_map[i]) / sample_num) for i in range(1, num_days)]) + ']')
        
if __name__ == "__main__":
    main()

