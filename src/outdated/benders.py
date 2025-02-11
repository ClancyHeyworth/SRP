import gurobipy as gp
import numpy as np
from util import ModelParams, Node
import random
from collections import defaultdict
from tabulate import tabulate
from tqdm import tqdm

def get_nodes(num_days:int, num_fields : int, cap : float, rain_probs : list[float]):
    N = []
    
    root_node = Node(None, None, -1)
    
    days_required = int(num_fields // (np.median(rain_probs) * cap)) + 1
    
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
            rain_node.prob = rain_probs[node.stage+1] * node.prob
            extend_tree(rain_node)

            no_rain_node = Node(False, node, node.stage+1)
            no_rain_node.prob = node.prob * (1 - rain_probs[node.stage+1])
            extend_tree(no_rain_node)
        
        elif node.stage >= num_days-1 and node.stage < num_days+days_required:
            next_node = Node(False, node, node.stage+1)
            next_node.prob = node.prob
            extend_tree(next_node)
        
        else:
            end_nodes.append(node)

    extend_tree(root_node)
    
    return N, end_nodes

def run_cvar(params : ModelParams, seed : int, window : int, verbal : bool = False):
    """
    Sets & Data
    """
    m = gp.Model()
    if not verbal:
        m.setParam('OutputFlag', 0)
        
    P = range(params.num_fields)
    D = range(params.num_days)
    
    random.seed(seed)
    np.random.seed(seed)
    
    Probs = [random.random() for d in D]
    end_cap = np.median(Probs) * params.cap
    Probs = Probs[:window]
    
    random.seed(seed)
    ripe_days = []
    for p in P:
        ripe_days.append(random.choice(D))
    
    D = D[:window]
    params.num_days = window

    N, end_nodes = get_nodes(params.num_days, params.num_fields, params.cap, Probs[:params.num_days+1])
    N_dash = N[1:]
    root_node = N[0]
    
    cap = params.cap
    
    phs_days = 4
    phs_decline = 0.3
    
    _rain_count = dict()
    def rain_count(n : Node):
        if n not in _rain_count:
            _rain_count[n] = 0
            if n is root_node:
                _rain_count[n] = 0
            elif n.parent is None:
                _rain_count[n] = n.rain
            elif n.rain == 1:
                _rain_count[n] = 1 + rain_count(n.parent)
        return _rain_count[n]
            

    def phs(p : int, n : Node):
        if n is None:
            return 0
        if rain_count(n) >= max([1, phs_days - 
            max([0, (n.stage - ripe_days[p]) * phs_decline])]):
            return 1
        if phs(p, n.parent) == 1:
            return 1
        return 0
    
    yield_dec = 0.1
    yield_rain = 0.15
    lowest_yield = 0
    max_yield = 1
    
    def get_yield(p : int, n : Node):
        return max(lowest_yield, max_yield - abs(n.stage - ripe_days[p]) * yield_dec - rain_count(n) * yield_rain)
    
    Yield = {(p, n) : get_yield(p, n) for p in P for n in N_dash}

    quality_price = 2
    phs_price = 1
    
    PHS = {(p, n) : phs(p, n) for p in P for n in N_dash}
    
    Lambda = 0.5
    alpha = params.alpha
    
    route = defaultdict(list)
    for n in end_nodes:
        nn = n
        while nn != root_node:
            if nn.rain == 0:
                route[n].append(nn)
            nn = nn.parent
    
    """
    Variables
    """
    
    X = {
        (p, n) : m.addVar(ub=1)
        for p in P for n in N_dash if n.rain == 0
    }
    
    Z = {
        n : m.addVar()
        for n in N_dash if n.rain == 0
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
    
    # Cap on nodes past num_days different
    OneB = {
        n : m.addConstr(
            gp.quicksum(X[p, n] for p in P) <= end_cap
        )
        for n in N_dash if n.stage >= params.num_days
    }
    
    Three = {
        (p, tuple(route[n])) : m.addConstr(
            gp.quicksum(X[p, n] for n in route[n] if n != root_node) <= 1
        )
        for p in P for n in end_nodes
    }
    
    Four = {
        n : m.addConstr(
            Z[n] == gp.quicksum(
                X[p, n] * Yield[p, n] * (quality_price + PHS[p, n] * (phs_price - quality_price))
                for p in P)
        )
        for n in Z
    }
    
    # CVaR Constraints
    
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
    
    m.optimize()
    
    if verbal:
        selected_end = np.random.choice(end_nodes)
        selected = []
        nn = selected_end
        while nn != root_node:
            selected.append(nn)
            nn = nn.parent
        selected = reversed(selected)
        
        print('Randomly Selected Scenario')
        print('Day Table:')
        table = []
        headers = ['Day', 'Weather', 'Prob Rain', 'Field, Harvest Prop, Yield', 'Ripe Fields', 'PHS Fields', 'Profit', 'Yields']

        for n in selected:
            weather = 'Rains' if n.rain == 1 else 'Clear'
            ripe = ' '.join([str(p) for p in P if ripe_days[p] == n.stage])
            phs_fields = ' '.join([str(p) for p in P if PHS[p, n] == 1])
            
            t = 1
            if n.stage < len(Probs):
                t = Probs[n.stage]
                
            if (p, n) in X:
                fields = ' '.join([str((p, round(X[p, n].X, 2), round(Yield[p, n], 3))) for p in P if round(X[p, n].X, 2) > 0])
                
                profit = Z[n].X
                
                table.append(
                    [n.stage, weather, t, fields, ripe, phs_fields, profit]
                )
            else:
                table.append(
                    [n.stage, weather, t, '', ripe, phs_fields, 0]
                )

        print(tabulate(table, headers=headers))
        
        print('Expected', Lambda * sum([n.prob * Beta[n].X for n in end_nodes]))
        print('Cvar', (1 - Lambda) * CVar.X)
        print('Obj', m.ObjVal)
    
    FirstNode = [n for n in N if n.stage == 0 and n.rain == 0][0]
    selected = [X[p, FirstNode].X for p in P]
    
    print('obj', m.ObjVal)
    
    return selected
    return m.ObjVal

# params = ModelParams(10, 10, 3, 0.2, verbose = True)
# import time
# t1 = time.time()
# print('Obj:', run_cvar(params, 30))
# t2 = time.time()
# print(t2 - t1)

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
        
    sample_num = 50
    for j in tqdm(range(sample_num)):
        params = ModelParams(num_fields, num_days, cap, 0.2, verbose = True)
        
        main_window = run_cvar(params, j, params.num_days)

        for i in range(1, num_days):
            params = ModelParams(num_fields, num_days, cap, 0.2, verbose = True)
            selected = run_cvar(params, j, i)
            
            similarity_map[i].append(weighted_jaccard(selected, main_window))
    print('[' + ','.join([str(sum(similarity_map[i]) / sample_num) for i in range(1, num_days)]) + ']')
            
# main()

import time
num_days = 15
num_fields = 10
cap = 3
params = ModelParams(num_fields, num_days, cap, 0.2, verbose = True)

t1 = time.time()
run_cvar(params, 54, params.num_days, verbal = True)
t2 = time.time()
print(t2 - t1)

# from cvar_extended import run_cvar as run_cvar2
# t1 = time.time()
# run_cvar2(params, 0, params.num_days, verbal = True)
# t2 = time.time()
# print(t2 - t1)