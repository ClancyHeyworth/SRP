import gurobipy as gp
import numpy as np
from util import get_nodes, ModelParams, Node
import random
from collections import defaultdict
from tabulate import tabulate
from tqdm import tqdm

def run_weather(params : ModelParams):
    """
    Sets & Data
    """
    P = range(params.num_fields)
    D = range(params.num_days)
    
    random.seed(1)
    Probs = [random.random() for d in D]
    
    ripe_days = []
    for p in P:
        ripe_days.append(random.choice(D))
        
    N, children, children_cdfs = get_nodes(params.num_days, Probs)
    N_dash = N[1:]
    root_node = N[0]
    # print(N[:3])
    # quit()
    
    cap = params.cap
    
    phs_days = 4
    phs_decline = 0.3
    
    def rain_count(n : Node):
        if n is root_node:
            return 0
        if n.parent is None:
            return n.rain
        if n.rain == 1:
            return 1 + rain_count(n.parent)
        return 0
    
    def phs(p : int, n : Node):
        if n is None:
            return 0
        if rain_count(n) >= max([1, phs_days - 
            max([0, (n.stage - ripe_days[p]) * phs_decline])]):
            return 1
        if phs(p, n.parent) == 1:
            return 1
        return 0
    
    base_yield = 1
    yield_dec = 0.1
    yield_inc = 0.1
    yield_rain = 0.15
    lowest_yield = 0.1
    
    def get_yield(p : int, n : Node):
        if n == root_node:
            return base_yield
        else:
            parent_yield = get_yield(p, n.parent)
            change = base_yield * yield_inc * (1 if n.stage <= ripe_days[p] else 0) \
                - yield_dec * (1 if n.stage > ripe_days[p] else 0) - yield_rain * n.rain
            return max(lowest_yield, parent_yield + change)
    
    Yield = {(p, n) :  get_yield(p, n) for p in P for n in N_dash}

    quality_price = 2
    phs_price = 1
    
    PHS = {(p, n) : phs(p, n) for p in P for n in N_dash}
    
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
        (p, n) : m.addVar()
        for p, n in X
    }
    
    """
    Objective
    """
    
    m.setObjective(
        gp.quicksum(n.prob * Z[p, n] for p, n in Z),
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
    
    FourA = {
        (p, n) : m.addConstr(
            Z[p, n] <= X[p, n] * Yield[p, n] * quality_price
        )
        for p, n in X
    }
    
    M = 2**20
    FourB = {
        (p, n) : m.addConstr(
            Z[p, n] <= X[p, n] * Yield[p, n] * phs_price + M*(1 - PHS[p, n]) 
        )
        for p, n in X
    }
    
    m.optimize()
    
    from util import sample
    random.seed(24)
    
    print('Randomly Selected Scenario')
    
    print('Day Table:')
    table = []
    headers = ['Day', 'Weather', 'Prob Rain', 'Field, Harvest Prop', 'Ripe Fields', 'PHS Fields', 'Profit', 'Yields']
    selected = sample(root_node, params.num_days, children, children_cdfs)
    
    for n in selected:
        temp = ','.join([str(Yield[p, n]) for p in P])
        # break
        # print(j, sum([X[p, j].X for p in P]))
        weather = 'Rains' if n.rain == 1 else 'Clear'
        ripe = ' '.join([str(p) for p in P if ripe_days[p] == n.stage])
        phs_fields = ' '.join([str(p) for p in P if PHS[p, n] == 1])
        if (p, n) in X:
            # fields = ' '.join([str((p, round(X[p, n].X, 2))) for p in P if round(X[p, n].X, 2) > 0])
            fields = ' '.join([str((p, round(Yield[p, n], 3))) for p in P if round(X[p, n].X, 2) > 0])
            
            profit = sum([Z[p, n].X for p in P])
            
            table.append(
                [n.stage, weather, Probs[n.stage], fields, ripe, phs_fields, profit]
            )
        else:
            table.append(
                [n.stage, weather, Probs[n.stage], '', ripe, phs_fields, 0]
            )
        # print(f'Day {j}: {[(p, round(X[p, j].X, 2)) for p in P if round(X[p, j].X, 2) > 0]} Ripe: {[p for p in P if ripe_days[p] == j]} Rain Prob : {day_probs[j]}')
    print(tabulate(table, headers=headers))
    
    return m.ObjVal

def run_cvar(params : ModelParams, seed : int):
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
        
    N, children, children_cdfs = get_nodes(params.num_days, Probs)
    N_dash = N[1:]
    root_node = N[0]
    
    cap = params.cap
    
    phs_days = 4
    phs_decline = 0.3
    
    def rain_count(n : Node):
        if n is root_node:
            return 0
        if n.parent is None:
            return n.rain
        if n.rain == 1:
            return 1 + rain_count(n.parent)
        return 0
    
    def phs(p : int, n : Node):
        if n is None:
            return 0
        if rain_count(n) >= max([1, phs_days - 
            max([0, (n.stage - ripe_days[p]) * phs_decline])]):
            return 1
        if phs(p, n.parent) == 1:
            return 1
        return 0
    
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
    
    end_nodes = [n for n in N if n.stage == D[-1]]
    
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
    for n in tqdm(end_nodes):
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
    
    m.optimize()
    
    from util import sample
    random.seed(24)
    
    print('Randomly Selected Scenario')
    
    print('Day Table:')
    table = []
    headers = ['Day', 'Weather', 'Prob Rain', 'Field, Harvest Prop, Yield', 'Ripe Fields', 'PHS Fields', 'Profit', 'Yields']
    selected = sample(root_node, params.num_days, children, children_cdfs)
    
    for n in selected:
        weather = 'Rains' if n.rain == 1 else 'Clear'
        ripe = ' '.join([str(p) for p in P if ripe_days[p] == n.stage])
        phs_fields = ' '.join([str(p) for p in P if PHS[p, n] == 1])
        if (p, n) in X:
            # fields = ' '.join([str((p, round(X[p, n].X, 2))) for p in P if round(X[p, n].X, 2) > 0])
            fields = ' '.join([str((p, round(X[p, n].X, 2), round(Yield[p, n], 3))) for p in P if round(X[p, n].X, 2) > 0])
            
            profit = Z[n].X
            
            table.append(
                [n.stage, weather, Probs[n.stage], fields, ripe, phs_fields, profit]
            )
        else:
            table.append(
                [n.stage, weather, Probs[n.stage], '', ripe, phs_fields, 0]
            )

    print(tabulate(table, headers=headers))
    
    print('Expected', Lambda * sum([n.prob * Z[n].X for n in Z]))
    print('Cvar', (1 - Lambda) * CVar.X)
    return m.ObjVal

params = ModelParams(10, 15, 3, 0.2, verbose = True)
import time
t1 = time.time()
print('Obj:', run_cvar(params, 30))
t2 = time.time()
print(t2 - t1)