import gurobipy as gp
import numpy as np
from util import get_nodes, ModelParams, Node
import random
from collections import defaultdict
from tabulate import tabulate
from tqdm import tqdm

def create_model(params : ModelParams, seed : int):
    model = gp.Model()
    
    cap = params.cap
    num_days = params.num_days
    alpha = params.alpha
    quality_price = 2
    phs_price = 1
    Lambda = 0.5
    
    P = range(params.num_fields)
    D = range(params.num_days)
    
    ripe_days = []
    
    random.seed(seed)
    probs = [random.random() for d in D]
    for p in P:
        ripe_days.append(random.choice(D))
    
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
    phs_days = 4
    phs_decline = 0.3
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
    
    max_yield = 1
    yield_dec = 0.1
    yield_rain = 0.15
    lowest_yield = 0.1
    def get_yield(p : int, n : Node):
        return max(lowest_yield, 
            max_yield - abs(n.stage - ripe_days[p]) * yield_dec - rain_count(n) * yield_rain)
    
    N = []
    children = defaultdict(list)
    children_cdfs = defaultdict(list)
    
    root_node = Node(None, None, -1)
    
    X = dict()
    Y = dict()
    Z = dict()
    Beta = dict()
    BetaMinus = dict()
    
    Var = model.addVar()
    CVar = model.addVar()
    
    LinExp = {'SevenRHS' : gp.LinExpr(1, Var), 'Objective' : gp.LinExpr(1 - Lambda, CVar)}

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
        
        for p in P:
            if node.rain == 0 and node != root_node:
                X[p, node] = model.addVar(ub=1)
            Y[p, node] = model.addVar(ub=1)
        if node.rain == 0 and node != root_node:
            Z[node] = model.addVar()
            
        if node.stage == num_days - 1:
            Beta[node] = model.addVar()
            BetaMinus[node] = model.addVar()
            
        # Constraint One
        if node.rain == 0 and node != root_node:
            model.addConstr(
                gp.quicksum(X[p, node] for p in P) <= cap
            )
            
        # Constraint Two
        if node == root_node:
            for p in P:
                model.addConstr(
                    Y[p, node] == 1
                )
        
        # Constraint 3A
        if node != root_node and node.rain == 0:
            for p in P:
                model.addConstr(Y[p, node] == Y[p, node.parent] - X[p, node])
                
        # Constraint 3B
        if node != root_node and node.rain == 1:
            for p in P:
                model.addConstr(Y[p, node] == Y[p, node.parent])
                
        # Constraint 4
        if node.rain == 0 and node != root_node:
            model.addConstr(
                Z[node] == gp.quicksum(
                    X[p, node] * get_yield(p, node) * (quality_price + phs(p, node) * (phs_price - quality_price))
                    for p in P)
            )
        
        if node.stage == num_days - 1:
            route = []
            if node.rain == 0:
                route = [node]
            nn = node.parent
            while nn != root_node:
                if nn.rain == 0:
                    route.append(nn)
                nn = nn.parent
                
            # Constraint 5
            model.addConstr(
                Beta[node] == gp.quicksum(
                    Z[nn] for nn in route
                )
            )
            
            # Constraint 6
            model.addConstr(
                Beta[node] + BetaMinus[node] >= Var
            )
            
            # Constraint 7
            LinExp['SevenRHS'] += (-1 / alpha) * BetaMinus[node] * node.prob
            
            LinExp['Objective'] += Lambda * node.prob * Beta[node]
            
        if node.stage < num_days-1:
            rain_node = Node(True, node, node.stage+1)
            children[node].append(rain_node)
            children_cdfs[node].append(probs[node.stage+1])
            rain_node.prob = probs[node.stage+1] * node.prob
            extend_tree(rain_node)

            no_rain_node = Node(False, node, node.stage+1)
            children[node].append(no_rain_node)
            children_cdfs[node].append(1.0) 
            no_rain_node.prob = node.prob * (1 - probs[node.stage+1])
            extend_tree(no_rain_node)

    extend_tree(root_node)
    
    Seven = model.addConstr(
        CVar == LinExp['SevenRHS']
    )

    model.setObjective(
        LinExp['Objective'],
        gp.GRB.MAXIMIZE
    )
    
    model.optimize()
    
    print('Obj', model.ObjVal)
    
    # from util import sample
    # random.seed(24)
    
    # print('Randomly Selected Scenario')
    
    # print('Day Table:')
    # table = []
    # headers = ['Day', 'Weather', 'Prob Rain', 'Field, Harvest Prop, Yield', 'Ripe Fields', 'PHS Fields', 'Profit', 'Yields']
    # selected = sample(root_node, params.num_days, children, children_cdfs)
    
    # for n in selected:
    #     weather = 'Rains' if n.rain == 1 else 'Clear'
    #     ripe = ' '.join([str(p) for p in P if ripe_days[p] == n.stage])
    #     phs_fields = ' '.join([str(p) for p in P if phs(p, n) == 1])
    #     if (p, n) in X:
    #         # fields = ' '.join([str((p, round(X[p, n].X, 2))) for p in P if round(X[p, n].X, 2) > 0])
    #         fields = ' '.join([str((p, round(X[p, n].X, 2), round(get_yield(p, n), 3))) for p in P if round(X[p, n].X, 2) > 0])
            
    #         profit = Z[n].X
            
    #         table.append(
    #             [n.stage, weather, probs[n.stage], fields, ripe, phs_fields, profit]
    #         )
    #     else:
    #         table.append(
    #             [n.stage, weather, probs[n.stage], '', ripe, phs_fields, 0]
    #         )

    # print(tabulate(table, headers=headers))
    
    # print('Expected', Lambda * sum([n.prob * Z[n].X for n in Z]))
    # print('Cvar', (1 - Lambda) * CVar.X)
    # return N, children, children_cdfs

def main():
    num_fields = 10
    num_days = 15
    cap = 3
    
    params = ModelParams(num_fields, num_days, cap, 0.2, verbose = True)
    import time
    t1 = time.time()
    create_model(params, 30)
    t2 = time.time()
    print(t2 - t1)
    
main()