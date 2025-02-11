import gurobipy as gp
import numpy as np
from util import Node
import random
from collections import defaultdict
from tabulate import tabulate
from tqdm import tqdm
from copy import deepcopy
import time

class ModelParams:
    """
    Stores model parameters.
    """
    def __init__(
        self,
        num_fields : int,
        num_days : int,
        cap : float,
        alpha : float,
        seed : int,
        verbose : bool = False,
        Lambda : float = 0.5,
        period : int = 30,
        output_table : bool = False):
        
        self.num_fields = num_fields
        self.num_days = num_days
        self.cap = cap
        self.alpha = alpha
        self.verbose = verbose
        self.Lambda = Lambda
        self.period = period
        self.seed = seed
        self.output_table = output_table
        
        random.seed(seed)
        self.rain_probs = []
        for _ in range(period + num_days):
            self.rain_probs.append(random.random())
        
        random.seed(seed)
        self.ripe_days = []
        for _ in range(num_fields):
            self.ripe_days.append(random.choice(range(period)))
            
        self.unharvested = {
            p : 1 for p in range(num_fields)
        }
        
class NodesOutput:
    """
    Stores information for get_nodes
    \\
        N : list of all Nodes in tree structure
        end_nodes : leaf nodes
        next_rain_node : first non-fixed rain node
        next_clear_node : first non-fixed non-rain node
        fixed_nodes : list of fixed nodes
    """
    def __init__(self,
        N : list[Node] = list(),
        end_nodes : list[Node] = list(),
        root_node : Node = None,
        next_rain_node : Node = None,
        next_clear_node : Node = None,
        fixed_nodes : list[Node] = list(),
        ):
        
        self.N = N
        self.end_nodes = end_nodes
        self.root_node = root_node
        self.next_rain_node = next_rain_node
        self.next_clear_node = next_clear_node
        self.fixed_nodes = fixed_nodes
    
def get_nodes(params : ModelParams, fixed_nodes : list[Node] = []):
    """
    Returns a NodeOutput object which contains relevent tree info.
    \\
        params : Model parameters
        fixed_nodes : list of Nodes which are fixed
    """
    rain_probs = params.rain_probs
    # end_cap = (1 - np.mean(rain_probs[:params.num_days])) * params.cap
    end_cap = (1 - np.mean(rain_probs[:len(fixed_nodes) + params.num_days])) * params.cap
    num_fields = params.num_fields
    
    # number of extension nodes needed to harvest all fields
    days_required = max(int(num_fields // end_cap) + 1, params.period - params.num_days)
    
    output = NodesOutput([], [], None, None, None, [])
    root_node = Node(None, None, -1)
    output.root_node = root_node
    
    # number of days to consider before extension nodes
    day_limit = params.num_days + len(fixed_nodes)
    day_limit = min(params.num_days + len(fixed_nodes), params.period+1)
    
    frontier : list[Node] = list()
    
    current_node = root_node
    for fixed_node in fixed_nodes:
        output.N.append(current_node)
        current_node = Node(fixed_node.rain, current_node, current_node.stage+1, 1)
        current_node.is_fixed = True
        output.fixed_nodes.append(current_node)
    
    
    frontier.append(current_node)
    while len(frontier) > 0:
        node = frontier.pop(0)
        
        output.N.append(node)
            
        if node.stage < day_limit-1:
            rain_node = Node(True, node, node.stage+1)
            rain_node.prob = rain_probs[node.stage+1] * node.prob

            no_rain_node = Node(False, node, node.stage+1)
            no_rain_node.prob = node.prob * (1 - rain_probs[node.stage+1])
            
            if output.next_clear_node == None:
                output.next_clear_node = no_rain_node
                output.next_rain_node = rain_node
                
            frontier.append(rain_node)
            frontier.append(no_rain_node)
            
        elif node.stage >= day_limit-1 and node.stage < day_limit+days_required:
            next_node = Node(False, node, node.stage+1)
            next_node.is_extension = True
            next_node.prob = node.prob
            frontier.append(next_node)
        
        else:
            output.end_nodes.append(node)
            
    return output

class CVaROutput:
    """
    Stores output for run_cvar
    \\
        clear_couple : tuple of node on first non-fixed clear day, and harvesting proportions on each field
        rain_couple : tuple of node on first non-fixed rain day, and harvesting proportions on each field (zero)
    """
    def __init__(self):
        self.clear_couple : tuple[Node, list[float]] = None
        self.rain_couple : tuple[Node, list[float]] = None
        self.profit : float = 0.0
        self.rain_profit : float = 0.0
        self.clear_profit : float = 0.0
        self.phs_fields = []
        
def run_cvar(params : ModelParams, last_run : bool = False, fixed_nodes_values : list[tuple[Node, list[float]]] = []):
    """
    Run the CVaR implementation of the model, with extension nodes.
    \\
    fixed_node_values : list tuples of the node, and the harvesting proportions of each field in that node for prior runs
    """
    """
    Sets & Data
    """
    t1 = time.time()
    m = gp.Model()
    if not params.verbose:
        m.setParam('OutputFlag', 0)
        
    P = range(params.num_fields)
    
    """
    Data
    """
    
    ripe_days = params.ripe_days
    
    # maybe delete
    params.num_days = min(params.num_days, params.period - len(fixed_nodes_values) + 1)
    # print(f'\n{params.num_days}\n')
    ###
    nodes_output = get_nodes(params, fixed_nodes=[t[0] for t in fixed_nodes_values])
    root_node = nodes_output.root_node
    N_dash = [n for n in nodes_output.N[1:]]
    end_nodes = nodes_output.end_nodes
    N = nodes_output.N
    
    rain_probs = params.rain_probs
    end_cap = (1 - np.mean(rain_probs[:len(fixed_nodes_values) + params.num_days])) * params.cap
    
    verbal = params.output_table
    
    cap = params.cap
    
    phs_days = 4
    phs_decline = 0.3
    
    _rain_count = dict()
    def rain_count(n : Node):
        try:
            if n not in _rain_count:
                _rain_count[n] = 0
                if n is root_node:
                    _rain_count[n] = 0
                elif n.parent is None:
                    _rain_count[n] = n.rain
                elif n.rain == 1:
                    _rain_count[n] = 1 + rain_count(n.parent)
        except:
            # print(n.parent, n.stage, n.rain)
            print('ERROR')
            quit()
        return _rain_count[n]
    
    _total_rain = dict()
    def total_rain(n : Node):
        if n.parent is None:
            return 0
        
        if n not in _total_rain:
            _total_rain[n] = n.rain + total_rain(n.parent)
        return _total_rain[n]

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
    yield_rain = 0.2
    lowest_yield = 0.05
    max_yield = 1
    max_yield = {p : 2 + np.mean(params.rain_probs) * ripe_days[p] * yield_rain for p in P}
    
    def get_yield(p : int, n : Node):
        return max(lowest_yield, max_yield[p] - abs(n.stage - ripe_days[p]) * yield_dec - total_rain(n) * yield_rain)
    
    Yield = {(p, n) : get_yield(p, n) for p in P for n in N_dash}

    quality_price = 2
    phs_price = 1
    
    PHS = {(p, n) : phs(p, n) for p in P for n in N_dash}
    
    Lambda = params.Lambda
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
        for p in P for n in N_dash
    }
    
    Z = {
        n : m.addVar()
        for n in N_dash
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
    Fix Variables
    """
    for i, fixed in enumerate(nodes_output.fixed_nodes):
        for p, prop in enumerate(fixed_nodes_values[i][1]):
            m.addConstr(
                X[p, fixed] == prop
            )
    
    RainZeroProd = {
        (p, n) : m.addConstr(X[p, n] == 0)
        for p in P for n in N if n.rain == 1
    }
    
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
        for n in N_dash if n.is_extension
    }
    
    # Fields total harvest must be less than or equal to 1 over route
    Three = {
        (p, tuple(route[n])) : m.addConstr(
            gp.quicksum(X[p, n] for n in route[n] if n != root_node) <= 1
        )
        for p in P for n in end_nodes
    }
    
    # The profit in a node from harvesting wheat is dictated by presence of PHS and sum of yield
    Four = {
        n : m.addConstr(
            Z[n] == gp.quicksum(
                X[p, n] * Yield[p, n] * (quality_price + PHS[p, n] * (phs_price - quality_price))
                for p in P)
        )
        for n in Z
    }
    
    """
    CVaR Constraints
    """
    
    # End profits determined by sum of profits over routes
    Five = {
        n : m.addConstr(
            Beta[n] == gp.quicksum(
                Z[nn] for nn in route[n]
            )
        )
        for n in end_nodes
    }
    
    # Sum of Beta and BetaMinus for all end-routes >= Var
    Six = {
        n : m.addConstr(
           Beta[n] + BetaMinus[n] >= Var 
        )
        for n in end_nodes
    }
    
    # Set CVaR
    Seven = m.addConstr(
        CVar == Var - (1 / alpha) * gp.quicksum(BetaMinus[n] * n.prob for n in end_nodes)
    )
    
    t2 = time.time()
    m.optimize()
    t3 = time.time()
    
    if verbal:
        print('Setup Time', t2-t1)
        print('Solve Time', t3-t2)

        selected_end = np.random.choice(end_nodes)
        selected = []
        nn = selected_end
        while nn != root_node:
            selected.append(nn)
            nn = nn.parent
        selected = list(reversed(selected))
        
        print('Randomly Selected Scenario')
        print('End Cap:', end_cap)
        print('Day Table:')
        table = []
        headers = ['Day', 'Fixed', 'Weather', 'Prob Rain', 'Field, Harvest Prop, Yield', 'Ripe Fields', 'PHS Fields', 'Profit']

        for n in selected:
            n : Node
            weather = 'Rains' if n.rain == 1 else 'Clear'
            is_fixed = 'Y' if n.is_fixed else 'N'
            ripe = ' '.join([str(p) for p in P if ripe_days[p] == n.stage])
            phs_fields = ' '.join([str(p) for p in P if phs(p, n) == 1])
            
            t = 'NA'
            if not n.is_fixed and not n.is_extension:
                t = rain_probs[n.stage]
                
            if n.rain == 0:
                fields = ' '.join([str((p, round(X[p, n].X, 2), round(get_yield(p, n), 3))) for p in P if round(X[p, n].X, 2) > 0])
                
                profit = Z[n].X
                
                table.append(
                    [n.stage, is_fixed, weather, t, fields, ripe, phs_fields, profit]
                )
            else:
                table.append(
                    [n.stage, is_fixed, weather, t, '', ripe, phs_fields, 0]
                )

        print(tabulate(table, headers=headers))
        
        print('Expected', Lambda * sum([n.prob * Beta[n].X for n in end_nodes]))
        print('Cvar', (1 - Lambda) * CVar.X)
        print('Obj', m.ObjVal)
    
    output = CVaROutput()
    output.clear_couple = (nodes_output.next_clear_node, [X[p, nodes_output.next_clear_node].X for p in P])
    output.rain_couple = (nodes_output.next_rain_node, [0 for _ in P])
    output.profit = Z[nodes_output.next_clear_node].X
    
    if (last_run):
        output.clear_profit = sum([Beta[n].x for n in end_nodes if nodes_output.next_clear_node in route[n]])
        output.rain_profit = sum([Beta[n].x for n in end_nodes if nodes_output.next_clear_node not in route[n]])
        
        # import matplotlib.pyplot as plt
        # import networkx as nx
        # N = nodes_output.N
        # G = nx.DiGraph()
        # vertices_dict = {n : i-1 for i, n in enumerate(N)}
        # edges = []
        # G.add_nodes_from(vertices_dict.values())
        # for n in N:
        #     if n.parent is None:
        #         continue
        #     edges.append((vertices_dict[n.parent], vertices_dict[n]))
        # G.add_edges_from(edges)
        # pos = nx.nx_agraph.graphviz_layout(G)
        # nx.draw(G, pos, with_labels=True, 
        #             node_size=10, 
        #             font_size=5, font_color="black", 
        #             edge_color="gray", linewidths=1.5)
        # plt.show()
    
    return output

def run_through(params : ModelParams, verbose : bool = True):
    """
    Runs the CVaR implementation with the given parameters through the whole period, fixing each first branch node as it goes.
    """
    params.output_table = False
    selects = []
    
    np.random.seed(params.seed)
    
    outcomes = []
    profits = []
    total_profit = 0
    
    test_profit = 0
    
    for i in tqdm(range(params.period+1), disable=not verbose):
        last_run = i == params.period
        
        model_output = run_cvar(params, fixed_nodes_values=selects, last_run=last_run)
        
        if (np.random.random() > params.rain_probs[i]):
            selects.append(model_output.clear_couple)
            outcomes.append(True)
            
            total_profit += model_output.profit
            profits.append(model_output.profit)
            test_profit = model_output.clear_profit
            # total_profit += model_output.profit
        else:
            selects.append(model_output.rain_couple)
            outcomes.append(False)
            profits.append(0)
            test_profit = model_output.rain_profit
    
    if verbose:
        table = []
        headers = ['Day', 'Available', 'Prob Rain', 'Field, Harvest Prop', 'Ripe Fields', 'Profit']
        for day, pair in enumerate(selects):
            node, harvested = pair
            
            entry = [day, outcomes[day], params.rain_probs[day]]
            
            fields = ' '.join([str((p, round(harvested[p], 2))) for p in range(params.num_fields) if harvested[p] > 0])
            entry.append(fields)
            
            ripe = ' '.join([str(p) for p in range(params.num_fields) if params.ripe_days[p] == day])
            entry.append(ripe)
            
            entry.append(profits[day])
            
            table.append(entry)
        print(tabulate(table, headers=headers))
        print('Profit:', total_profit)
        print('Test Profit', test_profit)
    return test_profit

def graph():
    num_fields = 20
    period = 30
    cap = 3
    max_window = 8
    num_samples = 50
    
    ratio = defaultdict(list)
    
    for i in tqdm(range(num_samples)):
        params = ModelParams(num_fields, max_window, cap, 0.2, verbose = False, output_table=False, period = period, seed=300*i)
        den = run_through(params, verbose=False)
        
        if (den == 0):
            continue
        
        for window in range(1, max_window):
            params = ModelParams(num_fields, window, cap, 0.2, verbose = False, output_table=False, period = period, seed=300*i)
            num = run_through(params, verbose=False)
            
            ratio[window].append(num / den)
            
        if (den == 0):
            continue
    
    import csv
    with open("test.csv", "wb") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(ratio.keys())
        writer.writerows(zip(*ratio.values()))

if __name__ == "__main__":
    graph()

# num_fields = 20
# period = 30
# cap = 3
# params = ModelParams(num_fields, 7, cap, 0.2, verbose = False, output_table=False, period = period, seed=22)
# run_through(params, verbose=True)

# params = ModelParams(num_fields, 2, cap, 0.2, verbose = False, output_table=False, period = period, seed=22892)
# run_through(params, verbose=True)


# num_fields = 1
# period = 20
# cap = 0.5
# params = ModelParams(num_fields, 15, cap, 0.2, verbose = True, output_table=True, period = period, seed=22892)
# run_cvar(params)

"""
1 0.8293652156236265
2 0.8580727490523397
3 0.9080917264688091
4 0.9257305990599591
5 0.9737671662686429
6 0.9821306736907093
7 0.99176476217142
"""