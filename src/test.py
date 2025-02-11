from dataclasses import dataclass
from util import Node
import random
import numpy as np

class ModelParams:
    
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

# @dataclass
# class NodesOutput:
#     N = list()
#     end_nodes = list()
#     root_node : Node = None
#     next_rain_node : Node = None
#     next_clear_node : Node = None
#     fixed_nodes = list()
    
class NodesOutput:
    def __init__(self,
        N : list[Node] = list(),
        end_nodes : list[Node] = list(),
        root_node : Node = None,
        next_rain_node : Node = None,
        next_clear_node : Node = None,
        fixed_nodes = list(),
        ):
        
        self.N = N
        self.end_nodes = end_nodes
        self.root_node = root_node
        self.next_rain_node = next_rain_node
        self.next_clear_node = next_clear_node
        self.fixed_nodes = fixed_nodes
    
def get_nodes2(params : ModelParams, fixed_nodes : list[Node] = []):
    
    rain_probs = params.rain_probs
    end_cap = (1 - np.mean(rain_probs[:params.num_days])) * params.cap
    num_fields = params.num_fields
    
    days_required = max(int(num_fields // end_cap) + 1, params.period - params.num_days)
    
    output = NodesOutput([], [], None, None, None, [])
    root_node = Node(None, None, -1)
    output.root_node = root_node
    output.fixed_nodes = fixed_nodes
    
    day_limit = params.num_days + len(fixed_nodes)
    
    frontier : list[Node] = list()
    
    current_node = root_node
    for fixed_node in fixed_nodes:
        output.N.append(current_node)
        current_node = Node(fixed_node.rain, current_node, current_node.stage+1, 1)
    
    print('Before loop', len(output.N))
    
    frontier.append(current_node)
    while len(frontier) > 0:
        node = frontier.pop(0)
        
        output.N.append(node)
        
        print('In loop', len(output.N))
            
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
            next_node.prob = node.prob
            frontier.append(next_node)
        
        else:
            output.end_nodes.append(node)
    print('After loop', len(output.N))
    return output

def main():
    num_fields = 40
    period = 4
    cap = 100000

    params = ModelParams(num_fields, 3, cap, 0.2, verbose = False, output_table=False, period = period, seed=2000)
    nodes_output = get_nodes2(params)

    next_node = Node(nodes_output.next_clear_node.rain, None, None, 1)
    nodes_output = get_nodes2(params, [next_node])
    
    # params = ModelParams(num_fields, 3, cap, 0.2, verbose = False, output_table=False, period = period, seed=2000)
    # nodes_output = get_nodes2(params)

    import matplotlib.pyplot as plt
    import networkx as nx
    N = nodes_output.N
    G = nx.DiGraph()
    vertices_dict = {n : i-1 for i, n in enumerate(N)}
    edges = []
    G.add_nodes_from(vertices_dict.values())
    for n in N:
        if n.parent is None:
            continue
        edges.append((vertices_dict[n.parent], vertices_dict[n]))
    G.add_edges_from(edges)
    pos = nx.nx_agraph.graphviz_layout(G)
    nx.draw(G, pos, with_labels=True, 
                node_size=10, 
                font_size=5, font_color="black", 
                edge_color="gray", linewidths=1.5)
    plt.show()
    
main()