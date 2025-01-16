from tqdm import tqdm 
from collections import defaultdict
import numpy as np
import random

def scenario_greater(w1 : tuple[int], w2 : tuple[int]):
    for i in range(len(w1)):
        if w1[i] < w2[i]:
            return False
    return True

def pertinent_filter(W : list[tuple[int]], num_days, w_probs : dict[tuple[int], float], alpha : float):
    INC = set()
    
    for s in tqdm(list(reversed(range(num_days)))):
        W_s = filter(lambda x : sum(x) == s, W)

        for w in W_s:
            comparable = filter(lambda w_dash : scenario_greater(w_dash, w), W)
            
            conf = np.sum([w_probs[w_dash] for w_dash in comparable])

            if conf > 1 - alpha:
                no_comparables = True
                for w_dash in INC:
                    if scenario_greater(w, w_dash) or scenario_greater(w_dash, w):
                        no_comparables = False
                        break
                if no_comparables:
                    INC.add(w)

    return INC

class Node:
    def __init__(self, rain : float, parent : "Node", 
                 stage : int, prob : float = 1.0):
        self.rain = rain
        self.parent = parent
        self.stage = stage
        self.prob = prob
    
    def __repr__(self):
        return f"Day {self.stage}, Rain {self.rain}"
    
def get_nodes(num_days:int, Probs : list[float]) \
        -> tuple[list[Node], defaultdict[Node, list[Node]], defaultdict[Node, list[float]]]:
    N = []
    children = defaultdict(list)
    children_cdfs = defaultdict(list)
    
    root_node = Node(None, None, -1)

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

    extend_tree(root_node)
    return N, children, children_cdfs

def sample(root_node : Node, num_days : int, children : defaultdict[Node, list[Node]], 
           children_cdfs : defaultdict[Node, list[float]]):
    """
    A function to sample a scenario from the tree.
    """
    sample = []
    node = root_node
    while node.stage < num_days - 1:
        # Get the child node according to the probability of each outcome (rain vs. no rain)
        sample_number = random.random() # random number between 0 and 1 used to sample the next day's weather
        child_index = next(idx for idx in range(len(children[node])) if children_cdfs[node][idx] >= sample_number)
        node = children[node][child_index]
        sample.append(node)
    return sample