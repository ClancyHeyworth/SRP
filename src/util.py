from tqdm import tqdm 
from collections import defaultdict
import numpy as np
import random

def scenario_greater(w1 : tuple[int], w2 : tuple[int]):
    for i in range(len(w1)):
        if w1[i] < w2[i]:
            return False
    return True

def pertinent_filter(W : list[tuple[int]], num_days : int, w_probs : dict[tuple[int], float], alpha : float):
    INC = set()
    W_s = defaultdict(set)
    
    for w in W:
        W_s[np.sum(w)].add(w)
    
    for s in tqdm(range(num_days-1, -1, -1)):
        for w in W_s[s]:
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

def scenario_greater2(w1: np.ndarray, w2: np.ndarray) -> bool:
    return np.all(w1 >= w2)

def pertinent_filter2(W : np.ndarray, num_days, day_probs, alpha : float):
    
    INC = set()
    W_s = defaultdict(list)
    print(np.unique([w.shape for w in W]))
    for w in W:
        W_s[np.sum(w)].append(w)
    
    for s in tqdm(range(num_days - 1, -1, -1)):

        for w in W_s[s]:
            conf = 0
            # comparable = filter(lambda w_dash : scenario_greater(w_dash, w), W)
            # for w_dash in comparable:
            #     conf += np.dot(w_dash, day_probs)
            comparable_mask = np.array([scenario_greater(w_dash, w) for w_dash in W])
            comparable = W[comparable_mask]

            # Using np.einsum for dot products
            conf += np.einsum('ij,j->i', comparable, day_probs).sum()
                # if len(conf) > 15:
                #     print('\nyoooo')
                #     print(w_dash.shape, w.shape)
                #     print(np.unique([w.shape for w in comparable]))
            try:
                if conf > 1 - alpha:
                    no_comparables = True
                    for w_dash in INC:
                        if scenario_greater2(w, w_dash) or scenario_greater2(w_dash, w):
                            no_comparables = False
                            break
                    if no_comparables:
                        INC.add(tuple(w))
            except:
                print('conf', conf)
                print('w', w)
                print(np.shape(w), np.shape(conf))
                quit()

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