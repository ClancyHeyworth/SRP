from run import run_node_model, run_paper_model
from util import ModelParams
import matplotlib.pyplot as plt

alphas = []
objs = []
params = ModelParams(20, 15, 2.0, 0 / 100)
for alpha in range(5, 99, 5):
    params.alpha = alpha / 100
    obj = run_paper_model(params)
    print(alpha / 100, obj)
    alphas.append(alpha / 100)
    objs.append(obj)
    
plt.scatter(alphas, objs, label = 'Paper Model')

node_obj = run_node_model(params)
plt.plot([0, 1], [node_obj, node_obj], color = 'r', label = 'Node Model')
plt.title('Comparison of Objective Value of Paper and Node MIP')
plt.xlabel('Alpha Value')
plt.ylabel('Objective Value')
plt.legend()
plt.show()