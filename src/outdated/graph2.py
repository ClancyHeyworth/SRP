
# Lambda 10% - 100%, cap = 3, window = 10, average of 100 samples
data = [0.9034322409234069, 0.8702060092296071, 0.8374470157127919, 0.8031300221545503, 0.7799861267051501, 0.7267797891803786, 0.6662281925756873, 0.6439517407366179, 0.5948509656244311, 0.5636009656244311]

import matplotlib.pyplot as plt

plt.plot([i / 100 for i in range(10, 101, 10)], data)
plt.xlabel('Lambda Value')
plt.ylabel('Mean Jaccard Score Compared to Lambda = 0')
plt.title('Effect of Lambda Value on First-Day Decision\nAveraged Over 100 Samples, For 10-Day Window')
import numpy as np
plt.xticks(np.arange(0.1, 1.1, 0.1))
plt.show()