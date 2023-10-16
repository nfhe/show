# coding=utf-8

import seaborn as sns
sns.set()
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

rewards1 = np.array([0, 0.1,0,0.2,0.4,0.5,0.6,0.9,0.9,0.9])
rewards2 = np.array([0, 0,0.1,0.4,0.5,0.5,0.55,0.8,0.9,1])
rewards=np.vstack((rewards1,rewards2)) # 合并为二维数组
df = pd.DataFrame(rewards).melt(var_name='episode',value_name='reward')

sns.lineplot(x="episode", y="reward", data=df)
plt.show()
