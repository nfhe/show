# coding=utf-8

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns;
sns.set() # 因为sns.set()一般不用改，可以在导入模块时顺便设置好

rewards = np.array([0, 0.1,0,0.2,0.4,0.5,0.6,0.9,0.9,0.9])
sns.lineplot(x=range(len(rewards)),y=rewards)
# sns.relplot(x=range(len(rewards)),y=rewards,kind="line") # 与上面一行等价
plt.xlabel("episode")
plt.ylabel("reward")
plt.title("data")
plt.show()
