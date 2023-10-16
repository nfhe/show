# coding=utf-8
import numpy as np
SEED = 1234
np.random.seed(SEED)

# # v = np.random.random()*0.4
# # w = (np.random.random()-0.5)*0.4

# # for i in range(20):
# #     v = np.random.random()*0.4
# #     w = (np.random.random()-0.5)*0.4
# #     print("v:",v,"w:",w)

# 		# if np.random.rand() < self.epsilon:
# 		# 	action = (self.max_action - self.min_action) * np.random.random(self.env.action_space.shape[0]) + self.min_action
# 		# else:
# 		# 	action = (
# 		# 		self.select_action(np.array(state)) + np.random.normal(0, self.max_action * self.expl_noise, size=self.action_dim)
# 		# 	).clip(-self.max_action, self.max_action)
# 		# return action

max_action = 0.8
min_action = -0.8
expl_noise = 0.1
# expl_noise_3 = 0.05
# action_dim = 2
# # action = (max_action - min_action) * np.random.random(2) + min_action
# # print("action:",action)

# # for i in range(20):
# #     action = np.random.random(2)*0.5
# #     print("action:",action)


# for i in range(20):
#     action = np.random.normal(0, max_action * expl_noise_3, size=action_dim)
#     # //生成2个随机数，范围是-0.25到0.25
#     # action = np.random.random(1)*0.5 - 0.25
#     w = (np.random.random()-0.5)*0.4
#     print("action:",action,"w:",w)


for i in range(20):
	# action = np.random.normal(0, 0.08, size=2)
	action = np.random.random(1)
	print("action:",action)
# a = np.random.normal(0, max_action * expl_noise, 2)