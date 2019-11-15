import numpy as np
import random
from env import env
import time

EPSILON = 0.1
ALPHA = 0.1
GAMMA = 0.9
MAX_STEP = 100

#===========================================================
def epsilon_greedy(Q, state):#epsilon 贪婪策略
    #在基于当前的 Q 值估计得出的状态 s 下选择一个动作 a
    #np.random.uniform()返回0-1之间的随机数
    if (np.random.uniform() > 1 - EPSILON) or ((Q[state, :] == 0).all()):#如果这个数大于 epsilon，或者此
                                                                         #state每列都有q值的话，那么我们将会进行「利用」
        M = np.random.randint(6)  #  0~5随机整数（探索）
    else:
        M = Q[state, :].argmax()#返回Q值最大的（利用）
    return M
#===========================================================
e = env()
Q = np.zeros((e.state_num, e.state_num))#step 1：初始化Q矩阵
print('now lets start')
for i in range(100):#step 2：小于最大步长时（或者直到训练被中止前），步骤 3 到步骤 5 会一直被重复。
    e = env()
    GAGA = [e.current_room]
    while (e.is_end is False) and (e.step < MAX_STEP):
        action = epsilon_greedy(Q, e.current_room)#step 3:选取下一个动作 a
        state = e.current_room
        reward = e.interact(action) #step 4：跟新当前的state与跟新当前的reward
        new_state = e.current_room
        GAGA.append(e.current_room)
        Q[state, action] = (1 - ALPHA) * Q[state, action] + \
            ALPHA * (reward + GAMMA * Q[new_state, :].max())#step 5：Bellman 方程跟新当前的q值
    print('Episode:', i, 'Total Step:', e.step, 'Total Reward:', e.total_reward)
    print('the route is'+str(GAGA))
print(Q)#把每个状态的q值输出
print('the BEST route is'+str(GAGA))
