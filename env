from __future__ import print_function
import copy
import tensorflow as tf
import numpy as np
import random



class env():
    def __init__(self):
        self.current_room=4
        self.step = 0
        self.total_reward = 0
        self.is_end = False
        self.state_num=6

    def interact(self, action):
        self.step+=1
        assert self.is_end is False
        mid = room_action_list[p[self.current_room]]

        if mid[action]==1:
           self.current_room=action


        if self.current_room == 5:
            reward= 10
            self.is_end = True
        else:
            reward =-1
        self.total_reward+=reward
        return  reward

    def current_room(self):
        return self.current_room
