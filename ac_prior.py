# simple boundary- and other agent avoiding strategy that tests interface
# 
# the strategy cheats: direct access to the coordinates of the agent head, 
# but could be equally implemented using observation + history

import numpy as np
from pygame.locals import *
import pygame

from multi_snake import MultiSnake
import time

import six

# import ipdb

def viable_actions(env, agent):
    if env.killed[agent]:
        # if agent is already dead there is no restriction
        return np.ones((4,)).astype(int)
    # get agent
    p = env.agents[agent]
    # determine allowed actions
    allowed_action = np.ones((4,)).astype(int)
    # make sure not to overstep boundary
    if p.x[0] == 0:
        allowed_action[1] = 0
    if p.x[0] == env.grid_dim -1:
        allowed_action[0] = 0
    if p.y[0] == 0:
        allowed_action[2] = 0
    if p.y[0] == env.grid_dim - 1:
        allowed_action[3] = 0
    
    # make sure not to step back
    if p.x[0] == p.x[1]:
        # last action was either 2 or 3
        if p.y[0] < p.y[1]:
            # last action was 2
            allowed_action[3] = 0
        else:
            # last action was 3
            allowed_action[2] = 0
    else:
        # last action was either 0 or 1
        if p.x[0] < p.x[1]:
            # last action was 1
            allowed_action[0] = 0
        else:
            allowed_action[1] = 0
    
    # make sure not to intersect itself or other agents
    # mark spaces of other agents
    isblocked = np.zeros((env.grid_dim, env.grid_dim)).astype(int)
    for i,pother in enumerate(env.agents):
        if env.killed[i]:
            continue
        for j in range(pother.length):
            isblocked[pother.x[j],pother.y[j]] = 1
        # if its another agent include safety margin around its head
        if i != agent:
            def clip_bdry(coord):
                return np.clip(coord,0,env.grid_dim-1)
            isblocked[clip_bdry(pother.x[0]+1),pother.y[0]] = 1
            isblocked[clip_bdry(pother.x[0]-1),pother.y[0]] = 1
            isblocked[pother.x[0],clip_bdry(pother.y[0]+1)] = 1
            isblocked[pother.x[0],clip_bdry(pother.y[0]-1)] = 1
    
    # disallow actions that could lead to collision
    if allowed_action[0] and isblocked[p.x[0]+1,p.y[0]]:
        allowed_action[0] = 0
    
    if allowed_action[1] and isblocked[p.x[0]-1,p.y[0]]:
        allowed_action[1] = 0
    
    if allowed_action[2] and isblocked[p.x[0],p.y[0]-1]:
        allowed_action[2] = 0
        
    if allowed_action[3] and isblocked[p.x[0],p.y[0]+1]:
        allowed_action[3] = 0
    
    # if no actions are viable, don't make any restrictions
    if not np.any(allowed_action):
        allowed_action = np.ones((4,)).astype(int)
    
    return allowed_action
    


