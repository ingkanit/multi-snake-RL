# Implementation of a Multi-Snake OpenAI-like gym environment
#
# Written by Ingmar Kanitscheider
#
# Inspired by code from https://github.com/bhairavmehta95/slitherin-gym 


import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
from copy import deepcopy

from pygame.locals import *
import pygame
import time
from copy import deepcopy
from collections import deque


# constants defining observation space
DEAD_OBS = -1
FREE_OBS = 0
FRUIT_OBS = 1
SELF_AGENT_OBS = 2
SELF_AGENT_HEAD_OBS = 2
OTHER_AGENT_OBS = 3
OTHER_AGENT_HEAD_OBS = 3

PRINT_DEBUG = 0

def prt_debug(x):
    if PRINT_DEBUG:
        print(x)


class MultiSnake(gym.Env):
    AGENT_COLORS = [
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
    ]

    class Agent:
        def __init__(self, snakeenv, x, y, spacing = 22, length=3, direction=0, max_buffer_length=2000):
            self.snakeenv = snakeenv
            self.init_length = length
            self.length = length
            self.max_buffer_length = max_buffer_length
            self.spacing = spacing
            self._reset(x, y, direction)


        def _reset(self, x, y, direction):
            self.length = self.init_length

            self.direction = direction

            self.x = -1 + np.zeros((self.max_buffer_length,)).astype(int)
            self.y = -1 + np.zeros((self.max_buffer_length,)).astype(int)
            
            # initial positions, no collision.
            if self.direction == 0:
                self.x[:self.length] = x - np.arange(self.length).astype(int)
                self.y[:self.length] = y

            if self.direction == 1:
                self.x[:self.length] = x + np.arange(self.length).astype(int)
                self.y[:self.length] = y

            if self.direction == 2:
                self.y[:self.length] = y + np.arange(self.length).astype(int)
                self.x[:self.length] = x

            if self.direction == 3:
                self.y[:self.length] = y - np.arange(self.length).astype(int)
                self.x[:self.length] = x


        def _act(self, action):
            if action == 0:
                if self.direction == 1:
                    return

                self.direction = 0

            elif action == 1:
                if self.direction == 0:
                    return

                self.direction = 1

            elif action == 2:
                if self.direction == 3:
                    return

                self.direction = 2

            elif action == 3:
                if self.direction == 2:
                    return

                self.direction = 3

            else:
                # continue direction
                pass
                
        def _remove_tail(self):
            # set last array element to -1
            self.x[self.length] = -1
            self.y[self.length] = -1


        def _update(self):

            # update previous positions
            for i in range(self.length, 0, -1):
                self.x[i] = self.x[i-1]
                self.y[i] = self.y[i-1]

            # update position of head of snake
            if self.direction == 0:
                self.x[0] += 1
            if self.direction == 1:
                self.x[0] -= 1
            if self.direction == 2:
                self.y[0] -= 1
            if self.direction == 3:
                self.y[0] += 1
            

        def _draw(self, surface, image, image_head):
            for i in range(0,self.length):
                if i==0:
                    # draw head
                    surface.blit(image_head, ((self.x[i]+1) * self.spacing, (self.y[i]+1)*self.spacing))
                else:
                    # draw tail
                    surface.blit(image, ((self.x[i]+1) * self.spacing, (self.y[i]+1)*self.spacing))

    def __init__(self, num_agents=1, num_fruits=3, grid_dim=4, spacing=22, init_length=3, reward_fruit = 1.0,
                                            reward_killed = 0.0, reward_finished = 0.0, flatten_states=True,
                                            history=1, save_gif=False):

        self._display_surf = None
        self._image_surf = None
        self._fruit_surf = None

        self.agents = []
        self.fruits = []
        self.num_agents = num_agents
        self.active_agents = num_agents
        self.num_fruits = num_fruits
        self.num_active_fruits = num_fruits
        self.init_length = init_length
        self.reward_fruit = reward_fruit
        self.reward_killed = reward_killed
        self.reward_finished = reward_finished
        self.flatten_states = flatten_states
        self.history = history
        self.save_gif = save_gif

        # self.window_dimension = window_dimension
        self.window_dimension = (grid_dim + 2)*spacing
        self.grid_dim = grid_dim
        self.spacing = spacing
        
        # initialize global map that keeps track of collision control
        # 0: free, 1: fruit, i>=2: agent i-2
        self.coll_map = np.zeros((grid_dim,grid_dim)).astype(int)
        
        for i in range(self.num_agents):
            agent = self._create_agent(i, self.init_length, create_object=True)
            self.agents.append(agent)

        self.killed = [False] * self.num_agents

        # Initialize goals
        for f in range(self.num_active_fruits):
            self.fruits.append(self._generate_goal())
    
        
        # return observation and action specs from the viewpoint of a single agent
        if self.flatten_states:
            self.observation_space = spaces.Box(low=-1, high=3, shape=(self.grid_dim**2*self.history,))
        else:
            self.observation_space = spaces.Box(low=-1, high=3, shape=(self.grid_dim, self.grid_dim, self.history))
        
        self.action_space = spaces.Discrete(4)

        min_reward = np.min([reward_fruit, reward_killed, reward_finished])
        max_reward = np.max([reward_fruit, reward_killed, reward_finished])
        self.reward_range = (min_reward, max_reward)

        self.window_init = False
                
        # initialize buffer for observations for every agent
        self.obs_buffer = [deque(maxlen=self.history)] * self.num_agents
        
        # initialize array of screen images if saving to gif
        if self.save_gif:
            self.gifbuffer = []


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def step(self, actions):
        # convert actions parameter to list if necessary
        if self.num_agents == 1 and not isinstance(actions, list):
            actions = [actions]
        
        new_obs = []
        killed_on_step = [False] * self.num_agents
        rewards = [0.0] * self.num_agents

        for i, p in enumerate(self.agents):
            # agent already killed?
            if self.killed[i]: continue
            
            p._act(actions[i])
            p._update()

            # agent encounters fruit?
            # only need to check head of snake
            found_fruit = False
            for f_i, f in enumerate(self.fruits):
                if f[0] == p.x[0] and f[1] == p.y[0]:
                    new_fruitpos = self._generate_goal()
                    if new_fruitpos is not None:
                        self.fruits[f_i] = new_fruitpos
                    else:
                        del self.fruits[f_i]
                    p.length += 1
                    rewards[i] = self.reward_fruit
                    found_fruit = True
            
            if not found_fruit:
                # remove tail from snake environment collision control
                self.coll_map[p.x[p.length],p.y[p.length]] = 0
                p._remove_tail()
            
            # does snake hit a wall?
            if p.x[0] < 0 or p.y[0] < 0 or p.x[0] >= self.grid_dim or p.y[0] >= self.grid_dim:
                killed_on_step[i] = True
                self._remove_agent(i)
                continue
            
            # does snake hit itself or another snake?
            if self.coll_map[p.x[0],p.y[0]] >= 2:
                agent_i = self.coll_map[p.x[0],p.y[0]] - 2
                if agent_i == i:
                    # snake hit itself
                    killed_on_step[i] = True
                    self._remove_agent(i)
                else:
                    # ipdb.set_trace()
                    # snake hit another snake
                    killed_on_step[i] = True
                    killed_on_step[agent_i] = True
                    self._remove_agent(i)
                    self._remove_agent(agent_i)
                continue
            
            # if snake has filled all grid cells, the game is won
            if p.length == self.grid_dim**2:
                rewards[i] += self.reward_finished
                self.active_agents -= 1
                self._remove_agent(i)
                prt_debug("Game finished!")
            
            # mark new snake head position on collision map
            if not killed_on_step[i]:
                self.coll_map[p.x[0],p.y[0]] = i + 2

        for i, k in enumerate(killed_on_step):
            if k:
                rewards[i] = self.reward_killed
                self.active_agents -= 1
                self.killed[i] = True

        done = False
        if self.active_agents <= 0:
            done = True
        
        for i in range(self.num_agents):
            ob = self._generate_obs(i)
            new_obs.append(ob)
            
        if self.num_agents == 1:
            new_obs = new_obs[0]
            rewards = rewards[0]

        return deepcopy(new_obs), deepcopy(rewards), done, {}


    def render(self, mode='human', wait=.15):
        if not self.window_init:
            self._pygame_init()
    
        self._draw_env()

        for i, f in enumerate(self.fruits):
            self._pygame_draw(self._display_surf, self._fruit_surf, f)

        for i, p in enumerate(self.agents):
            if self.killed[i]: continue
            p._draw(self._display_surf, self._agent_surfs[p.color_i], self._agent_surfs_head[p.color_i])

        pygame.display.flip()
        
        # save to gif if desired
        if self.save_gif:
            self.gifbuffer.append(pygame.surfarray.array3d(self._display_surf))
        
        time.sleep(wait)


    def reset(self):
        prt_debug("Reset:")
        
        # clear observation buffers
        [ob.clear() for ob in self.obs_buffer]
        
        # reset collision map
        self.coll_map = np.zeros((self.grid_dim,self.grid_dim)).astype(int)
        
        for i, p in enumerate(self.agents):
            self.killed[i] = False
            
            # prt_debug("Agent {}: x: {}, y: {}".format(i, x, y))
            self._create_agent(i, self.init_length, create_object=False)

        self.num_active_fruits = self.num_fruits
        self.fruits = []
        for f in range(self.num_active_fruits):
            self.fruits.append(self._generate_goal())

        self.active_agents = self.num_agents
        
        # reset returns observation of current state
        new_obs = []
        for i in range(self.num_agents):
            ob = self._generate_obs(i)
            new_obs.append(ob)
            
        if self.num_agents == 1:
            new_obs = new_obs[0]
        
        return deepcopy(new_obs)


    def close(self):
        pygame.quit()
        
    def get_active_agents(self):
        return [not self.killed[i] for i in range(self.num_agents)]
        
    def get_num_agents(self):
        return self.num_agents
        
    def write_gif(self, giffilename, duration = .15):
        import imageio
        imageio.mimwrite(giffilename, self.gifbuffer, format='GIF', duration = duration)
        
    
    def _create_agent(self, i, init_length, init_pose=None, create_object=True):
        # ipdb.set_trace()
        
        if init_pose is None:
            x, y, direction, dx, dy = self._sample_agent_position(init_length)
        else:
            x = init_pose['x']
            y = init_pose['y']
            direction = init_pose['direction']
            dx_array = [-1,1,0,0]
            dy_array = [0,0,1,-1]
            dx = dx_array[direction]
            dy = dy_array[direction]
            
            isfree = True
            for j in range(init_length):
                if self.coll_map[x+j*dx,y+j*dy] != 0:
                    isfree = False
            
            assert(isfree == True)
                
        if create_object:        
            agent = self.Agent(self, x, y, self.spacing, direction=direction, length=init_length, max_buffer_length = self.grid_dim **2)
            agent.color_i = i % len(self.AGENT_COLORS)
        else:
            agent = self.agents[i]
            agent._reset(x,y,direction)
        
        # mark occupied grid cells with index: agent_index+2
        for j in range(init_length):
            self.coll_map[x+j*dx,y+j*dy] = i+2
        
        if create_object:
            return deepcopy(agent)
            


    def _draw_env(self):
        self._display_surf.fill((0,0,0))

        for i in range(0, self.window_dimension, self.spacing):
            self._display_surf.blit(self._wall_surf, (0, i))
            self._display_surf.blit(self._wall_surf, (self.window_dimension - self.spacing, i))

        for i in range(0, self.window_dimension, self.spacing):
            self._display_surf.blit(self._wall_surf, (i, 0))
            self._display_surf.blit(self._wall_surf, (i, self.window_dimension - self.spacing))


    def _generate_goal(self):
        # find unoccupied grid cell for new goal
        free_posx, free_posy = np.where(self.coll_map == 0)
        
        if len(free_posx) == 0:
            prt_debug("Warning: No free space for fruit anymore")
            
            return None
        
        # pick random index among free grid cells
        rand_ind = np.random.randint(len(free_posx))
        x = free_posx[rand_ind]
        y = free_posy[rand_ind]
        
        # mark down fruit on collision map
        self.coll_map[x,y] = 1
        
        return [x,y]


    def _generate_obs(self, agent):
        
        if self.killed[agent]: 
            obs = DEAD_OBS + np.zeros((self.grid_dim, self.grid_dim))
        else:
            # generate current observation
            obs = np.zeros((self.grid_dim, self.grid_dim)) + FREE_OBS
            for i in range(self.agents[agent].length):
                obs[self.agents[agent].x[i]][self.agents[agent].y[i]] = SELF_AGENT_OBS

            for i, p in enumerate(self.agents):
                if self.killed[i] or i == agent: continue
                for j in range(p.length):
                    obs[p.x[j]][p.y[j]] = OTHER_AGENT_OBS

            for i, f in enumerate(self.fruits):
                obs[f[0]][f[1]] = FRUIT_OBS
        
        if self.flatten_states:
            obs = obs.flatten()
        else:
            obs = obs[:,:,None]
            
        # check observation buffer for this agent
        # if length of buffer smaller than required history - 1 (e.g. at the beginning of the episode)
        # fill with zeros
        for _ in range(self.history - len(self.obs_buffer[agent]) - 1):
            self.obs_buffer[agent].append(np.zeros_like(obs))
        
        # add current observation to buffer
        self.obs_buffer[agent].append(obs)
        
        # convert observation buffer to numpy array
        if self.flatten_states:
            obs_total = np.asarray(self.obs_buffer[agent]).flatten()
        else:
            # concatenate numpy arrays along 3rd axis
            obs_total = np.concatenate(self.obs_buffer[agent], axis=2)
        
        return deepcopy(obs_total)


    def _pygame_draw(self, surface, image, pos):
        surface.blit(image, ((pos[0]+1)*self.spacing, (pos[1]+1)*self.spacing))


    def _pygame_init(self):
        pygame.init()
        self._display_surf = pygame.display.set_mode((self.window_dimension, self.window_dimension), pygame.HWSURFACE)
        self._agent_surfs = []
        self._agent_surfs_head = []
        # self._running = True

        for i, p in enumerate(self.agents):
            image_surf = pygame.Surface([self.spacing - 4, self.spacing - 4])
            image_surf.fill(self.AGENT_COLORS[i % len(self.AGENT_COLORS)])
            self._agent_surfs.append(image_surf)
            
            # draw the head darker
            image_surf_head = pygame.Surface([self.spacing - 4, self.spacing - 4])
            image_surf_head.fill(tuple([c//2 for c in self.AGENT_COLORS[i % len(self.AGENT_COLORS)]]))
            self._agent_surfs_head.append(image_surf_head)

        self._fruit_surf = pygame.Surface([self.spacing - 4, self.spacing - 4])
        self._fruit_surf.fill((255, 0, 0))

        self._wall_surf = pygame.Surface([self.spacing - 4, self.spacing - 4])
        self._wall_surf.fill((255, 255, 255))
    
    def _remove_agent(self, i):
        # remove agent (except head) from collision map
        self.coll_map[self.coll_map == i+2] = 0

    
    def _sample_agent_position(self, init_length):
        # Agents positions are sampled by rejection sampling: keep sampling until there is free space
        found_freepos = False
        
        while not found_freepos:
        
            # first sample a direction
            direction = np.random.randint(0,4)
            
            # this determines the relative coordinates of the tail
            # e.g.: direction = 0 (E), tail in W direction, dx = -1, dy = 0
            dx_array = [-1,1,0,0]
            dy_array = [0,0,1,-1]
            
            # sample x,y coordinates such that the tail does not overlap with the wall
            minx = np.maximum(0,-(init_length-1)*dx_array[direction])
            maxx = self.grid_dim  + np.minimum(0,-(init_length-1)*dx_array[direction])
            miny = np.maximum(0,-(init_length-1)*dy_array[direction])
            maxy = self.grid_dim  + np.minimum(0,-(init_length-1)*dy_array[direction])
            
            
            # sample agent position
            x = np.random.randint(minx,maxx)
            y = np.random.randint(miny,maxy)
            
            # now we still have to check whether this agent collides with any other agent
            isfree = True
            for i in range(init_length):
                if self.coll_map[x+i*dx_array[direction],y+i*dy_array[direction]] != 0:
                    isfree = False
            
            if isfree:
                found_freepos = True
                
        return x,y,direction,dx_array[direction],dy_array[direction]
