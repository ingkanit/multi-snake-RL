# extend tensorforce wrapper for OpenAIGym to custom environments

import gym
import numpy as np
from tensorforce import TensorForceError
from tensorforce.contrib.openai_gym import OpenAIGym

class OpenAIGym_custom(OpenAIGym):
# This custom class wraps an unregistered Open AI gym
# also supports a custom multi-agent version of Open AI gyms
    def __init__(self, custom_gym, gym_id=None, monitor=None, monitor_safe=False, monitor_video=0, visualize=False):
        if gym_id is None:
            gym_id = 'Custom OpenAIGym'
        
        self.gym_id = gym_id
        self.gym = custom_gym
        self.visualize = visualize
        
        if monitor:
            if monitor_video == 0:
                video_callable = False
            else:
                video_callable = (lambda x: x % monitor_video == 0)
            self.gym = gym.wrappers.Monitor(self.gym, monitor, force=not monitor_safe, video_callable=video_callable)
    
    def get_num_agents(self):
        return self.gym.get_num_agents()
    
    def get_active_agents(self):
        return self.gym.get_active_agents()