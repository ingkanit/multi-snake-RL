# train a single snake agent using the baseline PPO algorithm

import sys
from baselines import logger
# from baselines.ppo2 import ppo2
from ppo import ppo2
import tensorflow as tf
import gym
from baselines.a2c.utils import fc, conv, conv_to_fc
from baselines.common.distributions import make_pdtype
# from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
# from baselines.common.vec_env.vec_normalize import VecNormalize
from baselines import bench, logger
from multi_snake import MultiSnake
from handle_args import handle_args

import numpy as np
import subprocess as sp

import os
import re

# set environment variable to free gpu device
out = sp.check_output('./free_gpus.sh').split()[0]
if isinstance(out, bytes):
    out = out.decode()
os.environ["CUDA_VISIBLE_DEVICES"] = out

save_int = 1000
trainhistdir = 'train_single_snake_ppo2_gammap9_largebuffer/'
save_gif = False
test_model = True
statlist = []
stat = np.array([])
reloadlist = []
numepisodes = 0
lastnumepisodes = 0
maxepisodes = int(6e5)
max_episodes_timestep = 1000
tol_frac = .9
gamma = .9
lam = 1.0
buf_size = 2048
minibatch_size = 32
nminibatches = buf_size // minibatch_size

best_avg_reward = -1
min_reward = -1

giffn = trainhistdir + 'video.gif'

# handle command line arguments
test_model, save_gif = handle_args(test_model, save_gif)

if save_gif:
    maxepisodes = 3

# TO DO:
# - Implement saving of best parameter values
# - Change multi_env such that the last 4 observations are given

class SingleSnakeCNNPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False):
        ob_shape = (nbatch,) + ob_space.shape
        nact = ac_space.n
        # observation input
        X = tf.placeholder(tf.float32, ob_shape, name='Ob') #obs
        # scale input by maximum (=2 for single snake)
        Xscaled = X / 2
        with tf.variable_scope("model", reuse=reuse):
            activ = tf.nn.relu
            # policy network
            # 2 conv layers
            ph1 = activ(conv(Xscaled, 'pi_c1', nf=16, rf=3, stride=1, init_scale=np.sqrt(2)))
            ph2 = activ(conv(ph1, 'pi_c2', nf=32, rf=3, stride=1, init_scale=np.sqrt(2)))
            # convert to fully connected layer
            ph2 = conv_to_fc(ph2)
            # one fully connected hidden layer
            ph3 = activ(fc(ph2, 'pi_fc', nh=256, init_scale=np.sqrt(2)))
            # linear output layer to policy logits
            # initialize with small weights to ensure large initial policy entropy
            pi = fc(ph3, 'pi', nact, init_scale=0.01)
            # value network
            # 2 conv layers
            vh1 = activ(conv(Xscaled, 'vf_c1', nf=16, rf=3, stride=1, init_scale=np.sqrt(2)))
            vh2 = activ(conv(vh1, 'vf_c2', nf=32, rf=3, stride=1, init_scale=np.sqrt(2)))
            # convert to fully connected layer
            vh2 = conv_to_fc(vh2)
            # one fully connected hidden layer
            vh3 = activ(fc(vh2, 'vf_fc', nh=128, init_scale=np.sqrt(2)))
            # linear output of value function
            vf = fc(vh3, 'vf', 1)[:,0]

        self.pdtype = make_pdtype(ac_space)
        self.pd = self.pdtype.pdfromflat(pi)

        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = None

        def step(ob, *_args, **_kwargs):
            a, v, neglogp = sess.run([a0, vf, neglogp0], {X:ob})
            return a, v, self.initial_state, neglogp

        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X:ob})

        self.X = X
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value
        
def makegymwrapper(gyminstance, visualize=False):
# gym wrapper that accepts a one-element numpy array as argument of the step function
# Also: environment automatically resets itself after it is done
# And: step function returns arrays
    class WrapGym(object):
        def __init__(self, gyminstance, visualize=False):
            self.gym = gyminstance
            self.num_envs = 1
            self.action_space = self.gym.action_space
            self.observation_space = self.gym.observation_space
            self.reward_range = self.gym.reward_range
            self.l = 0
            self.reward = 0.0
            self.episodes = 0
            self.visualize = visualize
        def seed(self, seed=None):
            return self.gym.seed(seed=seed)
        def step(self, action):
            global statlist

            obs, reward, done, _ = self.gym.step(int(action))
            info = dict()
            self.l += 1
            self.reward += reward
            # force end of epoch if timesteps > max_episodes_timestep
            if self.l >= max_episodes_timestep:
                done = True
            if self.visualize:
                self.render()
            if done:
                self.reset()
                info = dict(episode=dict(l=self.l, r=self.reward))
                self.episodes += 1
                statlist.append([self.l,self.reward])
                # output during testing
                if test_model:
                    print("Finished episode {ep} after {ts} timesteps (reward: {reward})".format(ep=self.episodes, ts=self.l,
                                                                                 reward=self.reward))
                    
                    # force end of program if maximum number of episodes has been completed
                    if self.episodes >= maxepisodes:
                        if save_gif:
                            self.gym.write_gif(giffn)
                        sys.exit(0)                    
                self.l = 0
                self.reward = 0.0
            return obs, np.array([reward]), np.array([done]), [info]
        def reset(self):
            return self.gym.reset()
        def render(self, mode='human'):
            return self.gym.render(mode=mode)
        def close(self):
            return self.gym.close()

    return WrapGym(gyminstance, visualize=visualize)

def findlastepisode(traindir):
    # list directory contents
    errstatfiles = os.listdir(traindir)
    
    # find files in directory that look like reward statistics files and extract their numbers
    errstatnumbers = []
    for x in errstatfiles:
        matchobj = re.match(r'(?:Reward_stat_)(\d+)(?:\.npz)',x)
        if matchobj is not None:
            errstatnumbers.append(int(matchobj.group(1)))
        
    # find highest number
    lastfilenumber = np.array(errstatnumbers).max().astype(int)
    
    return lastfilenumber    
    
def on_update(model, update, epinfobuf):
    global stat
    global lastnumepisodes
    global best_avg_reward
    global reloadlist

    stat = np.asarray(statlist)
    numepisodes = stat.shape[0]

    if not test_model and (numepisodes - lastnumepisodes >= save_int):
        # calculate average reward over ~save_int episodes
        new_avg_reward = np.mean(stat[lastnumepisodes:numepisodes,1])
        if new_avg_reward >= best_avg_reward:
            # save model weights
            model.save("{}Params_{}.npz".format(trainhistdir, numepisodes))
            print("Model saved.")
        elif (new_avg_reward - min_reward) < tol_frac * (best_avg_reward - min_reward):
            # avg reward unacceptably bad: restore model weights
            # reload model weights
            lastepisode = findlastepisode(trainhistdir)
            model.load("{}Params_{}.npz".format(trainhistdir, lastepisode))
            print("New avg reward {:.3f} worse than previous avg reward {:.3f}".format(new_avg_reward, best_avg_reward))
            print("Reloading weights.")
            reloadlist.append(numepisodes)
        else:
            print("No reward record, but let's keep going for now.")    
            
        np.savez("{}Reward_stat_{}.npz".format(trainhistdir, numepisodes),stat=stat,reloadlist=np.asarray(reloadlist).astype(int))
        print("Statistics saved")
        lastnumepisodes = numepisodes
        
    if test_model and (numepisodes == 0):
        lastepisode = findlastepisode(trainhistdir)
        model.load("{}Params_{}.npz".format(trainhistdir, lastepisode))
        print("Model loaded.")
        
    if numepisodes >= maxepisodes:
        return False
    else:
        return True


def main():
    tf.Session().__enter__()

    seed = 1
    
    np.random.seed(seed)
    tf.set_random_seed(seed)
    
    # disable logging during testing
    if test_model:
        log_interval = int(1e20)
    else:
        log_interval = 1
    
    # env = gym.make('CartPole-v0')
    # Create Multi-Snake environment
    num_agents = 1
    spacing = 22
    grid_dim = 10
    history = 4
    env = MultiSnake(num_agents=num_agents, num_fruits=3, spacing=spacing, grid_dim=grid_dim, flatten_states=False,
                    reward_killed=-1.0,history=4, save_gif=save_gif)

    env.reset()

    env = makegymwrapper(env, visualize=test_model)
    ppo2.learn(policy=SingleSnakeCNNPolicy, env=env, nsteps=buf_size, nminibatches=nminibatches,
        lam=lam, gamma=gamma, noptepochs=4, log_interval=log_interval,
        ent_coef=.01,
        lr=1e-3,
        cliprange=.2,
        total_timesteps=int(1e20),
        callback_fn=on_update)
    


if __name__ == '__main__':
    main()

