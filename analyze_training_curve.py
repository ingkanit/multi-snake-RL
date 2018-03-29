# analyze_training_curve.py
# plot training curve of run

import numpy as np
import os
import re
import string
import argparse

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['image.aspect'] = 'auto'
mpl.rcParams['image.interpolation'] = 'none'

def smooth(data, winlen=30):
    w = np.ones(winlen)
    numtrials = data.shape[0]
    NT = data.shape[1]
    data_smooth = np.zeros((numtrials,NT))
    # extend data to avoid boundary effects
    add_start = int(np.ceil((winlen-1)*1.0/2))
    add_end = int(np.floor((winlen-1)*1.0/2))
    for i in range(numtrials):
#        data_smooth[i,:] = np.convolve(w/w.sum(),data[i,:]-np.mean(data[i,:]),mode='same')+np.mean(data[i,:])
        datathis = np.concatenate([np.mean(data[i,:winlen])*np.ones(add_start),
                                   data[i,:],
                                   np.mean(data[i,-winlen:])*np.ones(add_end)])
        data_smooth[i,:] = np.convolve(w/w.sum(),datathis,mode='valid')
    
    return data_smooth

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

    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--trainhistdir', help="Training directory of network",
                        default = 'train_single_snake_withhead_larger/')
    parser.add_argument('--savepng', action='store_true', default=False, help="Save figure as PNG image")
    args = parser.parse_args()
    # trainhistdir = 'train_single_snake_withhead_larger/'
    trainhistdir = args.trainhistdir
    # trainhistdir = 'train_single_snake_dqn/'
    # errstatpattern = r'(?:Reward_stat_)(\d+)(?:\.npz)'

    lastfilenumber = findlastepisode(trainhistdir)

    # load file
    stat = np.load('{}Reward_stat_{}.npz'.format(trainhistdir,lastfilenumber))['stat'][:lastfilenumber,:]
    
    # smooth reward with a sliding window
    smooth_reward_slide = smooth(stat[:,[1]].T, winlen=300)[0,:]
    # average reward over a period of 1000 episodes
    episode_avg_window = 1000
    if stat.shape[0] >= episode_avg_window:
        num_window_points = stat.shape[0]//episode_avg_window
        avg_reward = np.zeros((num_window_points,))
        for i in range(num_window_points):
            avg_reward[i] = np.mean(stat[i*episode_avg_window:(i+1)*episode_avg_window,1])
        
    mpl.rcParams.update({'font.size': 20})

    plt.ioff()
    
    # plot in matplotlib figure
    plt.figure()
    plt.clf()
    # plt.plot(np.arange(stat.shape[0]) + 1, stat[:,1],'.')
    plt.plot(np.arange(stat.shape[0]) + 1, smooth_reward_slide)
    if stat.shape[0] >= episode_avg_window:
        plt.plot(np.arange(num_window_points)*episode_avg_window, avg_reward, 'r')

    plt.xlabel('Episode')
    plt.ylabel('Reward (smoothed)')
    plt.title(trainhistdir[:-1], fontsize=15)
    
    plt.tight_layout()

    if args.savepng:
        plt.savefig(trainhistdir[:-1] + '.png')
    else:
        plt.show()

if __name__ == '__main__':
    main()


# raw_input('Press ENTER to continue')