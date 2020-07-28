'''
saves ~ 200 episodes generated from a random policy
'''

import os, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import gym
from pybullet_envs.bullet.robonyanRepositioningGymEnv import RobonyanRepositioningGymEnv
from gym import spaces

import numpy as np
import random

#from model import make_model

MAX_FRAMES = 500 # max length of carracing
MAX_TRIALS = 20 # just use this to extract one trial.

render_mode = False # for debugging.

DIR_NAME = 'record'
if not os.path.exists(DIR_NAME):
    os.makedirs(DIR_NAME)

#model = make_model(load_model=False)
env = RobonyanRepositioningGymEnv(renders=True, isDiscrete=False)

total_frames = 0
#model.make_env(render_mode=render_mode, full_episode=True)
for trial in range(MAX_TRIALS): # 200 trials per worker
  try:
    #random_generated_int = random.randint(0, 2**31-1)
    filename = DIR_NAME+"/"+str(trial + 1)+".npz"
    #recording_obs = []
    recording_img = []
    recording_oned_s = []
    recording_action = []
    recording_reward = []
    recording_done = []

    #model.reset()
    obs = env.reset() # pixels

    for frame in range(MAX_FRAMES):
      if render_mode:
        env.render("human")
      else:
        env.render("rgb_array")

      recording_img.append(obs[0])
      recording_oned_s.append(obs[1])

      action = env.action_space.sample()

      recording_action.append(action)
      obs, reward, done, info = env.step(action)

      recording_reward.append(reward)
      recording_done.append(done)

      if done:
        break

    total_frames += (frame+1)
    print("dead at", frame+1, "total recorded frames for this worker", total_frames)
    recording_img = np.array(recording_img, dtype=np.uint8)
    recording_oned_s = np.array(recording_oned_s, dtype=np.float16)
    recording_action = np.array(recording_action, dtype=np.float16)
    recording_reward = np.array(recording_reward, dtype=np.float16)
    recording_done = np.array(recording_done)

    np.savez_compressed(filename, img=recording_img,
                        oned_s=recording_oned_s,
                        action=recording_action,
                        reward=recording_reward,
                        done=recording_done)
  except gym.error.Error:
    print("stupid gym error, life goes on")
    #model.env.close()
    #model.make_env(render_mode=render_mode)
    continue
#model.env.close()
