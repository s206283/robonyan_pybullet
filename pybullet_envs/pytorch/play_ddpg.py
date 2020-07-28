#!/usr/bin/env python3
import argparse
import gym
import pybullet_envs

import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import gym
from pybullet_envs.bullet.robonyanRepositioningGymEnv import RobonyanRepositioningGymEnv

import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from pandas import Series, DataFrame
from collections import namedtuple
from itertools import count
from PIL import Image
from tqdm import tqdm
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import datetime

writer = SummaryWriter()

env = RobonyanGymEnv(renders=True, isDiscrete=False)

save_path = os.path.join("saves", "ddpg")

from lib import model

import numpy as np
import torch

ENV_ID = "MinitaurBulletEnv-v0"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True, help="Model file to load")
    parser.add_argument("-e", "--env", default=ENV_ID, help="Environment name to use, default=" + ENV_ID)
    parser.add_argument("-r", "--record", help="If specified, sets the recording dir, default=Disabled")
    args = parser.parse_args()

    spec = gym.envs.registry.spec(args.env)
    spec._kwargs['render'] = False
    env = RobonyanRepositioningGymEnv(renders=True, isDiscrete=False)
    if args.record:
        env = gym.wrappers.Monitor(env, args.record)

    net = model.DDPGActor(env.observation_space.shape[0], env.action_space.shape[0])
    net.load_state_dict(torch.load(args.model))

    obs = env.reset()
    total_reward = 0.0
    total_steps = 0
    while True:
        obs_v = torch.FloatTensor([obs])
        mu_v = net(obs_v)
        action = mu_v.squeeze(dim=0).data.numpy()
        action = np.clip(action, -1, 1)
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        total_steps += 1
        if done:
            break
    print("In %d steps we got %.3f reward" % (total_steps, total_reward))
