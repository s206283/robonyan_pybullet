import gym
import numpy as np
from PIL import Image
import torch
from gym import spaces
from collections import deque

class Observer():

    def __init__(self, env, args):
        self._env = env
        self.observation_type = args.observation_type

    @property
    def action_space(self):
        return self._env.action_space

    @property
    def observation_space(self):
        return self._env.observation_space

    def reset(self):
        if observation_type = "from_state":
            return self._env.reset()
        else:
            self._env.reset()
            return self.transform(self._env.render(mode='rgb_array').transpose((2, 0, 1)))

    def render(self):
        self.env.render(mode="human")

    def step(self, action):
        state, reward, done, info = sef._env.step(action)
        if self.observation_type = "from_state":
            return state, reward, done, info
        else:
            pixel = self.transform(self._env.render('rgb_array').transpose((2, 0, 1)))
            return pixel, reward, done, info

    def transform(self, pixel):
        pixel = Image.fromarray(pixel)
        resize = pixel.resize((32, 32))
        resize = np.array(resize).astype("float")
        normalized = resize / 255.0

        return normalized
