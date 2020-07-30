import gym
import numpy as np
from PIL import Image
import torch
from gym import spaces
from collections import deque

class Observer():

    def __init__(self, env, frame_count, args):
        self._env = env
        self.frame_count = frame_count
        self._frames = deque([], maxlen=frame_count)
        self.observation_type = args.observation_type

    @property
    def action_space(self):
        return self._env.action_space

    @property
    def observation_space(self):
        return self._env.observation_space

    def reset(self):
        if self.observation_type == "from_state":
            return self._env.reset()
        else:
            self._env.reset()
            pixel = self.transform(self._env.render(mode='rgb_array'))
            for _ in range(self.frame_count):
                self._frames.append(pixel)
            return self._get_obs()

    def render(self):
        self.env.render(mode="human")

    def step(self, action):
        state, reward, done, info = self._env.step(action)
        if self.observation_type == "from_state":
            return state, reward, done, info
        else:
            pixel = self.transform(self._env.render(mode='rgb_array'))
            self._frames.append(pixel)
            return self._get_obs(), reward, done, info

    def transform(self, pixel):
        pixel = Image.fromarray(pixel)
        resize = pixel.resize((32, 32))
        resize = np.array(resize).astype("float")
        normalized = resize / 255.0
        feature = normalized.transpose((2, 0, 1))

        return feature

    def _get_obs(self):
        assert len(self._frames) == self.frame_count
        return np.concatenate(list(self._frames), axis=0)
