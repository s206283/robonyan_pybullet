import os, inspect
import numpy as np

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import gym
from pybullet_envs.bullet.robonyangraspGymEnv import RobonyanGraspGymEnv
from gym import spaces

# 環境の生成
env = RobonyanGraspGymEnv(renders=True, isDiscrete=False)
env.render(mode="human")
env.reset()

# ランダム行動
while not done and t < 50:
    # env.step(env.action_space.sample())
    action = env.action_space.sample()
    #action = np.array([-0.1] * 12)

    #物理モデル1ステップ---------------------------
    observation, reward, done, info = env.step(action)
    #print(observation.shape)
    #image = observation[0]
    #prox = observation[1]
    #image = image.transpose((2, 0, 1))
    #image = np.ascontiguousarray(image, dtype=np.float32) / 255
    #image = torch.from_numpy(image)
    #print(image.shape)
    #resize_img = resize(image).unsqueeze(0).to(device)
    #print(resize_img)
    #print(resize_img.shape)
    #print(prox.shape)
    #print(action)
    #print(observation)
    reward_total = reward_total + reward
    print(reward_total)

    #時間を進める----------------------------------
    t += 1
    # end while loop ------------------------------

print('reward=',reward_total)
