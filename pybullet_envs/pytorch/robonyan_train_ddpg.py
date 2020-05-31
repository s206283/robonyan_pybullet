#add parent dir to find package. Only needed for source code build, pip install doesn't need it.
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import gym
from pybullet_envs.bullet.robonyanGymEnv import RobonyanGymEnv

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
log_path = os.path.join("saves", "log")

os.makedirs(save_path, exist_ok=True)
os.makedirs(log_path, exist_ok=True)

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#リプレイメモリー
Transition = namedtuple('Transition',
                        ('img_state', 'state_1d', 'action', 'img_next_state', 'next_state_1d' , 'reward' ,'done'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    def reset(self):
        self.memory = []
        self.position = 0

class DDPGActor(nn.Module):
    def __init__(self, obs_size, act_size):
        super(DDPGActor, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_size, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, act_size),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)


class DDPGCritic(nn.Module):
    def __init__(self, obs_size, act_size):
        super(DDPGCritic, self).__init__()

        self.obs_net = nn.Sequential(
            nn.Linear(obs_size, 400),
            nn.ReLU(),
        )

        self.out_net = nn.Sequential(
            nn.Linear(400 + act_size, 300),
            nn.ReLU(),
            nn.Linear(300, 1)
        )

    def forward(self, x, a):
        obs = self.obs_net(x)
        return self.out_net(torch.cat([obs, a], dim=1))

class MultiActer(nn.Module):

    def __init__(self, img_h, img_w, oneD_size, act_size):

        super(MultiActer, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(img_w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(img_h)))
        conv_input_size = convw * convh * 32
        #conv_input_size = 19 * 37 * 32
        #self.head = nn.Linear(conv_input_size, outputs)

        self.full_connection = nn.Sequential(
            nn.Linear(in_features=conv_input_size+3, out_features=1024), # '+3' in_oneDの3ユニット分を追加
            nn.ReLU(),
            nn.Linear(1024, 300),
            nn.ReLU(),
            nn.Linear(300, act_size),
            nn.Tanh()
        )

    def forward(self, img, oneD):

        conv_input_size = 13 * 13 * 32

        # ひとつめのデータを用いて畳み込みの計算
        x = self.conv(img)

        # 畳み込み層からの出力を1次元化
        x = x.view(x.size(0), conv_input_size)

        # 1次元化した畳み込み層のからの出力と2つめの入力を結合
        x = torch.cat([x, oneD], dim=1)

        # 全結合層に入力して計算
        y = self.full_connection(x)

        return y

class MultiCritic(nn.Module):
    def __init__(self, img_h, img_w, oneD_size, act_size):

        super(MultiCritic, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(img_w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(img_h)))
        conv_input_size = convw * convh * 32
        #conv_input_size = 19 * 37 * 32
        #self.head = nn.Linear(conv_input_size, outputs)

        self.obs_net = nn.Sequential(
            nn.Linear(in_features=conv_input_size+oneDsize, out_features=1024), # '+3' in_oneDの3ユニット分を追加
            nn.ReLU(),
        )

        self.out_net = nn.Sequential(
            nn.Linear(1024 + act_size, 300),
            nn.ReLU(),
            nn.Linear(300, 1)
        )

    def forward(self, img, oneD, a):

        conv_input_size = 13 * 13 * 32

        # ひとつめのデータを用いて畳み込みの計算
        x = self.conv(img)

        # 畳み込み層からの出力を1次元化
        x = x.view(x.size(0), conv_input_size)

        # 1次元化した畳み込み層のからの出力と2つめの入力を結合
        x = torch.cat([x, oneD], dim=1)

        # 全結合層に入力して計算
        x = self.obs_net(x)

        # 行動を結合
        x = torch.cat([x, a], dim=1)

        # 全結合層に入力して計算
        y = self.out_net(x)

        return y

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    #batch作成
    img_state_batch = torch.cat(batch.img_state)
    state_1d_batch = torch.cat(batch.state_1d)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    img_next_state_batch = torch.cat(batch.img_next_state)
    next_state_1d_batch = torch.cat(batch.next_state_1d)
    done_batch = torch.cat(batch.done)

    reward_batch = reward_batch.unsqueeze(1)
    done_batch = done_batch.unsqueeze(1)

    #Qネットのloss関数計算
    next_actions = policy_target_net(img_next_state_batch, next_state_1d_batch)
    next_Q_values =q_target_net(img_next_state_batch, next_state_1d_batch, next_actions)
    expected_Q_values = (next_Q_values * GAMMA)*(1.0-done_batch) + reward_batch

    Q_values =  q_net(img_state_batch, next_state_1d_batch, action_batch)

    #Qネットのloss関数
    q_loss = F.mse_loss(Q_values,expected_Q_values)
    #writer.add_scalar('q_loss/train', q_loss, i_episode + 1)

    #Qネットの学習
    q_optimizer.zero_grad()
    #誤差逆伝搬
    q_loss.backward()
    #重み更新
    q_optimizer.step()

    #policyネットのloss関数
    actions = policy_net(img_state_batch, state_1d_batch)
    p_loss = -q_net(img_state_batch, state_1d_batch, actions).mean()
    #writer.add_scalar('p_loss/train', p_loss, i_episode + 1)

    #policyネットの学習
    p_optimizer.zero_grad()
    #誤差逆伝搬
    p_loss.backward()
    #重み更新
    p_optimizer.step()

    tau = 0.001
    #targetネットのソフトアップデート
    #学習の収束を安定させるためゆっくり学習するtarget netを作りloss関数の計算に使う。
    #学習後のネット重み×tau分を反映させ、ゆっくり追従させる。
    for target_param, local_param in zip(policy_target_net.parameters(), policy_net.parameters()):
      target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    for target_param, local_param in zip(q_target_net.parameters(), q_net.parameters()):
      target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


def save_model(self, env_name, suffix="", actor_path=None, critic_path=None):
    if not os.path.exists('models/'):
        os.makedirs('models/')

    if actor_path is None:
        actor_path = "models/ddpg_actor_{}_{}".format(env_name, suffix)
    if critic_path is None:
        critic_path = "models/ddpg_critic_{}_{}".format(env_name, suffix)
    print('Saving models to {} and {}'.format(actor_path, critic_path))
    torch.save(policy_net.state_dict(), actor_path)
    torch.save(q_net.state_dict(), critic_path)

def load_model(self, actor_path, critic_path):
    print('Loading models from {} and {}'.format(actor_path, critic_path))
    if actor_path is not None:
        policy_net.load_state_dict(torch.load(actor_path))
    if critic_path is not None:
        q_net.load_state_dict(torch.load(critic_path))

def impact_penalty(self, ram1, ram2, impact):
    r = 0
    for i in range(3):
        a = 1 / (1 + math.e** (ram1 * (impact[i] - ram2)))
        r += (1 - a)*impact[i]

    r = -r

env.render(mode="human")

#action,observationの要素数取得
n_actions = env.action_space.shape[0]
init_obs = env.reset()
#init_img = init_obs[0]
init_1d = init_obs[1]
#img_resize = init_img.resize((320, 180))
#img_height , img_width, _ = img_resize.shape
img_height = 128
img_width = 128
n_obs_1d = init_1d.shape

#print('num actions,observations',n_actions,init_img.shape,n_prox )


#ネットワーク
#policyネットとそのtargetネット
policy_net = MultiActer(img_height, img_width, n_obs_1d, n_actions).to(device)
policy_target_net = MultiActer(img_height, img_width, n_obs_1d, n_actions).to(device)
#Qネットとそのtargetネット
q_net = MultiCritic(img_height, img_width, n_obs_1d, n_actions).to(device)
q_target_net = MultiCritic(img_height, img_width, n_obs_1d, n_actions).to(device)
#学習用optimizer生成
p_optimizer = optim.Adam(policy_net.parameters(),lr=1e-3)
q_optimizer = optim.Adam(q_net.parameters(),lr=3e-3, weight_decay=0.0001)
#学習用リプレイメモリ生成
memory = ReplayMemory(10000)
memory.reset()
#ノイズの大きさ計算用、最初は大きくして学習が進んだら小さくするためのカウンタ
steps_done = 0

# 画像のリサイズ
resize = T.Compose([T.ToPILImage(),
                    T.Resize((128, 128), interpolation=Image.CUBIC),
                    T.ToTensor()])

#BATCH_SIZE = 128
BATCH_SIZE = 128
#Qネット学習時の報酬の割引率
GAMMA = 0.98
#steps_doneに対するノイズの減少係数
SIGMA_START = 1.0 #最初
SIGMA_END = 0.3 #最後
SIGMA_DECAY = 800 #この回数で約30%まで減衰

#学習数。num_epsode=200を変更する。
num_episodes = 1000

max_step = 500

rewards = []
#mean_rewards = []

num_epsodes_list = []
#total_numsteps = 0
updates = 0

best_reward = None

for i_episode in range(num_episodes):
    #whileループ内初期化
    observation = env.reset()
    img_state = observation[0]
    img_state = img_state.transpose((2, 0, 1))
    img_state = np.ascontiguousarray(img_state, dtype=np.float32) / 255
    tensor_img_state = torch.from_numpy(img_state)
    tensor_img_state = resize(tensor_img_state).unsqueeze(0).to(device)
    state_1d = observation[1]
    tensor_state_1d = torch.tensor(state_1d,device=device,dtype=torch.float)
    tensor_state_1d = torch.unsqueeze(tensor_state_1d, 0)

    done = False
    episode_reward = 0.0
    t = 0
    num_epsodes_list.append(i_episode + 1)

    noise =np.array([random.uniform(-0.5,0.5) for i in range(n_actions)],dtype = np.float)
    theta = 0.08
    sigma = SIGMA_END + (SIGMA_START - SIGMA_END) * math.exp(-1. * steps_done / SIGMA_DECAY)
    #steps_done += 1

    while not done and t < max_step:
        #指令値生成------------------------------------
        sample = random.random()

        #d = (1 - t/max_step)

        with torch.no_grad():
            action = policy_net(tensor_img_state, tensor_state_1d)
            action = action.cpu().data.numpy()

        #最後は純粋にネットワークのデータを取得するためノイズ無し-------------------
        if (i_episode != num_episodes -1 ):
          #OUNoise
            noise = noise - theta * noise + sigma * np.array([random.uniform(-1.0,1.0) for i in range(len(noise))])
            action += noise

        max_pos = 0.02
        max_rot = 0.1745
        pos = action[0][:3]
        rot = action[0][3:]
        clip_pos = np.clip(pos, -max_pos, max_pos)
        clip_rot = np.clip(rot, -max_rot, max_rot)
        action = np.concatenate([clip_pos, clip_rot], 0)
        #action = np.clip(action, -1, 1)

        #物理モデル1ステップ---------------------------
        next_observation, reward, done, info = env.step(action)
        img_next_state = next_observation[0]
        img_next_state = img_next_state.transpose((2, 0, 1))
        img_next_state = np.ascontiguousarray(img_state, dtype=np.float32) / 255
        next_state_1d = next_observation[1]
        #reward = reward * d
        # impact_compute
        impact_1 = max(0, (next_state_1d[3] - state_1d[3]))
        impact_2 = max(0, (next_state_1d[4] - state_1d[4]))
        impact_3 = max(0, (next_state_1d[5] - state_1d[5]))
        impact = [impact_1, impact_2, impact_3]
        impact_penalty = self.impact_penalty(2, 2, impact)

        #episode_reward = episode_reward + reward
        episode_reward = reward

        #学習用にメモリに保存--------------------------
        #tensor_observation = torch.tensor(observation,device=device,dtype=torch.float)
        tensor_action = torch.tensor(action,device=device,dtype=torch.float)
        tensor_acton = torch.unsqueeze(tensor_action, 0)
        tensor_img_next_state = torch.from_numpy(img_next_state)
        tensor_img_next_state = resize(tensor_img_next_state).unsqueeze(0).to(device)
        tensor_next_state_1d = torch.tensor(next_state_1d,device=device,dtype=torch.float)
        tensor_next_state_1d = torch.unsqueeze(tensor_next_state_1d, 0)
        #tensor_next_observation = torch.tensor(next_observation,device=device,dtype=torch.float)
        tensor_reward = torch.tensor([reward],device=device,dtype=torch.float)
        tensor_done =  torch.tensor([done],device=device,dtype=torch.float)
        memory.push(tensor_img_state, tensor_state_1d, tensor_action, tensor_img_next_state, tensor_next_state_1d, tensor_reward,tensor_done)

        #observation(state)更新------------------------
        #observation = next_observation
        tensor_img_state = tensor_img_next_state
        tensor_state_1d = tensor_next_state_1d
        #学習してpolicy_net,target_neを更新
        optimize_model()
        #データ保存------------------------------------
        #if i_episode == num_episodes -1 :
        #時間を進める----------------------------------

        t += 1
        # end while loop ------------------------------


    #rewards.append(episode_reward)
    writer.add_scalar('reward/train', episode_reward, i_episode + 1)
    rewards.append(episode_reward)

    if ((i_episode + 1) % 10 == 0) or (i_episode == num_episodes - 1):
        #state = torch.Tensor([env.reset()])
        """
        #test
        observation = env.reset()
        img_state = observation[0]
        img_state = img_state.transpose((2, 0, 1))
        img_state = np.ascontiguousarray(img_state, dtype=np.float32) / 255
        tensor_img_state = torch.from_numpy(img_state)
        tensor_img_state = resize(tensor_img_state).unsqueeze(0).to(device)
        state_1d = observation[1]
        tensor_state_1d = torch.tensor(state_1d,device=device,dtype=torch.float)
        tensor_state_1d = torch.unsqueeze(tensor_state, 0)
        episode_reward = 0
        t = 0
        while not done and t < max_step:
            #d = (1 - t/max_step)
            action = policy_net(tensor_img_state, tensor_prox_state)
            action = action.cpu().data.numpy()
            next_observation, reward, done, _ = env.step(action[0])
            #reward = reward * d
            img_next_state = next_observation[0]
            img_next_state = img_next_state.transpose((2, 0, 1))
            img_next_state = np.ascontiguousarray(img_next_state, dtype=np.float32) / 255
            next_state_1d = next_observation[1]
            tensor_img_next_state = torch.from_numpy(img_next_state)
            tensor_img_state = resize(tensor_img_next_state).unsqueeze(0).to(device)
            tensor_next_state_1d = torch.tensor(next_state_1d,device=device,dtype=torch.float)
            tensor_next_state_1d = torch.unsqueeze(tensor_next_state_1d, 0)
            episode_reward += reward

            #next_state = torch.Tensor([next_state])

            #state = next_state
            tensor_img_state = tensor_img_next_state
            tensor_state_1d = tensor_next_state_1d

            t += 1

            #if done:
                #break

        writer.add_scalar('reward/test', episode_reward, i_episode)
        """

        if best_reward is None or best_reward < episode_reward:
            if best_reward is not None:
                print("Best reward updated: %.3f -> %.3f" % (best_reward, episode_reward))
                name = "best_%+.3f_%d.pth" % (episode_reward, i_episode)
                fname = os.path.join(save_path, name)
                torch.save(policy_net.state_dict(), fname)
            best_reward = episode_reward

        #rewards.append(episode_reward)
        #mean_reward = np.mean(rewards[-10:])
        #mean_rewards.append(mean_reward)
        print("Episode: {}, total numsteps: {}, reward: {}, average reward: {}".format(i_episode, t, rewards[-1], np.mean(rewards[-10:])))

  # end for loop --------------------------------------

  # グラフ作成
  pit.plot(num_epsodes_list, rewards)
  plt.xlabel('num_episodes')
  plt.ylabel('reward')
  plt.show()

  data = {'reward':rewards, 'num_episodes':num_epsodes_list}
  df = Dataframe(data)

  path = os.path.join(log_path, "reward.csv")

  df.to_csv(path)
