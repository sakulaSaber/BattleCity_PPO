#运行该主程序，首先停5秒，然后运行screen_start.py文件，即开始游戏
#然后继续停3秒，之后就可以有agent来操作
# -*- coding: utf-8 -*-

import cv2
import pandas as pd
from getkeys import key_check
import directkey

import time
import keyboard as kb
import numpy as np
from pynput.keyboard import Controller, Key, Listener
from script.screen_start_pause import start_game
#from script.screen_mytank import mytank
from script.screen_num_rec import num_rec
from script.screen_test import grab_screen

from utils import pre_processing
import argparse
import os
from torch.utils.data import   BatchSampler, SubsetRandomSampler
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

window_enemy = (959,369,1055,385)# 敌军数量窗口
window_self = (767,653,795,685)# 我方坦克数量窗口(959,384,1055,405)
window_global = (209,335,727,853)# 全局窗口
window_base = (462,826,475,853)# 基地窗口
WIDTH = 264
HEIGHT = 264


#输入状态画面
class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()
        # 网络层定义
        self.conv1 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2)
        self.fc1 =  nn.Sequential(nn.Linear((WIDTH // 4) * (HEIGHT // 4) * 32, 512), nn.Tanh())
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 6)
        
        # 激活函数和池化层
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Dropout
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.max_pool(x)
        x = self.relu(self.conv2(x))
        x = self.max_pool(x)        
        x = self.relu(self.conv3(x))
        x = self.max_pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        return nn.functional.softmax(self.fc3(x), dim=1)


class ValueNet(nn.Module):
    def __init__(self):
        super(ValueNet, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Linear((WIDTH // 4) * (HEIGHT // 4) * 32, 512),nn.Tanh(),
            nn.Dropout(0.5),
            nn.Linear(512, 256), 
            nn.Dropout(0.5),
            nn.Linear(512, 6),
        )

    def forward(self, input):
        return self.net(input)

def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of PPO to play Flappy Bird""")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--num_iters", type=int, default=20000)
    parser.add_argument("--log_path", type=str, default="tensorboard_ppo")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--lmbda", type=float, default=0.95)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--eps", type=float, default=0.2)
    parser.add_argument("--batch_size",type=int, default=2048 )
    parser.add_argument("--mini_batch_size",type=int, default=64 )

    args = parser.parse_args()
    return args

def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)

#停止判断
def pause_game(paused):
    keys = key_check()
    if 'ESC' in keys:
        if paused:
            paused = False
            print('start game')
            time.sleep(1)
        else:
            paused = True
            print('pause game')
            time.sleep(1)
    if paused:
        print('paused')
        while True:
            keys = key_check()
            # pauses game and can get annoying.
            if 'ESC' in keys:
                if paused:
                    paused = False
                    print('start game')
                    time.sleep(1)
                    break
                else:
                    paused = True
                    time.sleep(1)
    return paused

def take_action(action):
    if action == 0: #表示什么都不做
        pass
    elif action == 1:
        directkey.go_forward() # w
    elif action == 2:
        directkey.go_back() # s
    elif action == 3:
        directkey.go_left() # a
    elif action == 4:
        directkey.go_right() # d
    elif action == 5:
        directkey.attack() # j
        
def action_judge(self_tank, enemy_tank, next_self_tank, next_enemy_tank, isbase_flag, flag, stop, emergence_break):
    if isbase_flag == 1: #基地被毁,游戏结束
        reward = -100
        print("基地炸了扣100分")
        return reward, stop, emergence_break
    if next_enemy_tank < enemy_tank: #敌人数量减少
        reward = 500
        print("奖励500分")
        return reward, stop, emergence_break  
    if next_self_tank < self_tank or flag == 0: #我方数量减少
        reward = -30
        print("你没了一条命扣30分")
        return reward, stop, emergence_break          
    if next_enemy_tank == 1 and stop == 0: #此时地图上还剩下1个敌人，说明游戏快结束了
        print("奖励一次100分")
        reward = 100
        stop = 1 #表示仅此奖励一次
        emergence_break = 100
        return reward, stop, emergence_break
    # 如果其他条件都不满足, 返回默认值
    return 3, stop, emergence_break  # 返回默认reward为0

def is_end(prev_screen):
    # 持续捕获基地窗口的屏幕内容
    curr_screen = grab_screen(window_base)
    
    next_self_tank = num_rec(window_self)#当前我方坦克的数量
    #check_sleep = 1 if next_self_tank > 0 else 0
    next_enemy_tank = num_rec(window_enemy)#观察敌方坦克变化
    
    #以下均为终止条件
    # #如果基地被毁，则为输
    if not np.array_equal(prev_screen, curr_screen):
        print("你基地炸了，游戏失败")
        flag = 0 #输了
        isbase_flag = 1
        done = 1
        return flag, isbase_flag, done, next_self_tank, next_enemy_tank
            
    elif next_enemy_tank == 0:
        print("游戏通关，即将进入下一关")
        flag = 1 #赢了
        isbase_flag = 0
        done = 1
        return flag, isbase_flag, done, next_self_tank, next_enemy_tank
    
    #如果我方坦克不存在并且此时我方坦克数量为0，则为输
    elif next_self_tank == 0:  #加速训练，测试只能有两条命
        print("你全军覆没了，游戏失败")
        flag = 0 #输了
        isbase_flag = 0
        done = 1
        return flag, isbase_flag, done, next_self_tank, next_enemy_tank
    return 999, 0, 0, next_self_tank, next_enemy_tank

#action_size = 6#动作空间维度

def get_state():
    state = []
    for _ in range(4):
        preimage = pre_processing(grab_screen(window_global), WIDTH, HEIGHT)
        preimage = torch.tensor(preimage)  # 将 numpy array 转换为 tensor
        processed_image = torch.cat(tuple(preimage for _ in range(4)))[None, :, :, :]
        state.append(processed_image)
    state = torch.cat(state, dim=1)
    return state.cuda()


def train(opt):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1993)
    else:
        torch.manual_seed(123)
    actor = PolicyNet().cuda()
    critic = ValueNet().cuda()

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=opt.lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=opt.lr)
    
    writer = SummaryWriter(opt.log_path)
    
    max_reward = 0
    iter = 0
    replay_memory = []
    evaluate_num = 0  # Record the number of evaluations
    evaluate_rewards = []
    
    stop = 0
    emergence_break = 0
    self_tank = num_rec(window_self)#当前我方坦克的数量
    enemy_tank = num_rec(window_enemy)#观察敌方坦克变化
    paused = False
    
    while iter < opt.num_iters:
        
        start_game()
        print("第",iter+1,"轮")
        #获取状态
        state = get_state()
       
        flag = 999 # 判断输赢
        isbase_flag = 0 #判断是否是因为基地被毁而输的
        done = 0#终止条件
        episode_return = 0.0
        
        # 捕获基地窗口的屏幕
        prev_screen = grab_screen(window_base)
        
        #清空key列表
        clearn = key_check()
        
        while not done:
            #动作
            prediction = actor(state)
            action_dist = torch.distributions.Categorical(prediction)
            action_sample = action_dist.sample()
            action = action_sample.item()
            take_action(action)

            #终态与奖励
            flag, isbase_flag, done, next_self_tank, next_enemy_tank = is_end(prev_screen)
            reward, stop, emergence_break = action_judge(self_tank, enemy_tank,
                                                                next_self_tank, next_enemy_tank,
                                                                isbase_flag, flag, stop, emergence_break)
                
            #获取下一状态
            next_state = get_state()
            
            replay_memory.append([state, action, reward, next_state, done])
            
            self_tank = next_self_tank#当前我方坦克的数量
            enemy_tank = next_enemy_tank#观察敌方坦克变化
            state = next_state
            episode_return += reward
            
            paused = pause_game(paused)
            
            if len(replay_memory) > opt.batch_size:
                state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = zip(*replay_memory)
                states = torch.cat(state_batch, dim=0).cuda()
                actions = torch.tensor(action_batch).view(-1, 1).cuda()
                rewards = torch.tensor(reward_batch).view(-1, 1).cuda()
                dones = torch.tensor(terminal_batch).view(-1, 1).int().cuda()
                next_states = torch.cat(next_state_batch, dim=0).cuda()

                with torch.no_grad():
                    td_target = rewards + opt.gamma * critic(next_states) * (1 - dones)
                    td_delta = td_target - critic(states)
                    advantage = compute_advantage(opt.gamma, opt.lmbda, td_delta.cpu()).cuda()
                    old_log_probs = torch.log(actor(states).gather(1, actions)).detach()

                for _ in range(opt.epochs):
                    for index in BatchSampler(SubsetRandomSampler(range(opt.batch_size)), opt.mini_batch_size, False):
                        log_probs = torch.log(actor(states[index]).gather(1, actions[index]))
                        ratio = torch.exp(log_probs - old_log_probs[index])
                        surr1 = ratio * advantage[index]
                        surr2 = torch.clamp(ratio, 1 - opt.eps, 1 + opt.eps) * advantage[index]  # 截断
                        actor_loss = torch.mean(-torch.min(surr1, surr2))
                        critic_loss = torch.mean(
                            nn.functional.mse_loss(critic(states[index]), td_target[index].detach()))
                        actor_optimizer.zero_grad()
                        critic_optimizer.zero_grad()
                        actor_loss.backward()
                        critic_loss.backward()
                        actor_optimizer.step()
                        critic_optimizer.step()
                replay_memory = []
                
        if episode_return > max_reward:
            max_reward = episode_return
            print(" max_reward Iteration: {}/{}, Reward: {}".format(iter + 1, opt.num_iters, episode_return))
            
        iter += 1
        if (iter+1) % 10 == 0:
            evaluate_num += 1
            evaluate_rewards.append(episode_return)
            print("evaluate_num:{} \t episode_return:{} \t".format(evaluate_num, episode_return))
            writer.add_scalar('step_rewards', evaluate_rewards[-1], global_step= iter)
        if (iter+1) % 100 == 0:
            actor_dict = {"net": actor.state_dict(), "optimizer": actor_optimizer.state_dict()}
            critic_dict = {"net": critic.state_dict(), "optimizer": critic_optimizer.state_dict()}
            torch.save(actor_dict, "{}/actor_good".format(opt.saved_path))
            torch.save(critic_dict, "{}/critic_good".format(opt.saved_path))
            

if __name__ == '__main__':

    opt = get_args()
    train(opt)
