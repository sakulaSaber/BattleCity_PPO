#运行该主程序，首先停5秒，然后运行screen_start.py文件，即开始游戏
#然后继续停3秒，之后就可以有agent来操作
# -*- coding: utf-8 -*-

import cv2
import pandas as pd
from getkeys import key_check
import directkey
import keyboard

import time
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
torch.backends.cudnn.enabled = False

from CBMA.cbma_test import CBAM

window_enemy = (959,369,1055,385)# 敌军数量窗口
window_self = (767,653,795,685)# 我方坦克数量窗口(959,384,1055,405)
window_global = (209,335,727,853)# 全局窗口
window_base = (462,826,475,853)# 基地窗口
WIDTH = 132
HEIGHT = 132


#输入状态画面
class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()
        # 网络层定义
        self.conv1 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2)
        self.cbam1 = CBAM(32)  # 在conv1后加入CBAM
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.cbam2 = CBAM(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2)
        self.cbam3 = CBAM(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=2)
        self.cbam4 = CBAM(256)
        self.fc1 =  nn.Sequential(nn.Linear((WIDTH // 8) * (HEIGHT // 8) * 64, 256), nn.Tanh())
        
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 6)
        
        # 激活函数和池化层
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Dropout
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.cbam1(x) # 应用CBAM
        x = self.max_pool(x)
        x = self.relu(self.conv2(x))
        x = self.cbam2(x) # 应用CBAM
        x = self.max_pool(x)        
        x = self.relu(self.conv3(x))
        x = self.cbam3(x) 
        x = self.max_pool(x)
        x = self.relu(self.conv4(x))
        x = self.cbam4(x) 
        x = self.max_pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        return nn.functional.softmax(self.fc3(x), dim=1)


class ValueNet(nn.Module):
    def __init__(self):
        super(ValueNet, self).__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2), nn.ReLU(),
            CBAM(32),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2), nn.ReLU(),
            CBAM(64),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2), nn.ReLU(),
            CBAM(128),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        )
        self.fc_net = nn.Sequential(
            nn.Linear((WIDTH // 8) * (HEIGHT // 8) * 128, 256), nn.Tanh(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.Dropout(0.5),
            nn.Linear(128, 6),
        )

    def forward(self, input):
        conv_out = self.conv_net(input)
        flatten = conv_out.view(conv_out.size(0), -1)  # flatten the tensor
        output = self.fc_net(flatten)
        return output

def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of PPO to play battle city""")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--num_iters", type=int, default=10000)
    parser.add_argument("--log_path", type=str, default="tensorboard_ppo")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--lmbda", type=float, default=0.95)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--eps", type=float, default=0.2)
    parser.add_argument("--batch_size",type=int, default=512)
    parser.add_argument("--mini_batch_size",type=int, default=32 )

    args = parser.parse_args()
    return args

def compute_advantage(gamma, lmbda, tensor_delta):
    td_delta = np.array(tensor_delta.detach())  # 修改部分
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta[0]
        advantage_list.append(advantage)
    advantage_list.reverse()
    advantage_list = np.array(advantage_list)  # 添加这行
    return torch.tensor(advantage_list, dtype=torch.float)  # 确保输入是 numpy array

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
        
def action_judge(self_tank, enemy_tank, next_self_tank, next_enemy_tank, isbase_flag, flag):
    if isbase_flag == 1: #基地被毁,游戏结束
        reward = -100
        print("基地炸了扣100分")
        return reward
    if next_enemy_tank < enemy_tank: #敌人数量减少
        reward = 500
        print("奖励500分")
        return reward  
    if next_self_tank < self_tank or flag == 0: #我方数量减少
        reward = -30
        print("你没了一条命扣30分")
        return reward          
    if next_enemy_tank == 1: #此时地图上还剩下1个敌人，说明游戏快结束了
        print("奖励一次100分")
        reward = 100
        return reward
    # 如果其他条件都不满足, 返回默认值
    return 3  # 返回默认reward为3

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
    return state


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
    
    if os.path.exists("{}/actor".format(opt.saved_path)):
        checkpoint = torch.load("{}/actor".format(opt.saved_path))
        actor.load_state_dict(checkpoint['net'])
        actor_optimizer.load_state_dict(checkpoint['optimizer'])
        print("load actor succ")
    
    if os.path.exists("{}/critic".format(opt.saved_path)):
        checkpoint = torch.load("{}/critic".format(opt.saved_path))
        critic.load_state_dict(checkpoint['net'])
        critic_optimizer.load_state_dict(checkpoint['optimizer'])
        print("load critic succ")
    
    max_reward = 0
    iter = 0
    replay_memory = []
    evaluate_num = 0  # Record the number of evaluations
    evaluate_rewards = []
    
    self_tank = num_rec(window_self)#当前我方坦克的数量
    enemy_tank = num_rec(window_enemy)#观察敌方坦克变化
    paused = False
    
    while iter < opt.num_iters:
        
        start_game()
        print("第",iter+1,"轮")
        #获取状态
        state = get_state()
        state_cuda = state.cuda()
        
        flag = 999 # 判断输赢
        isbase_flag = 0 #判断是否是因为基地被毁而输的
        done = 0#终止条件
        episode_return = 0.0
        stop = 0
        
        # 捕获基地窗口的屏幕
        prev_screen = grab_screen(window_base)
        
        #清空key列表
        clearn = key_check()
        
        while not done:
            if stop == 1:
                break
            #动作
            #这行代码将当前的状态（state_cuda）输入到策略网络（actor）中，
            # 并获取网络的输出。这个输出（prediction）通常是一个概率分布，
            # 表示在当前状态下采取各种可能动作的概率。状态被转移到CUDA设备上（如果可用），以利用GPU加速计算。
            prediction = actor(state_cuda)
            # 这行代码使用PyTorch的Categorical分布来创建一个概率分布对象。
            # Categorical分布是多项分布的特例，它用于模拟具有给定概率的离散随机变量。
            # 在这个场景中，这个随机变量表示不同的动作，而prediction向量中的每个元素表示采取相应动作的概率。
            action_dist = torch.distributions.Categorical(prediction)
            # 这行代码从action_dist表示的概率分布中抽取一个样本。在这个上下文中，
            # 这意味着根据策略网络输出的概率分布随机选择一个动作。这是强化学习中探索（exploration）的一个重要部分，
            # 它允许代理在训练过程中尝试不同的动作。
            action_sample = action_dist.sample()
            # 这行代码将抽取的样本（action_sample）转换为一个Python标量（如果它是一个单元素张量的话）。
            # 这个标量代表了所选择的动作。这一步是必要的，因为后续的take_action函数需要一个具体的动作值作为输入，而不是一个张量。
            action = action_sample.item()
            
            take_action(action)

            #终态与奖励
            flag, isbase_flag, done, next_self_tank, next_enemy_tank = is_end(prev_screen)
            reward = action_judge(self_tank, enemy_tank, next_self_tank, next_enemy_tank, isbase_flag, flag)
                                                               
                
            #获取下一状态
            next_state = get_state()
            next_state_cuda = next_state.cuda()
            
            replay_memory.append([state_cuda, action, reward, next_state_cuda, done])

            
            self_tank = next_self_tank#当前我方坦克的数量
            enemy_tank = next_enemy_tank#观察敌方坦克变化
            state_cuda = next_state_cuda
            episode_return += reward       
            
            paused = pause_game(paused)
            
            if len(replay_memory) > opt.batch_size:
                stop = 1
                # 从经验回放池中解压数据，分别获取状态、动作、奖励、下一个状态和是否终止的标志的批次。
                state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = zip(*replay_memory)
                # 将批次数据转换为PyTorch张量，并使用.cuda()将它们移动到GPU上（如果可用），以利用GPU加速计算。
                states = torch.cat(state_batch, dim=0).cuda()
                actions = torch.tensor(action_batch).view(-1, 1).cuda()
                rewards = torch.tensor(reward_batch).view(-1, 1).cuda()
                dones = torch.tensor(terminal_batch).view(-1, 1).int().cuda()
                next_states = torch.cat(next_state_batch, dim=0).cuda()
                print("Shape of actions: ", actions.size())
                
                with torch.no_grad():
                    # 在不计算梯度的上下文中，计算时间差分（TD）目标和优势函数。TD目标是基于奖励、
                    # 折扣因子（opt.gamma）和价值网络对下一个状态的估计计算得出的。优势函数通过compute_advantage函数计算，
                    # 它利用TD残差和GAE（Generalized Advantage Estimation）方法来估计。
                    td_target = rewards + opt.gamma * critic(next_states) * (1 - dones)
                    td_delta = td_target - critic(states)
                    advantage = compute_advantage(opt.gamma, opt.lmbda, td_delta.cpu()).cuda()
                    
                    print("Shape of actor(states): ", actor(states).size())
                    old_log_probs = torch.log(actor(states).gather(1, actions)).detach()

                for _ in range(opt.epochs):# 计算策略损失和价值损失。策略损失使用PPO的clip方法来限制策略更新步长。价值损失是价值网络输出和TD目标之间的均方误差。
                    # 进行多轮更新（opt.epochs），在每轮中，使用BatchSampler和SubsetRandomSampler来迭代经验回放池中的小批次数据。
                    for index in BatchSampler(SubsetRandomSampler(range(opt.batch_size)), opt.mini_batch_size, False):
                        # actor(states[index])：对于当前的小批量数据，通过策略网络（actor）计算动作的概率分布。
                        # .gather(1, actions[index])：使用gather方法收集与实际采取的动作相对应的概率。
                        # torch.log()：计算这些概率的对数。
                        log_probs = torch.log(actor(states[index]).gather(1, actions[index]))
                        # 这行代码计算了新旧策略概率之比。old_log_probs[index]是旧策略下采取相同动作的对数概率，
                        # 而log_probs是新策略下的对数概率。torch.exp()用于计算e的指数，从而得到新旧概率的比值。
                        ratio = torch.exp(log_probs - old_log_probs[index])
                        # surr1：计算未截断的目标，即概率比率乘以优势函数。
                        # surr2：计算截断的目标。torch.clamp()函数将概率比率限制在[1 - opt.eps, 1 + opt.eps]的范围内，这是PPO算法的一个关键特点，用于避免策略更新过大。
                        surr1 = ratio * advantage[index]
                        surr2 = torch.clamp(ratio, 1 - opt.eps, 1 + opt.eps) * advantage[index]  # 截断
                        # 计算策略损失，取surr1和surr2中较小的一个，并对其取负值，因为我们是在最大化目标函数。然后计算这些损失的平均值。
                        actor_loss = torch.mean(-torch.min(surr1, surr2))
                        # 使用均方误差（MSE）损失函数计算价值网络（critic）的损失。critic(states[index])计算当前状态的价值估计，
                        # td_target[index]是目标价值（TD目标）。.detach()用于防止在计算TD目标时计算梯度，因为我们只想更新价值网络的参数。
                        critic_loss = torch.mean(
                            nn.functional.mse_loss(critic(states[index]), td_target[index].detach()))
                        # 在每个小批次数据上执行反向传播来计算梯度，然后执行梯度下降步骤来更新策略网络和价值网络的权重。
                        actor_optimizer.zero_grad()
                        critic_optimizer.zero_grad()
                        actor_loss.backward()
                        critic_loss.backward()
                        actor_optimizer.step()
                        critic_optimizer.step()
                    
                replay_memory = []
                 
        print("2:{}".format(torch.cuda.memory_allocated(0)))
                
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
            torch.save(actor_dict, "{}/actor".format(opt.saved_path))
            torch.save(critic_dict, "{}/critic".format(opt.saved_path))
            

if __name__ == '__main__':

    opt = get_args()
    train(opt)
