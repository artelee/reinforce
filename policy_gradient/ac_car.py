import copy
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import deque, Counter
import gym
import time
from utils import plot_total_reward
from dezero import Model
import dezero.functions as F
import dezero.layers as L
from dezero import optimizers
import dezero as dezero
import datetime
import os

import imageio
from moviepy.editor import ImageSequenceClip

def save_model(agent, reward, episode):
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")

    save_dir = "saved_models_ac"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


class PolicyNet(Model):
    def __init__(self, state_size=2, action_size=3):
        super().__init__()
        self.l1 = L.Linear(256, in_size=state_size)
        self.l2 = L.Linear(128)
        self.l3 = L.Linear(64)
        self.l4 = L.Linear(action_size)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        x = F.softmax(x)
        return x

class ValueNet(Model):
    def __init__(self, state_size=2):
        super().__init__()
        self.l1 = L.Linear(256, in_size=state_size)
        self.l2 = L.Linear(128)
        self.l3 = L.Linear(64)
        self.l4 = L.Linear(1)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = self.l4(x)
        return x

class Agent:
    def __init__(self):
        self.gamma = 0.98
        self.lr_pi = 0.0003
        self.lr_v = 0.0001
        self.action_size = 3
        self.state_size = 2
        self.entropy_coef = 0.01

        self.pi = PolicyNet(state_size=self.state_size, action_size=self.action_size)
        self.v = ValueNet(state_size=self.state_size)
        self.optimizer_pi = optimizers.Adam(self.lr_pi).setup(self.pi)
        self.optimizer_v = optimizers.Adam(self.lr_v).setup(self.v)


    def get_action(self, state):
        state = state[np.newaxis, :]
        probs = self.pi(state)[0]
        action = np.random.choice(self.action_size, p=probs.data)
        return action, probs  # return only action and full probs
        
    def update(self, state, probs, action, reward, next_state, done):
        state = state[np.newaxis, :]
        next_state = next_state[np.newaxis, :]

        v = self.v(state)
        
        next_v = self.v(next_state) * (1 - done)
        
        target = reward + self.gamma * next_v
        target.unchain()
        
        advantage = target - v
        advantage.unchain()

        loss_v = F.mean_squared_error(v, target)
        self.v.cleargrads()
        loss_v.backward()
        self.optimizer_v.update()

        action_prob = probs[action] 
        log_prob = F.log(action_prob + 1e-8)
        
        entropy = -F.sum(probs * F.log(probs + 1e-8))
        
        loss_pi = -(log_prob * advantage + self.entropy_coef * entropy)
        
        self.pi.cleargrads()
        loss_pi.backward()
        self.optimizer_pi.update()

    
def normalize(state):
    pos = (state[0] + 1.2) / 1.8
    vel = (state[1] + 0.07) / 0.14
    return np.array([pos, vel], dtype=np.float32)

episodes = 50000
env = gym.make("MountainCar-v0", render_mode="rgb_array")
agent = Agent()
reward_history = []

for episode in range(episodes):
    state = normalize(env.reset()[0])
    done = False
    total_reward = 0

    while not done:
        action, probs = agent.get_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated | truncated

        agent.update(state, probs, action, reward, normalize(next_state), done)
        
        state = normalize(next_state)
        total_reward += reward

    reward_history.append(total_reward)
    if episode % 10 == 0:
        print(f"에피소드: {episode}, 총 보상: {total_reward}")

    if (episode == episodes - 1):
        save_model(agent, total_reward, episode)

plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.plot(range(len(reward_history)), reward_history)



now = datetime.datetime.now()
exp_dir = f"test_results/{now.strftime('%Y%m%d_%H%M%S')}"
os.makedirs(exp_dir, exist_ok=True)
graph_path = os.path.join(exp_dir, "reward_history.png")
plt.savefig(graph_path, dpi=150)  # 해상도는 필요에 따라 조정
plt.show()
# --- 파라미터 저장 ---
params_text = f"""
gamma = {agent.gamma}
lr-pi = {agent.lr_pi}
lr-v = {agent.lr_v}
episodes = {episodes}
entropy_coef = {agent.entropy_coef}
# Actor (Policy) Network
pi-layer1 = {agent.pi.l1.out_size}/{agent.pi.l1.in_size}
pi-layer2 = {agent.pi.l2.out_size}/{agent.pi.l2.in_size}
pi-layer3 = {agent.pi.l3.out_size}/{agent.pi.l3.in_size}
pi-layer4 = {agent.pi.l4.out_size}/{agent.pi.l4.in_size}
# Critic (Value) Network 
v-layer1 = {agent.v.l1.out_size}/{agent.v.l1.in_size}
v-layer2 = {agent.v.l2.out_size}/{agent.v.l2.in_size}
v-layer3 = {agent.v.l3.out_size}/{agent.v.l3.in_size}
v-layer4 = {agent.v.l4.out_size}/{agent.v.l4.in_size}
"""
with open(os.path.join(exp_dir, "params.txt"), "w", encoding="utf-8") as f:
    f.write(params_text)

num_test_episodes = 5
rewards = []
for i in range(num_test_episodes):

    env_test = gym.make("MountainCar-v0", render_mode="rgb_array")

    state = normalize(env_test.reset()[0])
    done = False
    total_reward = 0
    frames = []

    while not done:
        frame = env_test.render()
        if frame is not None:
            frames.append(frame)

        action, probs = agent.get_action(state)
        next_state, reward, terminated, truncated, info = env_test.step(action)
        done = terminated | truncated
        agent.update(state, probs, action, reward, normalize(next_state), done)
        state = normalize(next_state)
        total_reward += reward
        # env_test.render()
    rewards.append(total_reward)
    video_path = os.path.join(exp_dir, f"test_episode_{i+1}_{total_reward}.mp4")
    print("최종 테스트 보상:", total_reward)
    clip = ImageSequenceClip(frames, fps=30)
    clip.write_videofile(video_path, codec="libx264")
    print(f"테스트 에피소드 {i+1} 보상: {total_reward} (동영상 저장 완료)")


with open(os.path.join(exp_dir, "rewards.txt"), "w", encoding="utf-8") as f:
    for idx, r in enumerate(rewards):
        f.write(f"에피소드 {idx+1}: {r}\n")
    f.write(f"최대 보상: {max(rewards)}\n")

env_test.close()
print("완료")
