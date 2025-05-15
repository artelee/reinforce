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

    save_dir = "saved_models"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    filename = f"{timestamp}_ep{episode}_reward{int(reward)}.npz"
    save_path = os.path.join(save_dir, filename)

    model_params = {}
    for name, param in agent.qnet.__dict__.items():
        if isinstance(param, dezero.Parameter):
            model_params[name] = param.data

    np.savez(save_path,
             **model_params,
             epsilon=agent.epsilon,
             episode=episode,
             reward=reward)

    print(f"모델이 저장되었습니다: {filename}")

def normalize(state):
    pos = (state[0] + 1.2) / 1.8
    vel = (state[1] + 0.07) / 0.14
    return np.array([pos, vel], dtype=np.float32)

class PolicyNet(Model):
    def __init__(self, action_size=2):
        super().__init__()
        self.l1 = L.Linear(128)
        self.l2 = L.Linear(action_size)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        x = F.softmax(x)
        return x

class ValueNet(Model):
    def __init__(self):
        super().__init__()
        self.l1 = L.Linear(128)
        self.l2 = L.Linear(1)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x

class Agent:
    def __init__(self):
        self.gamma = 0.98
        self.lr_pi = 0.0002
        self.lr_v = 0.0005
        self.action_size = 2

        self.pi = PolicyNet()
        self.v = ValueNet()
        self.optimizer_pi = optimizers.Adam(self.lr_pi).setup(self.pi)
        self.optimizer_v = optimizers.Adam(self.lr_v).setup(self.v)

    def get_action(self, state):
        state = state[np.newaxis, :]
        probs = self.pi(state)
        probs = probs[0]
        action = np.random.choice(len(probs), p=probs.data)
        return action, probs[action]


    def update(self, state, action_prob, reward, next_state, done):
        state = state[np.newaxis, :]
        next_state = next_state[np.newaxis, :]

        target = reward + self.gamma * self.v(next_state) * (1 - done)
        target.unchain()
        v = self.v(state)
        loss_v = F.mean_squared_error(v, target)

        # 정책의 손실계산
        delta = target - v
        delta.unchain()
        loss_pi = -F.log(action_prob) * delta

        self.v.cleargrads()
        self.pi.cleargrads()
        loss_v.backward()
        loss_pi.backward()
        self.optimizer_v.update()
        self.optimizer_pi.update()

episodes = 3000
env = gym.make("MountainCar-v0", render_mode="rgb_array")
agent = Agent()
reward_history = []

for episode in range(episodes):
    state = env.reset()[0]
    done = False
    total_reward = 0

    while not done:
        action = agent.get_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated | truncated
        # reward = custom_reward(state, action, next_state)

        agent.update(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

    reward_history.append(total_reward)
    if episode % 10 == 0:
        print("episode :{}, total reward : {}".format(episode, total_reward))
    if (episode > 1000 and episode % 500 == 0) or (episode == episodes - 1):
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
"""
with open(os.path.join(exp_dir, "params.txt"), "w", encoding="utf-8") as f:
    f.write(params_text)


num_test_episodes = 5
rewards = []
for i in range(num_test_episodes):

    env_test = gym.make("MountainCar-v0", render_mode="rgb_array")

    agent.epsilon = 0
    state = normalize(env_test.reset()[0])
    done = False
    total_reward = 0
    frames = []

    while not done:
        frame = env_test.render()
        if frame is not None:
            frames.append(frame)

        action = agent.get_action(state)
        next_state, reward, terminated, truncated, info = env_test.step(action)
        done = terminated | truncated
        state = next_state
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
