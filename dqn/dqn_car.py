import copy
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import deque, Counter
import gym
import time

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

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)
        self.buffer.append(data)

    def __len__(self):
        return len(self.buffer)
    
    def get_batch(self):
        data = random.sample(self.buffer, self.batch_size)

        state = np.stack([x[0] for x in data])
        action = np.array([x[1] for x in data])
        reward = np.array([x[2] for x in data])
        next_state = np.stack([x[3] for x in data])
        done = np.array([x[4] for x in data]).astype(np.int32)
        return state, action, reward, next_state, done

class QNet(Model):
    def __init__(self, action_size):
        super().__init__()
        self.l1 = L.Linear(128)
        self.l2 = L.Linear(64)
        self.l3 = L.Linear(64)  
        self.out = L.Linear(action_size)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        return self.out(x)

class DQNAgent:
    def __init__(self):
        self.gamma = 0.98
        self.lr = 0.0001
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.buffer_size = 10000
        self.batch_size = 64
        self.action_size = 3

        self.replay_buffer = ReplayBuffer(self.buffer_size, self.batch_size)
        self.qnet = QNet(self.action_size)
        self.qnet_target = QNet(self.action_size)
        # self.optimizer = optimizers.SGD(self.lr)
        self.optimizer = optimizers.Adam(self.lr)
        self.optimizer.setup(self.qnet)

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            state = state[np.newaxis, :]
            qs = self.qnet(state)
            return qs.data.argmax()

    def update(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)
        if len(self.replay_buffer) < self.batch_size:
            return

        state, action, reward, next_state, done = self.replay_buffer.get_batch()
        qs = self.qnet(state)
        q = qs[np.arange(self.batch_size), action]

        next_qs = self.qnet_target(next_state)
        next_q = next_qs.max(axis=1)
        next_q.unchain()
        target = reward + (1 - done) * self.gamma * next_q

        loss = F.mean_squared_error(q, target)

        self.qnet.cleargrads()
        loss.backward()
        self.optimizer.update()

    def sync_qnet(self):
        self.qnet_target = copy.deepcopy(self.qnet)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

episodes = 10000
sync_interval = 40
env = gym.make("MountainCar-v0", render_mode="rgb_array")
agent = DQNAgent()
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

    if episode % sync_interval == 0:
        agent.sync_qnet()

    agent.decay_epsilon()

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
lr = {agent.lr}
epsilon = {agent.epsilon}
epsilon_min = {agent.epsilon_min}
epsilon_decay = {agent.epsilon_decay}
buffer_size = {agent.buffer_size}
batch_size = {agent.batch_size}
action_size = {agent.action_size}
episodes = {episodes}
sync_interval = {sync_interval}
QNet 구조: {agent.qnet.l1.out_size}-{agent.qnet.l2.out_size}-{agent.qnet.l3.out_size}-{agent.qnet.out.out_size}
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
