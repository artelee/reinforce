import copy
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import gym
import time

from dezero import Model
import dezero.functions as F
import dezero.layers as L
from dezero import optimizers
import dezero as dezero

# --- 상태 정규화 ---
def normalize(state):
    pos = (state[0] + 1.2) / 1.8
    vel = (state[1] + 0.07) / 0.14
    return np.array([pos, vel], dtype=np.float32)

def shaped_reward(position):
    if position >= 0.5:         # 오른쪽 목표 도달
        return 100.0
    elif position <= -0.9:      # 왼쪽 끝까지 가는 것도 보상!
        return 5.0
    elif position > -0.4:
        return (1 + position) ** 2
    else:
        return 0.0

# --- Replay Buffer ---
class ReplayBuffer:
    def __init__(self, size, batch_size):
        self.buffer = deque(maxlen=size)
        self.batch_size = batch_size

    def add(self, s, a, r, s2, d):
        self.buffer.append((s, a, r, s2, d))

    def sample(self):
        batch = random.sample(self.buffer, self.batch_size)
        s = np.stack([x[0] for x in batch])
        a = np.array([x[1] for x in batch])
        r = np.array([x[2] for x in batch])
        s2 = np.stack([x[3] for x in batch])
        d = np.array([x[4] for x in batch]).astype(np.int32)
        return s, a, r, s2, d

# --- Q-Network ---
class QNet(Model):
    def __init__(self, action_size):
        super().__init__()
        self.l1 = L.Linear(128)
        self.l2 = L.Linear(128)
        self.l3 = L.Linear(64)
        self.out = L.Linear(action_size)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        return self.out(x)

# --- DQN Agent ---
class DQNAgent:
    def __init__(self, action_size):
        self.gamma = 0.99
        self.lr = 1e-3
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.action_size = action_size

        self.qnet = QNet(action_size)
        self.qnet_target = copy.deepcopy(self.qnet)
        self.optimizer = optimizers.Adam(self.lr)
        self.optimizer.setup(self.qnet)
        self.buffer = ReplayBuffer(10000, 64)

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)
        q = self.qnet(state[np.newaxis, :])
        return int(q.data.argmax())

    def update(self):
        if len(self.buffer.buffer) < self.buffer.batch_size:
            return

        s, a, r, s2, d = self.buffer.sample()
        q_values = self.qnet(s)
        q = q_values[np.arange(len(a)), a]

        with dezero.no_grad():
            next_q = self.qnet_target(s2)
            q_max = next_q.max(axis=1)
            target = r + (1 - d) * self.gamma * q_max

        loss = F.mean_squared_error(q, target)

        self.qnet.cleargrads()
        loss.backward()
        self.optimizer.update()

    def sync(self):
        self.qnet_target = copy.deepcopy(self.qnet)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

# --- 학습 환경 설정 ---
env = gym.make("MountainCar-v0", render_mode="rgb_array")
agent = DQNAgent(action_size=3)

episodes = 200
sync_interval = 20
reward_log = []

for ep in range(episodes):
    state = normalize(env.reset()[0])
    done = False
    total_reward = 0

    while not done:
        action = agent.get_action(state)
        next_raw, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state = normalize(next_raw)

        shaped = shaped_reward(next_raw[0])
        reward = -1.0 + shaped

        if next_raw[0] >= 0.5:
            print(f"🏁 Goal reached at episode {ep}, reward: {reward:.1f}")

        agent.buffer.add(state, action, reward, next_state, done)
        agent.update()

        total_reward += reward
        state = next_state

    agent.decay_epsilon()
    if ep % sync_interval == 0:
        agent.sync()

    reward_log.append(total_reward)
    print(f"[{ep}] reward: {total_reward:.2f}, epsilon: {agent.epsilon:.3f}")

# --- 보상 시각화 ---
plt.plot(reward_log)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("MountainCar-v0 DQN 학습 결과")
plt.grid()
plt.show()

# --- 테스트 실행 ---
env_test = gym.make("MountainCar-v0", render_mode="human")
state = normalize(env_test.reset()[0])
agent.epsilon = 0.0
done = False
total_reward = 0

while not done:
    action = agent.get_action(state)
    next_raw, _, terminated, truncated, _ = env_test.step(action)
    state = normalize(next_raw)
    done = terminated or truncated
    total_reward += -1.0 + shaped_reward(next_raw[0])
    env_test.render()
    time.sleep(0.02)

print("🎉 최종 테스트 보상:", total_reward)
env_test.close()
