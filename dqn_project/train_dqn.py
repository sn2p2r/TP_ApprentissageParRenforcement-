# -*- coding: utf-8 -*-
import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from env import ThreePawnsEnv

# -----------------------------
# Hyperparamètres
# -----------------------------
EPISODES = 5000         # nombre d'épisodes max
GAMMA = 0.95           # facteur discount
EPSILON = 1.0          # exploration initiale
EPSILON_DECAY = 0.995  # décroissance plus lente
EPSILON_MIN = 0.1
LEARNING_RATE = 0.001
BATCH_SIZE = 64
MEMORY_SIZE = 5000
TARGET_UPDATE = 20     # mise à jour du réseau cible
MAX_STEPS = 30         # nombre max de coups par épisode

# -----------------------------
# Réseau DQN
# -----------------------------
class DQNAgent(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNAgent, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# -----------------------------
# Agent RL
# -----------------------------
class Agent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.gamma = GAMMA
        self.epsilon = EPSILON
        self.epsilon_decay = EPSILON_DECAY
        self.epsilon_min = EPSILON_MIN
        self.learning_rate = LEARNING_RATE

        self.model = DQNAgent(state_size, action_size)
        self.target_model = DQNAgent(state_size, action_size)  # réseau cible
        self.update_target()

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        self.steps = 0

    def update_target(self):
        """Copie des poids vers le réseau cible"""
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        """Stocker une transition dans la mémoire"""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, valid_actions_mask=None):
        """Choisir une action (epsilon-greedy)"""
        if np.random.rand() <= self.epsilon:
            if valid_actions_mask is not None:
                valid_actions = np.where(valid_actions_mask)[0]
                if len(valid_actions) > 0:
                    return np.random.choice(valid_actions)
            return random.randrange(self.action_size)

        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_tensor).squeeze(0).cpu().numpy()

        if valid_actions_mask is not None:
            q_values[~valid_actions_mask] = -np.inf

        return int(np.argmax(q_values))

    def replay(self, batch_size):
        """Rejouer des transitions (experience replay)"""
        if len(self.memory) < batch_size:
            return None

        minibatch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = torch.FloatTensor(np.array(states))
        next_states = torch.FloatTensor(np.array(next_states))
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        current_q = self.model(states).gather(1, actions)

        with torch.no_grad():
            next_q = self.target_model(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + (1 - dones) * self.gamma * next_q

        loss = self.criterion(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps += 1
        if self.steps % TARGET_UPDATE == 0:
            self.update_target()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.item()

    def load(self, name):
        self.model.load_state_dict(torch.load(name))
        self.update_target()

    def save(self, name):
        torch.save(self.model.state_dict(), name)


# -----------------------------
# Entraînement
# -----------------------------
def train_dqn():
    env = ThreePawnsEnv()
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = Agent(state_size, action_size)

    rewards_history = []
    win_rate_history = []
    wins = 0
    draws = 0
    losses = 0

    for e in range(EPISODES):
        state, _ = env.reset()
        state = np.array(state, dtype=np.float32)

        total_reward = 0
        done = False
        steps = 0   # compteur de coups

        # Boucle limitée par MAX_STEPS
        while not done and steps < MAX_STEPS:
            # masque des coups valides
            valid_actions_mask = np.zeros(env.action_space.n, dtype=bool)
            for f, t in env._get_all_valid_moves(2):
                valid_actions_mask[f * 9 + t] = True

            action = agent.act(state, valid_actions_mask)
            next_state, reward, done, truncated, info = env.step(action)
            next_state = np.array(next_state, dtype=np.float32)

            # shaping des récompenses
            if info.get("winner") == "blue":
                reward = 10.0
            elif info.get("winner") == "none":
                reward = -2.0
            elif info.get("invalid"):
                reward = -5.0
            else:
                reward = -1.0

            agent.remember(state, action, reward, next_state, done)
            agent.replay(BATCH_SIZE)

            state = next_state
            total_reward += reward
            steps += 1  # incrémenter le compteur

        # comptage des issues
        if info.get("winner") == "blue":
            wins += 1
        elif info.get("winner") == "none":
            draws += 1
        else:
            losses += 1

        rewards_history.append(total_reward)
        win_rate = wins / (e + 1)
        win_rate_history.append(win_rate)

        print(f"Épisode {e+1}/{EPISODES} | Reward: {total_reward:.1f} | "
              f"Epsilon: {agent.epsilon:.2f} | "
              f"Wins: {wins}, Draws: {draws}, Losses: {losses} | "
              f"WinRate: {win_rate*100:.1f}%")

        # early stop si taux de victoire > 80% sur les 50 derniers épisodes
        if e > 50 and np.mean(win_rate_history[-50:]) > 0.8:
            print("✅ Arrêt anticipé : l'agent a appris une stratégie gagnante.")
            break

    agent.save("dqn_model.pth")
    print("✅ Entraînement terminé, modèle sauvegardé.")

    # courbes
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(rewards_history)
    plt.title("Récompenses par épisode")
    plt.xlabel("Épisodes")
    plt.ylabel("Reward")

    plt.subplot(1, 2, 2)
    plt.plot([w * 100 for w in win_rate_history])
    plt.title("Taux de victoire (%)")
    plt.xlabel("Épisodes")
    plt.ylabel("WinRate %")

    plt.tight_layout()
    plt.savefig("training_curves.png")
    plt.show()


if __name__ == "__main__":
    train_dqn()
