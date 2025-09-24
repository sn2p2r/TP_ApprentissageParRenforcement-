# -*- coding: utf-8 -*-
import numpy as np
import random
import pickle
from collections import defaultdict
from env import ThreePawnsEnv

# Hyperparamètres
EPISODES = 5000      # Nombre d'épisodes d'entraînement
EPSILON = 0.2        # Exploration
GAMMA = 0.95         # Facteur de discount
MAX_STEPS = 30       # Limite max de coups par partie (évite les boucles infinies)


def epsilon_greedy(Q, state, env, epsilon=0.1):
    """Politique epsilon-greedy pour choisir une action"""
    state_key = tuple(state)
    if random.random() < epsilon:
        valid_actions = [f * 9 + t for f, t in env._get_all_valid_moves(2)]
        if not valid_actions:
            return 0
        return random.choice(valid_actions)
    else:
        q_vals = Q[state_key]
        return max(q_vals, key=q_vals.get, default=0)


def mc_control():
    env = ThreePawnsEnv()
    Q = defaultdict(lambda: defaultdict(float))
    counts = defaultdict(int)  # compte le nombre de visites (s,a)

    for ep in range(EPISODES):
        state, _ = env.reset()
        episode = []
        done = False
        steps = 0

        # Génération d'un épisode
        while not done and steps < MAX_STEPS:
            action = epsilon_greedy(Q, state, env, EPSILON)
            next_state, reward, terminated, truncated, info = env.step(action)
            episode.append((state, action, reward))
            state = next_state
            done = terminated or truncated
            steps += 1

        # Retour Monte Carlo (First-Visit, mais avec moyenne incrémentale)
        G = 0
        visited = set()
        for s, a, r in reversed(episode):
            G = GAMMA * G + r
            key = (tuple(s), a)
            if key not in visited:
                counts[key] += 1
                Q[tuple(s)][a] += (G - Q[tuple(s)][a]) / counts[key]
                visited.add(key)

        # Logs
        if (ep + 1) % 100 == 0:
            print(f"Épisode {ep+1}/{EPISODES} terminé.")

    # Sauvegarde de la politique
    with open("monte_carlo_project/mc_policy.pkl", "wb") as f:
        pickle.dump(dict(Q), f)
    print("✅ Politique sauvegardée : mc_policy.pkl")


if __name__ == "__main__":
    mc_control()
