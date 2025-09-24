# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import numpy as np
import torch

from env import ThreePawnsEnv
from train_dqn import Agent, DQNAgent


class DQNLauncher:
    """
    Interface Tkinter pour jouer automatiquement avec le modèle DQN
    - Affiche le plateau (3x3)
    - Le DQN (Bleu) joue seul contre Rouge (fixe)
    - Rouge ne bouge jamais : il sert d'obstacle
    - Statistiques affichées : parties jouées, victoires bleues, taux de victoire
    """

    def __init__(self, root, env, agent):
        self.root = root
        self.env = env
        self.agent = agent

        # Reset initial de l'environnement
        self.state, _ = self.env.reset()

        # Canvas pour le plateau
        self.canvas = tk.Canvas(root, width=300, height=300)
        self.canvas.pack()

        # Boutons
        self.btn_frame = tk.Frame(root)
        self.btn_frame.pack()

        self.btn_next = tk.Button(
            self.btn_frame, text="Tour suivant (DQN)", command=self.step_dqn
        )
        self.btn_next.grid(row=0, column=0)

        self.btn_reset = tk.Button(
            self.btn_frame, text="Nouvelle Partie", command=self.new_game
        )
        self.btn_reset.grid(row=0, column=1)

        self.btn_quit = tk.Button(self.btn_frame, text="Quitter", command=root.quit)
        self.btn_quit.grid(row=0, column=2)

        # Label des stats
        self.stats_label = tk.Label(root, text="Taux de victoire : 0.0% (0/0)")
        self.stats_label.pack()

        # Compteurs
        self.total_games = 0
        self.blue_wins = 0

        # Affichage initial du plateau
        self.update_board()

    def update_board(self):
        """Met à jour l'affichage du plateau"""
        img_array = self.env.render(mode="rgb_array")
        img = Image.fromarray(img_array)
        self.photo = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

    def step_dqn(self):
        """Fait jouer un coup au DQN"""
        valid_actions_mask = np.zeros(self.env.action_space.n, dtype=bool)
        for f, t in self.env._get_all_valid_moves(2):  # uniquement les coups bleus valides
            valid_actions_mask[f * 9 + t] = True

        action = self.agent.act(self.state, valid_actions_mask)
        next_state, reward, terminated, truncated, info = self.env.step(action)
        self.state = next_state
        self.update_board()

        if terminated:
            self.total_games += 1
            if info.get("winner") == "blue":
                self.blue_wins += 1
                messagebox.showinfo("Résultat", "Victoire BLEU (DQN) 🎉")
            else:
                messagebox.showinfo("Résultat", "Match nul (aucun coup possible)")

            win_rate = (self.blue_wins / self.total_games) * 100
            self.stats_label.config(
                text=f"Taux de victoire : {win_rate:.1f}% ({self.blue_wins}/{self.total_games})"
            )
            self.new_game()

    def new_game(self):
        """Relance une partie"""
        self.state, _ = self.env.reset()
        self.update_board()


def main():
    env = ThreePawnsEnv(render_mode="rgb_array")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # Charger l’agent entraîné
    agent = Agent(state_size, action_size)
    agent.model = DQNAgent(state_size, action_size)
    agent.load("dqn_model.pth")
    agent.epsilon = 0.0  # Mode greedy = pas d'exploration aléatoire

    # Lancer l’interface
    root = tk.Tk()
    root.title("Jeu Three Pawns - DQN Launcher")
    app = DQNLauncher(root, env, agent)
    root.mainloop()


if __name__ == "__main__":
    main()
