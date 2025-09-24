# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import messagebox
import pickle
from env import ThreePawnsEnv


CELL_SIZE = 100     # Taille des cases du plateau
PAWN_RADIUS = 30    # Rayon des pions (rouges/bleus)


class MCLauncher:
    def __init__(self, root, env, policy):
        self.root = root
        self.env = env
        self.policy = policy
        self.state, _ = self.env.reset()

        # Canvas pour afficher la grille et les pions
        self.canvas = tk.Canvas(root, width=3*CELL_SIZE, height=3*CELL_SIZE, bg="white")
        self.canvas.pack()

        # Boutons
        self.step_button = tk.Button(root, text="Jouer un coup", command=self.step_agent)
        self.step_button.pack(side=tk.LEFT, padx=10)

        self.reset_button = tk.Button(root, text="Nouvelle Partie", command=self.new_game)
        self.reset_button.pack(side=tk.RIGHT, padx=10)

        # Afficher le plateau initial
        self.update_board()

    def draw_grid(self):
        """Dessine la grille 3x3"""
        for i in range(4):
            # Lignes horizontales
            self.canvas.create_line(0, i*CELL_SIZE, 3*CELL_SIZE, i*CELL_SIZE, width=2)
            # Lignes verticales
            self.canvas.create_line(i*CELL_SIZE, 0, i*CELL_SIZE, 3*CELL_SIZE, width=2)

    def update_board(self):
        """Met à jour l'affichage du plateau"""
        self.canvas.delete("all")  # Effacer ancien dessin
        self.draw_grid()

        for idx, val in enumerate(self.state):
            r, c = divmod(idx, 3)
            x = c * CELL_SIZE + CELL_SIZE//2
            y = r * CELL_SIZE + CELL_SIZE//2

            if val == 1:  # Rouge
                self.canvas.create_oval(
                    x - PAWN_RADIUS, y - PAWN_RADIUS,
                    x + PAWN_RADIUS, y + PAWN_RADIUS,
                    fill="red"
                )
            elif val == 2:  # Bleu
                self.canvas.create_oval(
                    x - PAWN_RADIUS, y - PAWN_RADIUS,
                    x + PAWN_RADIUS, y + PAWN_RADIUS,
                    fill="blue"
                )

    def step_agent(self):
        """Un coup joué par la politique Monte Carlo"""
        state_tuple = tuple(self.state)
        if state_tuple in self.policy:
            q_values = self.policy[state_tuple]
            action = max(q_values, key=q_values.get)  # action avec meilleure Q-value
        else:
            action = self.env.action_space.sample()   # aléatoire si inconnu

        next_state, reward, terminated, truncated, info = self.env.step(action)
        self.state = next_state

        self.update_board()

        if terminated:
            winner = info.get("winner", "aucun")
            messagebox.showinfo("Fin de partie", f"Gagnant: {winner}")

    def new_game(self):
        """Réinitialise une partie"""
        self.state, _ = self.env.reset()
        self.update_board()


def main():
    env = ThreePawnsEnv()
    try:
        with open("mc_policy.pkl", "rb") as f:
            policy = pickle.load(f)
    except FileNotFoundError:
        print("⚠️ Aucune politique trouvée, policy vide utilisée.")
        policy = {}

    root = tk.Tk()
    root.title("Monte Carlo - Three Pawns")
    app = MCLauncher(root, env, policy)
    root.mainloop()


if __name__ == "__main__":
    main()
