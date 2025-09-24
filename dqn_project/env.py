# -*- coding: utf-8 -*-
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import random


class ThreePawnsEnv(gym.Env):
    """
    Jeu simplifié :
    - Rouge place 3 pions fixes, sans alignement possible
    - Bleu place 3 pions de départ (pas alignés)
    - Bleu (contrôlé par le DQN) déplace ses pions pour essayer d’aligner 3
    - La partie se termine si :
        * Bleu aligne ses 3 pions → victoire
        * Bleu ne peut plus bouger → défaite / nul
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        self.render_mode = render_mode
        self.board = np.zeros(9, dtype=int)  # 0 = vide, 1 = rouge, 2 = bleu

        self.observation_space = spaces.Box(low=0, high=2, shape=(9,), dtype=int)
        self.action_space = spaces.Discrete(81)  # 9x9 actions possibles

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros(9, dtype=int)

        # Placer les rouges (fixes, jamais alignés)
        while True:
            red_positions = random.sample(range(9), 3)
            for p in red_positions:
                self.board[p] = 1
            if not self._check_victory(1):
                break
            else:
                for p in red_positions:
                    self.board[p] = 0

        # Placer les bleus (jamais alignés au départ)
        while True:
            blue_positions = random.sample([i for i in range(9) if self.board[i] == 0], 3)
            for p in blue_positions:
                self.board[p] = 2
            if not self._check_victory(2):
                break
            else:
                for p in blue_positions:
                    self.board[p] = 0

        return self.board.copy(), {}

    def _check_victory(self, player=2):
        """Vérifie si un joueur aligne 3 pions"""
        board = self.board.reshape(3, 3)
        for i in range(3):
            if np.all(board[i, :] == player):
                return True
            if np.all(board[:, i] == player):
                return True
        if np.all(np.diag(board) == player):
            return True
        if np.all(np.diag(np.fliplr(board)) == player):
            return True
        return False

    def _neighbors(self, pos):
        """Retourne les cases voisines (y compris diagonales)"""
        neighbors = []
        row, col = divmod(pos, 3)
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = row + dr, col + dc
                if 0 <= nr < 3 and 0 <= nc < 3:
                    neighbors.append(nr * 3 + nc)
        return neighbors

    def _is_valid_move(self, from_pos, to_pos):
        """Vérifie si un déplacement bleu est valide"""
        if self.board[from_pos] != 2:
            return False
        if self.board[to_pos] != 0:
            return False
        if to_pos not in self._neighbors(from_pos):
            return False
        return True

    def _get_all_valid_moves(self, player=2):
        """Retourne tous les déplacements possibles"""
        moves = []
        for pos in range(9):
            if self.board[pos] == player:
                for neigh in self._neighbors(pos):
                    if self.board[neigh] == 0:
                        moves.append((pos, neigh))
        return moves

    def step(self, action):
        """Exécute une action"""
        from_pos = action // 9
        to_pos = action % 9

        reward = -1.0
        terminated = False
        truncated = False
        info = {}

        if not self._is_valid_move(from_pos, to_pos):
            reward = -5.0
            info["invalid"] = True
            return self.board.copy(), reward, terminated, truncated, info

        self.board[to_pos] = 2
        self.board[from_pos] = 0

        if self._check_victory(2):
            reward = 10.0
            terminated = True
            info["winner"] = "blue"
            return self.board.copy(), reward, terminated, truncated, info

        if not self._get_all_valid_moves(2):
            reward = -2.0
            terminated = True
            info["winner"] = "none"
            return self.board.copy(), reward, terminated, truncated, info

        return self.board.copy(), reward, terminated, truncated, info

    def render(self, mode="human", filename=None):
        """Affiche le plateau"""
        board_img = np.ones((300, 300, 3), dtype=np.uint8) * 255
        cell_size = 100

        for i in range(1, 3):
            board_img[i * cell_size - 1:i * cell_size + 1, :] = 0
            board_img[:, i * cell_size - 1:i * cell_size + 1] = 0

        for idx, val in enumerate(self.board):
            row, col = divmod(idx, 3)
            cx, cy = col * cell_size + 50, row * cell_size + 50
            if val == 1:
                color = (255, 0, 0)
            elif val == 2:
                color = (0, 0, 255)
            else:
                continue
            rr, cc = np.ogrid[:300, :300]
            mask = (rr - cy) ** 2 + (cc - cx) ** 2 <= 30 ** 2
            board_img[mask] = color

        if mode == "rgb_array":
            return board_img
        elif mode == "human":
            plt.imshow(board_img)
            plt.axis("off")
            plt.show()
