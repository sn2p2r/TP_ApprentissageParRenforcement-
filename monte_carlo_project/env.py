# -*- coding: utf-8 -*-
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random
from PIL import Image, ImageDraw


class ThreePawnsEnv(gym.Env):
    """
    Jeu Three Pawns (version Monte Carlo)
    - Rouge : 3 pions fixes posés aléatoirement (jamais alignés)
    - Bleu : 3 pions posés aléatoirement (jamais alignés)
    - Seul Bleu bouge (déplacement d’un pion par coup)
    - Fin si Bleu aligne ses 3 pions (victoire)
    - Fin si Bleu ne peut plus bouger (match nul/défaite)
    """

    metadata = {"render_modes": ["ansi", "rgb_array"]}

    def __init__(self):
        super().__init__()
        self.board = np.zeros(9, dtype=int)  # 0=vide, 1=rouge, 2=bleu

        self.observation_space = spaces.Box(low=0, high=2, shape=(9,), dtype=int)
        self.action_space = spaces.Discrete(81)  # 9*9 actions possibles

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros(9, dtype=int)

        # Placement des rouges (fixes)
        while True:
            red_positions = random.sample(range(9), 3)
            for p in red_positions:
                self.board[p] = 1
            if not self._check_victory(1):
                break
            else:
                for p in red_positions:
                    self.board[p] = 0

        # Placement des bleus (jamais alignés au départ)
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
        b = self.board.reshape(3, 3)
        for i in range(3):
            if np.all(b[i, :] == player): return True
            if np.all(b[:, i] == player): return True
        if np.all(np.diag(b) == player): return True
        if np.all(np.diag(np.fliplr(b)) == player): return True
        return False

    def _neighbors(self, pos):
        neighbors = []
        r, c = divmod(pos, 3)
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0: continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < 3 and 0 <= nc < 3:
                    neighbors.append(nr * 3 + nc)
        return neighbors

    def _is_valid_move(self, f, t):
        return (self.board[f] == 2 and self.board[t] == 0 and t in self._neighbors(f))

    def _get_all_valid_moves(self, player=2):
        moves = []
        for pos in range(9):
            if self.board[pos] == player:
                for neigh in self._neighbors(pos):
                    if self.board[neigh] == 0:
                        moves.append((pos, neigh))
        return moves

    def step(self, action):
        f, t = divmod(action, 9)
        reward, terminated, truncated = -1.0, False, False
        info = {}

        if not self._is_valid_move(f, t):
            reward = -5.0
            info["invalid"] = True
            return self.board.copy(), reward, terminated, truncated, info

        # Déplacer pion bleu
        self.board[t] = 2
        self.board[f] = 0

        if self._check_victory(2):
            reward, terminated = 1.0, True
            info["winner"] = "blue"
            return self.board.copy(), reward, terminated, truncated, info

        if not self._get_all_valid_moves(2):
            reward, terminated = -1.0, True
            info["winner"] = "none"
            return self.board.copy(), reward, terminated, truncated, info

        return self.board.copy(), reward, terminated, truncated, info

    def render(self, mode="ansi", filename=None):
        """
        Rendu visuel identique à DQN : plateau avec image PNG + pions colorés
        """
        b = self.board.reshape(3, 3)

        if mode == "ansi":
            symbols = {0: ".", 1: "R", 2: "B"}
            return "\n".join(" ".join(symbols[v] for v in row) for row in b)

        elif mode == "rgb_array":
            # Charger ton plateau (image de fond identique au DQN)
            board_img = Image.open("board.png").convert("RGB").resize((300, 300))
            draw = ImageDraw.Draw(board_img)

            # Coordonnées pour les cases
            cell_size = 100
            for idx, val in enumerate(self.board):
                r, c = divmod(idx, 3)
                x = c * cell_size + cell_size // 2
                y = r * cell_size + cell_size // 2
                if val == 1:  # Rouge
                    draw.ellipse((x - 25, y - 25, x + 25, y + 25), fill=(255, 0, 0))
                elif val == 2:  # Bleu
                    draw.ellipse((x - 25, y - 25, x + 25, y + 25), fill=(0, 0, 255))

            if filename:
                board_img.save(filename)

            return np.array(board_img)

        else:
            raise NotImplementedError(f"Mode {mode} non supporté")
