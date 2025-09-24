import gymnasium as gym
from gymnasium import spaces
import numpy as np

class ThreeInRowEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, render_mode=None):
        super(ThreeInRowEnv, self).__init__()
        self.render_mode = render_mode

        self.observation_space = spaces.Box(low=-1, high=1, shape=(9,), dtype=np.int8)
        self.action_space = spaces.Discrete(81)

        self.winning_lines = [
            [0,1,2], [3,4,5], [6,7,8],
            [0,3,6], [1,4,7], [2,5,8],
            [0,4,8], [2,4,6],
        ]

        self.neighbors = {
            0: [1,3,4], 1: [0,2,4], 2: [1,5,4],
            3: [0,6,4], 4: [0,1,2,3,5,6,7,8],
            5: [2,8,4], 6: [3,7,4], 7: [6,8,4], 8: [5,7,4]
        }

        self.state = None
        self.phase = None
        self.agent_pions = 0
        self.opponent_pions = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.zeros(9, dtype=np.int8)
        self.phase = "placement"
        self.agent_pions = 0
        self.opponent_pions = 0
        return self.state, {}

    def step(self, action):
        reward, terminated = 0.0, False

        if self.phase == "placement":
            pos = action % 9
            if self.state[pos] != 0:
                return self.state, -0.5, False, False, {}

            self.state[pos] = 1
            self.agent_pions += 1

            if self.agent_pions == 3:
                free = np.where(self.state == 0)[0]
                if len(free) >= 3:
                    np.random.shuffle(free)
                    for i in range(3):
                        self.state[free[i]] = -1
                    self.opponent_pions = 3
                self.phase = "movement"

        elif self.phase == "movement":
            from_node, to_node = divmod(action, 9)
            if self.state[from_node] == 1 and self.state[to_node] == 0 and to_node in self.neighbors[from_node]:
                self.state[from_node] = 0
                self.state[to_node] = 1
            else:
                return self.state, -0.5, False, False, {}

            if self.check_win(1):
                return self.state, 1.0, True, False, {}

            self.opponent_move()

            if self.check_win(-1):
                return self.state, -1.0, True, False, {}

        return self.state, reward, terminated, False, {}

    def check_win(self, player):
        return any(all(self.state[pos] == player for pos in line) for line in self.winning_lines)

    def opponent_move(self):
        own_positions = np.where(self.state == -1)[0]
        np.random.shuffle(own_positions)
        for pos in own_positions:
            neighs = list(self.neighbors[pos])
            np.random.shuffle(neighs)
            for neigh in neighs:
                if self.state[neigh] == 0:
                    self.state[pos] = 0
                    self.state[neigh] = -1
                    return

    def render(self):
        if self.render_mode == "human":
            symbols = {0: ".", 1: "X", -1: "O"}
            print("\nGrille actuelle :")
            for i in range(3):
                row = self.state[i*3:(i+1)*3]
                print(" ".join(symbols[cell] for cell in row))
            print(f"Phase : {self.phase} | Pions Agent : {self.agent_pions} | Pions Adversaire : {self.opponent_pions}")

    def get_valid_actions(self):
        valid = []
        if self.phase == "placement":
            valid = [i for i in range(9) if self.state[i] == 0]
        elif self.phase == "movement":
            for from_node in range(9):
                if self.state[from_node] == 1:
                    for to_node in self.neighbors[from_node]:
                        if self.state[to_node] == 0:
                            action = from_node * 9 + to_node
                            valid.append(action)
        return valid