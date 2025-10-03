from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
import numpy as np
import random
import gymnasium as gym
from gymnasium import spaces

class DroneNavigationEnv(AECEnv):
    metadata = {'render_modes': ['human'], 'name': 'drone_navigation_v0'}

    def __init__(self, width=20, height=20, num_drones=2, num_targets=5, num_obstacles=5):
        super().__init__()
        self.width = width
        self.height = height
        self.num_drones = num_drones
        self.num_targets = num_targets
        self.num_obstacles = num_obstacles
        self.possible_agents = [f"drone_{i}" for i in range(num_drones)]
        self.agent_name_mapping = {agent: i for i, agent in enumerate(self.possible_agents)}
        self.action_spaces = {agent: spaces.Discrete(9) for agent in self.possible_agents}  # 8 directions + stay
        self.observation_spaces = {agent: spaces.Box(low=0, high=1, shape=(self.width, self.height, 3), dtype=np.float32) for agent in self.possible_agents}
        self.reset()

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()
        self.drones = []
        self.targets = []
        self.obstacles = []
        occupied = set()
        # Place drones
        for _ in range(self.num_drones):
            while True:
                x, y = random.randint(0, self.width-1), random.randint(0, self.height-1)
                if (x, y) not in occupied:
                    self.drones.append([x, y])
                    occupied.add((x, y))
                    break
        # Place targets
        for _ in range(self.num_targets):
            while True:
                x, y = random.randint(0, self.width-1), random.randint(0, self.height//2)
                if (x, y) not in occupied:
                    self.targets.append({'x': x, 'y': y, 'reached': False})
                    occupied.add((x, y))
                    break
        # Place obstacles
        for _ in range(self.num_obstacles):
            while True:
                x, y = random.randint(0, self.width-1), random.randint(0, self.height-1)
                if (x, y) not in occupied:
                    self.obstacles.append((x, y))
                    occupied.add((x, y))
                    break
        self.timestep = 0
        self.rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
        self._update_obs()
        return self._obs()

    def _update_obs(self):
        self.observations = {}
        for i, agent in enumerate(self.possible_agents):
            obs = np.zeros((self.width, self.height, 3), dtype=np.float32)
            # Obstacles
            for ox, oy in self.obstacles:
                obs[ox, oy, 0] = 1.0
            # Targets
            for t in self.targets:
                if not t['reached']:
                    obs[t['x'], t['y'], 1] = 1.0
            # Drone
            x, y = self.drones[i]
            obs[x, y, 2] = 1.0
            self.observations[agent] = obs

    def _obs(self):
        return {agent: self.observations[agent] for agent in self.agents}

    def step(self, action):
        agent = self.agent_selection
        idx = self.agent_name_mapping[agent]
        x, y = self.drones[idx]
        # Action: 0=stay, 1=up, 2=down, 3=left, 4=right, 5=up-left, 6=up-right, 7=down-left, 8=down-right
        moves = [(0,0), (0,-1), (0,1), (-1,0), (1,0), (-1,-1), (1,-1), (-1,1), (1,1)]
        dx, dy = moves[action]
        nx, ny = x + dx, y + dy
        # Check bounds and obstacles
        if 0 <= nx < self.width and 0 <= ny < self.height and (nx, ny) not in self.obstacles:
            self.drones[idx] = [nx, ny]
        # Check for reaching targets
        reward = 0.0
        for t in self.targets:
            if not t['reached'] and self.drones[idx][0] == t['x'] and self.drones[idx][1] == t['y']:
                t['reached'] = True
                reward += 1.0
        self.rewards[agent] = reward
        self._update_obs()
        # Terminate if all targets reached
        done = all(t['reached'] for t in self.targets)
        self.terminations[agent] = done
        self.truncations[agent] = False
        self.agent_selection = self._agent_selector.next()
        self.timestep += 1

    def render(self):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle
        plt.clf()
        ax = plt.gca()
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        # Obstacles
        for ox, oy in self.obstacles:
            ax.add_patch(plt.Rectangle((ox, oy), 1, 1, color='black'))
        # Targets
        for t in self.targets:
            color = 'green' if t['reached'] else 'red'
            ax.add_patch(Circle((t['x'] + 0.5, t['y'] + 0.5), 0.4, color=color, alpha=0.7))
        # Drones
        for x, y in self.drones:
            ax.add_patch(Circle((x + 0.5, y + 0.5), 0.3, color='blue', alpha=0.8))
        plt.title(f"Timestep: {self.timestep}")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.pause(0.3)

    def close(self):
        pass
