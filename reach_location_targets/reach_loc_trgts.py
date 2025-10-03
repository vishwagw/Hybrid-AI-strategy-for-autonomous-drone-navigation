class Obstacle:
	def __init__(self, x, y):
		self.x = x
		self.y = y

import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def visualize(self):
		plt.clf()
		ax = plt.gca()
		ax.set_xlim(0, self.width)
		ax.set_ylim(0, self.height)
		# Draw targets
		for t in self.targets:
			color = 'green' if t.reached else 'red'
			ax.add_patch(Circle((t.x + 0.5, t.y + 0.5), 0.4, color=color, alpha=0.7, label='Target' if color=='red' else 'Reached'))
		# Draw drones
		for d in self.drones:
			ax.add_patch(Circle((d.x + 0.5, d.y + 0.5), 0.3, color='blue', alpha=0.8, label='Drone'))
		plt.title(f"Timestep: {self.timestep}")
		plt.xlabel("X")
		plt.ylabel("Y")
		plt.pause(0.3)

# Basic simulation for drones reaching static/moving targets (no obstacles)
import random
import math

class Target:
	def __init__(self, x, y):
		self.x = x
		self.y = y
		self.reached = False

	def step(self):
		pass  # Static targets do not move

class Drone:
	def __init__(self, x, y, env, vision_range=5):
		self.x = x
		self.y = y
		self.env = env
		self.vision_range = vision_range

	def local_observation(self):
		# Returns targets within vision range
		obs = []
		for t in self.env.targets:
			if not t.reached:
				dist = math.hypot(self.x - t.x, self.y - t.y)
				if dist <= self.vision_range:
					obs.append((t.x, t.y))
		return obs


	def move_toward(self, tx, ty):
		# Move one step toward (tx, ty), avoiding obstacles
		best_move = (self.x, self.y)
		min_dist = math.hypot(self.x - tx, self.y - ty)
		# Try all 8 directions and stay
		for dx in [-1, 0, 1]:
			for dy in [-1, 0, 1]:
				nx, ny = self.x + dx, self.y + dy
				if (dx == 0 and dy == 0):
					continue
				if not self.env.is_valid(nx, ny):
					continue
				if self.env.is_obstacle(nx, ny):
					continue
				dist = math.hypot(nx - tx, ny - ty)
				if dist < min_dist:
					min_dist = dist
					best_move = (nx, ny)
		self.x, self.y = best_move

	def step(self):
		# Simple rule: move toward nearest unreached target
		min_dist = float('inf')
		nearest = None
		for t in self.env.targets:
			if not t.reached:
				dist = math.hypot(self.x - t.x, self.y - t.y)
				if dist < min_dist:
					min_dist = dist
					nearest = t
		if nearest:
			self.move_toward(nearest.x, nearest.y)
			# Check if reached
			if self.x == nearest.x and self.y == nearest.y:
				nearest.reached = True

class Environment:

	def __init__(self, width, height, num_drones, num_targets, num_obstacles):
		self.width = width
		self.height = height
		self.drones = [Drone(random.randint(0, width-1), random.randint(0, height-1), self) for _ in range(num_drones)]
		self.targets = []
		self.obstacles = []
		occupied = set()
		# Place targets
		while len(self.targets) < num_targets:
			x, y = random.randint(0, width-1), random.randint(0, height//2)
			if (x, y) not in occupied:
				self.targets.append(Target(x, y))
				occupied.add((x, y))
		# Place obstacles
		while len(self.obstacles) < num_obstacles:
			x, y = random.randint(0, width-1), random.randint(0, height-1)
			if (x, y) not in occupied:
				self.obstacles.append(Obstacle(x, y))
				occupied.add((x, y))
		self.timestep = 0

	def is_obstacle(self, x, y):
		return any(o.x == x and o.y == y for o in self.obstacles)

	def is_valid(self, x, y):
		return 0 <= x < self.width and 0 <= y < self.height

	def step(self):
		for t in self.targets:
			t.step()
		for d in self.drones:
			d.step()
		self.timestep += 1

	def all_targets_reached(self):
		return all(t.reached for t in self.targets)

	def print_state(self):
		print(f"Timestep: {self.timestep}")
		for i, d in enumerate(self.drones):
			print(f"Drone {i}: ({d.x}, {d.y})")
		for i, t in enumerate(self.targets):
			status = 'reached' if t.reached else 'active'
			print(f"Target {i}: ({t.x}, {t.y}) [{status}]")

	def visualize(self):
		plt.clf()
		ax = plt.gca()
		ax.set_xlim(0, self.width)
		ax.set_ylim(0, self.height)
		# Draw obstacles
		for o in self.obstacles:
			ax.add_patch(plt.Rectangle((o.x, o.y), 1, 1, color='black'))
		# Draw targets
		for t in self.targets:
			color = 'green' if t.reached else 'red'
			ax.add_patch(Circle((t.x + 0.5, t.y + 0.5), 0.4, color=color, alpha=0.7))
		# Draw drones
		for d in self.drones:
			ax.add_patch(Circle((d.x + 0.5, d.y + 0.5), 0.3, color='blue', alpha=0.8))
		plt.title(f"Timestep: {self.timestep}")
		plt.xlabel("X")
		plt.ylabel("Y")
		plt.pause(0.3)

if __name__ == "__main__":
	env = Environment(width=20, height=20, num_drones=2, num_targets=5, num_obstacles=5)
	max_steps = 50
	plt.ion()
	fig = plt.figure(figsize=(6, 6))
	for _ in range(max_steps):
		env.visualize()
		if env.all_targets_reached():
			print("All targets reached!")
			break
		env.step()
	else:
		print("Simulation ended. Not all targets were reached.")
	plt.ioff()
	plt.show()
# for agent (drone) reaching located targets in the simulation environment.
