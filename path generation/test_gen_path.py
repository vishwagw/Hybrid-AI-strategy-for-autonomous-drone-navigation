
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import time

WIDTH, HEIGHT = 15, 15
OBSTACLES = {(5, 5), (5, 6), (5, 7), (7, 5), (8, 5), (9, 5)}
drone_pos = (2, 2)
target_pos = (12, 12)

def is_valid(x, y):
    return 0 <= x < WIDTH and 0 <= y < HEIGHT and (x, y) not in OBSTACLES

def get_neighbors(pos):
    x, y = pos
    moves = [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (1,1), (-1,1), (1,-1)]
    return [(x+dx, y+dy) for dx, dy in moves if is_valid(x+dx, y+dy)]

# BFS for shortest path
def bfs(start, goal):
    queue = deque([[start]])
    visited = set([start])
    while queue:
        path = queue.popleft()
        node = path[-1]
        if node == goal:
            return path
        for neighbor in get_neighbors(node):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(path + [neighbor])
    return None

# DFS for all paths (with a reasonable limit)
def dfs_all_paths(start, goal, max_paths=20, max_depth=30):
    stack = [([start], set([start]))]
    paths = []
    while stack and len(paths) < max_paths:
        path, visited = stack.pop()
        node = path[-1]
        if node == goal:
            paths.append(list(path))
            continue
        if len(path) > max_depth:
            continue
        for neighbor in get_neighbors(node):
            if neighbor not in visited:
                stack.append((path + [neighbor], visited | {neighbor}))
    return paths

def plot_env(drone, target, obstacles, all_paths=None, shortest_path=None, step=None, show_all_paths=False):
    plt.clf()
    ax = plt.gca()
    ax.set_xlim(-0.5, WIDTH-0.5)
    ax.set_ylim(-0.5, HEIGHT-0.5)
    for x in range(WIDTH):
        for y in range(HEIGHT):
            rect = plt.Rectangle((x-0.5, y-0.5), 1, 1, edgecolor='gray', facecolor='none', lw=0.5)
            ax.add_patch(rect)
    for (ox, oy) in obstacles:
        ax.add_patch(plt.Rectangle((ox-0.5, oy-0.5), 1, 1, color='black'))
    # Draw all alternative paths in red
    if show_all_paths and all_paths:
        for path in all_paths:
            xs, ys = zip(*path)
            plt.plot(xs, ys, color='red', alpha=0.3, lw=2)
    # Draw shortest path in green
    if shortest_path:
        xs, ys = zip(*shortest_path)
        plt.plot(xs, ys, color='green', lw=3, label='Shortest Path')
    plt.plot(target[0], target[1], 'go', markersize=12, label='Target')
    plt.plot(drone[0], drone[1], 'bo', markersize=12, label='Drone')
    if step is not None:
        plt.title(f'Drone moving to target (Step {step})')
    else:
        plt.title('Drone Path Planning')
    plt.legend()
    plt.pause(0.5)

if __name__ == "__main__":
    all_paths = dfs_all_paths(drone_pos, target_pos, max_paths=20, max_depth=30)
    shortest_path = bfs(drone_pos, target_pos)
    if not shortest_path:
        print("No path found!")
    else:
        plt.ion()
        # First show all alternative paths in red
        plot_env(drone_pos, target_pos, OBSTACLES, all_paths=all_paths, shortest_path=None, show_all_paths=True)
        time.sleep(2)
        # Then animate the drone moving along the shortest path in green
        for i, pos in enumerate(shortest_path):
            plot_env(pos, target_pos, OBSTACLES, all_paths=None, shortest_path=shortest_path[:i+1], step=i)
            time.sleep(0.2)
        plt.ioff()
        plt.show()
        print("Drone reached the target!")
