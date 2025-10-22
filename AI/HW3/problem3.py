#Hailey Newbold
#CSC 362 D01
#10/21/2025
#HW3: Problem 3

import math, random, time
import matplotlib.pyplot as plt
from queue import PriorityQueue
import numpy as np

# Cell structure
class Cell:
    def __init__(self, x, y, is_wall=False):
        self.x, self.y = x, y
        self.is_wall = is_wall
        self.g = float("inf")
        self.h = 0
        self.f = float("inf")
        self.parent = None

    def __lt__(self, other):
        return self.f < other.f

# Weighted A* implementation
def weighted_a_star(maze, alpha=1, beta=1):
    rows, cols = len(maze), len(maze[0])
    cells = [[Cell(i, j, maze[i][j] == 1) for j in range(cols)] for i in range(rows)]
    start, goal = (0, 0), (rows - 1, cols - 1)

    def heuristic(a, b):
        dx, dy = a[0] - b[0], a[1] - b[1]
        return math.sqrt(dx * dx + dy * dy)

    start_cell = cells[start[0]][start[1]]
    start_cell.g = 0
    start_cell.h = heuristic(start, goal)
    start_cell.f = alpha * start_cell.g + beta * start_cell.h

    open_set = PriorityQueue()
    open_set.put((start_cell.f, start))

    while not open_set.empty():
        _, (x, y) = open_set.get()
        current = cells[x][y]

        if (x, y) == goal:
            break

        for dx, dy in [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols and not cells[nx][ny].is_wall:
                cost = math.sqrt(dx*dx + dy*dy)
                new_g = current.g + cost
                if new_g < cells[nx][ny].g:
                    neighbor = cells[nx][ny]
                    neighbor.g = new_g
                    neighbor.h = heuristic((nx, ny), goal)
                    neighbor.f = alpha * neighbor.g + beta * neighbor.h
                    neighbor.parent = current
                    open_set.put((neighbor.f, (nx, ny)))

    # Reconstruct path
    path = set()
    current = cells[goal[0]][goal[1]]
    while current and current.parent:
        path.add((current.x, current.y))
        current = current.parent
    return path

# Maze generation
def generate_maze(rows=100, cols=100, wall_prob=0.25):
    maze = [[1 if random.random() < wall_prob else 0 for _ in range(cols)] for _ in range(rows)]
    maze[0][0] = 0
    maze[rows-1][cols-1] = 0
    return maze

# Visualization
def visualize_maze(maze, path, alpha, beta):
    rows, cols = len(maze), len(maze[0])
    grid = np.zeros((rows, cols, 3))

    for i in range(rows):
        for j in range(cols):
            if maze[i][j] == 1:
                grid[i, j] = [0, 0, 0]  # wall = black
            else:
                grid[i, j] = [1, 1, 1]  # free = white
    for (x, y) in path:
        grid[x, y] = [1, 0, 0]  # path = red

    grid[0, 0] = [0, 1, 0]  # start = green
    grid[-1, -1] = [0, 0, 1]  # goal = blue

    plt.figure(figsize=(6, 6))
    plt.imshow(grid, interpolation="nearest")
    plt.title(f"Weighted A* (α={alpha}, β={beta})")
    plt.axis("off")
    plt.savefig(f"weighted_astar_alpha{alpha}_beta{beta}.png", bbox_inches='tight')
    plt.show()

# MAIN
maze = generate_maze(100, 100, wall_prob=0.35)

# Try different (α, β) values
params = [(1, 1), (2, 1), (1, 2), (0.5, 2), (2, 0.5)]
for alpha, beta in params:
    print(f"\nRunning A* with α={alpha}, β={beta}")
    start_time = time.time()
    path = weighted_a_star(maze, alpha, beta)
    end_time = time.time()
    print(f"Path length: {len(path)} | Time: {end_time - start_time:.3f}s")
    visualize_maze(maze, path, alpha, beta)