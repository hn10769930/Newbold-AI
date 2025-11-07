#Hailey Newbold
#CSC 362 D01
#11/06/2025
#HW4: Problem 1

import networkx as nx
import matplotlib.pyplot as plt

# -------------------------------
# Nim Game Logic
# -------------------------------
def get_moves(state):
    moves = []
    for i, heap in enumerate(state):
        for take in range(1, heap + 1):
            new_state = list(state)
            new_state[i] -= take
            moves.append(tuple(new_state))
    return moves

# -------------------------------
# Build Game Tree
# -------------------------------
def build_tree(state, player, G, parent=None):
    node = (state, player, id(parent))
    G.add_node(node)
    if parent:
        G.add_edge(parent, node)
    if state == (0,0):
        return node
    next_player = 2 if player == 1 else 1
    for move in get_moves(state):
        build_tree(move, next_player, G, node)
    return node

# -------------------------------
# Hierarchical Layout (Tree)
# -------------------------------
def hierarchy_pos(G, root, width=1., vert_gap=1., xcenter=0.5):
    """Compute hierarchical positions for a tree graph"""
    def _hierarchy_pos(G, node, width, vert_gap, xcenter, pos, level=0):
        children = list(G.successors(node))
        if len(children) == 0:
            pos[node] = (xcenter, -level)
        else:
            dx = width / len(children)
            nextx = xcenter - width/2 - dx/2
            for child in children:
                nextx += dx
                _hierarchy_pos(G, child, dx, vert_gap, nextx, pos, level+1)
            pos[node] = (xcenter, -level)
        return pos
    return _hierarchy_pos(G, root, width, vert_gap, xcenter, {})

# -------------------------------
# Console Tree Print
# -------------------------------
def print_tree(state, player, prefix=""):
    """Readable console tree"""
    print(f"{prefix}{state} - P{player}")
    if state == (0,0):
        return
    next_player = 2 if player == 1 else 1
    moves = get_moves(state)
    for i, move in enumerate(moves):
        if i == len(moves) - 1:
            branch = "└─ "
            next_prefix = prefix + "   "
        else:
            branch = "├─ "
            next_prefix = prefix + "│  "
        print_tree(move, next_player, prefix=next_prefix + branch)

# -------------------------------
# MAIN
# -------------------------------
initial_state = (3, 2)

# Console tree
print("\n=== READABLE CONSOLE NIM GAME TREE ===\n")
print_tree(initial_state, player=1)

# Graphical tree
G = nx.DiGraph()
root_node = build_tree(initial_state, player=1, G=G)
pos = hierarchy_pos(G, root_node)

color_map = ['lightblue' if n[1]==1 else 'lightgreen' for n in G.nodes()]

plt.figure(figsize=(12,8))
nx.draw(G, pos,
        with_labels=True,
        labels={n: f"{n[0]}-P{n[1]}" for n in G.nodes()},
        node_size=250,      # tiny nodes
        node_color=color_map,
        font_size=6,        # tiny font
        arrows=False)
plt.title("Readable Nim Game Tree (3,4)")
plt.show()
