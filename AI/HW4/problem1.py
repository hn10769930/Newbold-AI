#Hailey Newbold
#CSC 362 D01
#11/06/2025
#HW4: Problem 1

import networkx as nx
import matplotlib.pyplot as plt
from collections import deque

class Nim:
    def __init__(self, initial_state):
        self.initial_state = initial_state
    
    def get_moves(self, state):
        """Generate all possible moves from current state"""
        moves = []
        for i, heap in enumerate(state):
            for take in range(1, heap + 1):
                new_state = list(state)
                new_state[i] -= take
                moves.append((tuple(new_state), i, take))  # Include heap index and amount taken
        return moves
    
    def is_terminal(self, state):
        """Check if game is over"""
        return all(h == 0 for h in state)
    
    def evaluate(self, state, is_max):
        """Evaluate terminal state"""
        if all(h == 0 for h in state):
            return -1 if is_max else 1  # Previous player took last object
        return 0

def build_complete_game_tree(initial_state):
    """Build complete game tree for analysis with move information"""
    nim = Nim(initial_state)
    G = nx.DiGraph()
    visited = set()
    queue = deque([(initial_state, None, None, None)])  # (state, parent, heap, taken)
    
    while queue:
        current_state, parent, heap_idx, taken = queue.popleft()
        
        if current_state in visited:
            continue
        visited.add(current_state)
        
        # Add node with player info
        depth = sum(initial_state) - sum(current_state)
        player = "P1" if depth % 2 == 0 else "P2"
        
        # Store move information safely
        move_info = (heap_idx, taken) if heap_idx is not None and taken is not None else (None, None)
        
        G.add_node(current_state, 
                  player=player, 
                  value=None, 
                  terminal=nim.is_terminal(current_state),
                  move_from_parent=move_info,
                  depth=depth)
        
        if parent:
            G.add_edge(parent, current_state, move=move_info)
        
        # Add children if not terminal
        if not nim.is_terminal(current_state):
            moves = nim.get_moves(current_state)
            for move, h_idx, tkn in moves:
                if move not in visited:
                    queue.append((move, current_state, h_idx, tkn))
    
    return G

def minimax_with_tracking(state, is_max, G, parent=None, memo={}, depth=0):
    """Minimax with value tracking for analysis"""
    if G.nodes[state].get('terminal', False):
        value = -1 if is_max else 1
        G.nodes[state]['value'] = value
        G.nodes[state]['minimax_value'] = value
        return value
    
    if state in memo:
        return memo[state]
    
    moves = [move for move, _, _ in Nim(state).get_moves(state)]
    if is_max:
        value = float('-inf')
        best_move = None
        for move in moves:
            child_value = minimax_with_tracking(move, False, G, state, memo, depth+1)
            if child_value > value:
                value = child_value
                best_move = move
        if best_move:
            G.nodes[state]['best_move'] = best_move
            G.nodes[state]['minimax_chosen'] = True
    else:
        value = float('inf')
        best_move = None
        for move in moves:
            child_value = minimax_with_tracking(move, True, G, state, memo, depth+1)
            if child_value < value:
                value = child_value
                best_move = move
        if best_move:
            G.nodes[state]['best_move'] = best_move
            G.nodes[state]['minimax_chosen'] = True
    
    G.nodes[state]['value'] = value
    G.nodes[state]['minimax_value'] = value
    memo[state] = value
    return value

def alphabeta_with_pruning_tracking(state, is_max, G, parent=None, alpha=float('-inf'), 
                                  beta=float('inf'), memo={}, pruned_nodes=set(), depth=0):
    """Alpha-beta with pruning tracking for analysis"""
    if G.nodes[state].get('terminal', False):
        value = -1 if is_max else 1
        G.nodes[state]['value'] = value
        G.nodes[state]['alpha'] = alpha
        G.nodes[state]['beta'] = beta
        G.nodes[state]['alphabeta_value'] = value
        return value
    
    state_key = (state, alpha, beta)
    if state_key in memo:
        return memo[state_key]
    
    moves_info = Nim(state).get_moves(state)
    moves = [move for move, _, _ in moves_info]
    G.nodes[state]['alpha'] = alpha
    G.nodes[state]['beta'] = beta
    
    if is_max:
        value = float('-inf')
        best_move = None
        for i, (move, heap_idx, taken) in enumerate(moves_info):
            G.nodes[move]['evaluated'] = True
            child_value = alphabeta_with_pruning_tracking(move, False, G, state, alpha, beta, memo, pruned_nodes, depth+1)
            
            if child_value > value:
                value = child_value
                best_move = move
                G.nodes[state]['alphabeta_chosen'] = True
                G.nodes[state]['best_move_ab'] = move
            
            alpha = max(alpha, value)
            G.nodes[state]['alpha'] = alpha
            
            if alpha >= beta:
                # Mark remaining moves as pruned
                for j in range(i+1, len(moves_info)):
                    pruned_move, _, _ = moves_info[j]
                    pruned_nodes.add(pruned_move)
                    G.nodes[pruned_move]['pruned'] = True
                    G.nodes[pruned_move]['value'] = 'Pruned'
                break
    else:
        value = float('inf')
        best_move = None
        for i, (move, heap_idx, taken) in enumerate(moves_info):
            G.nodes[move]['evaluated'] = True
            child_value = alphabeta_with_pruning_tracking(move, True, G, state, alpha, beta, memo, pruned_nodes, depth+1)
            
            if child_value < value:
                value = child_value
                best_move = move
                G.nodes[state]['alphabeta_chosen'] = True
                G.nodes[state]['best_move_ab'] = move
            
            beta = min(beta, value)
            G.nodes[state]['beta'] = beta
            
            if beta <= alpha:
                # Mark remaining moves as pruned
                for j in range(i+1, len(moves_info)):
                    pruned_move, _, _ = moves_info[j]
                    pruned_nodes.add(pruned_move)
                    G.nodes[pruned_move]['pruned'] = True
                    G.nodes[pruned_move]['value'] = 'Pruned'
                break
    
    G.nodes[state]['value'] = value
    G.nodes[state]['alphabeta_value'] = value
    memo[state_key] = value
    return value

def print_complete_console_tree(G, initial_state, algorithm_type="minimax"):
    """Print the complete game tree in console with visual indicators"""
    print("\n" + "=" * 100)
    print(f"COMPLETE GAME TREE CONSOLE VIEW - {algorithm_type.upper()}")
    print("=" * 100)
    
    # Organize nodes by depth
    nodes_by_depth = {}
    for node in G.nodes():
        depth = G.nodes[node]['depth']
        nodes_by_depth.setdefault(depth, []).append(node)
    
    # Print tree level by level
    max_depth = max(nodes_by_depth.keys())
    
    for depth in sorted(nodes_by_depth.keys()):
        print(f"\n--- LEVEL {depth} ({G.nodes[nodes_by_depth[depth][0]]['player']}'s turn) ---")
        nodes_at_depth = sorted(nodes_by_depth[depth])
        
        for node in nodes_at_depth:
            # Get node information
            player = G.nodes[node]['player']
            value = G.nodes[node].get('value', '?')
            terminal = G.nodes[node].get('terminal', False)
            pruned = G.nodes[node].get('pruned', False)
            move_info = G.nodes[node].get('move_from_parent', (None, None))
            heap_idx, taken = move_info if move_info != (None, None) else (None, None)
            
            # Determine visual indicators
            indicator = " "
            if algorithm_type == "minimax" and G.nodes[node].get('minimax_chosen', False):
                indicator = "✓"  # Tick for chosen in minimax
            elif algorithm_type == "alphabeta" and G.nodes[node].get('alphabeta_chosen', False):
                indicator = "✓"  # Tick for chosen in alpha-beta
            elif pruned:
                indicator = "✗"  # X for pruned
            
            # Build node display
            move_str = f" (H{heap_idx}-{taken})" if heap_idx is not None and taken is not None else ""
            
            if terminal:
                node_display = f"{indicator} TERMINAL {node} → Value: {value}"
            elif pruned:
                node_display = f"{indicator} PRUNED {node}{move_str}"
            else:
                value_display = f"Value: {value}" if value != '?' else "Value: ?"
                node_display = f"{indicator} {node}{move_str} → {value_display}"
            
            print(f"  {node_display}")
            
            # Print children with indentation
            children = list(G.successors(node))
            if children:
                print("    Children:")
                for child in sorted(children):
                    child_move_info = G.edges[node, child].get('move', (None, None))
                    child_heap, child_taken = child_move_info if child_move_info != (None, None) else (None, None)
                    child_indicator = " "
                    
                    if algorithm_type == "minimax" and G.nodes[child].get('minimax_chosen', False):
                        child_indicator = "✓"
                    elif algorithm_type == "alphabeta" and G.nodes[child].get('alphabeta_chosen', False):
                        child_indicator = "✓"
                    elif G.nodes[child].get('pruned', False):
                        child_indicator = "✗"
                    
                    move_str = f" (H{child_heap}-{child_taken})" if child_heap is not None and child_taken is not None else ""
                    print(f"      {child_indicator} {child}{move_str}")

def print_optimal_path(G, initial_state, algorithm_type="minimax"):
    """Print the optimal path through the game tree"""
    print(f"\n{'='*60}")
    print(f"OPTIMAL PATH - {algorithm_type.upper()}")
    print(f"{'='*60}")
    
    current_state = initial_state
    path = []
    max_moves = 20
    
    for i in range(max_moves):
        if current_state not in G.nodes:
            break
            
        if algorithm_type == "minimax":
            best_move = G.nodes[current_state].get('best_move')
        else:
            best_move = G.nodes[current_state].get('best_move_ab')
            
        if not best_move:
            break
            
        player = G.nodes[current_state]['player']
        move_info = G.nodes[best_move].get('move_from_parent', (None, None))
        heap_idx, taken = move_info if move_info != (None, None) else (None, None)
        value = G.nodes[best_move].get('value', '?')
        
        if heap_idx is not None and taken is not None:
            print(f"  {player} at {current_state} → Remove {taken} from heap {heap_idx} → {best_move} (Value: {value})")
        else:
            print(f"  {player} at {current_state} → {best_move} (Value: {value})")
            
        path.append((current_state, best_move, heap_idx, taken))
        current_state = best_move
        
        if G.nodes[current_state].get('terminal', False):
            print(f"  {G.nodes[current_state]['player']} at {current_state} → TERMINAL (Game Over)")
            break
    
    return path

def print_algorithm_statistics(G, pruned_nodes, algorithm_type="minimax"):
    """Print statistics about the algorithm performance"""
    print(f"\n{'='*60}")
    print(f"ALGORITHM STATISTICS - {algorithm_type.upper()}")
    print(f"{'='*60}")
    
    total_nodes = len(G.nodes())
    
    if algorithm_type == "minimax":
        evaluated_nodes = total_nodes
        pruned_count = 0
    else:
        evaluated_nodes = len([n for n in G.nodes() if G.nodes[n].get('evaluated', False) or G.nodes[n].get('terminal', False)])
        pruned_count = len(pruned_nodes)
    
    chosen_nodes = len([n for n in G.nodes() if G.nodes[n].get('minimax_chosen', False) or G.nodes[n].get('alphabeta_chosen', False)])
    terminal_nodes = len([n for n in G.nodes() if G.nodes[n].get('terminal', False)])
    
    print(f"Total nodes in game tree: {total_nodes}")
    print(f"Nodes evaluated: {evaluated_nodes}")
    print(f"Nodes pruned: {pruned_count}")
    print(f"Optimal path nodes: {chosen_nodes}")
    print(f"Terminal nodes: {terminal_nodes}")
    
    if algorithm_type == "alphabeta":
        efficiency = (pruned_count / total_nodes) * 100 if total_nodes > 0 else 0
        print(f"Pruning efficiency: {efficiency:.1f}%")

def main():
    initial_state = (3, 4)  # Using (3,4) as requested
    
    print("COMPREHENSIVE NIM GAME ANALYSIS - STATE (3, 4)")
    print("=" * 80)
    
    # Build complete game tree
    print("Building complete game tree...")
    G = build_complete_game_tree(initial_state)
    
    # MINIMAX ANALYSIS
    print("\n" + "="*80)
    print("MINIMAX ALGORITHM ANALYSIS")
    print("="*80)
    
    G_minimax = build_complete_game_tree(initial_state)
    minimax_value = minimax_with_tracking(initial_state, True, G_minimax)
    
    print(f"Root node value: {minimax_value}")
    print(f"Optimal outcome: {'Player 1 wins' if minimax_value == 1 else 'Player 2 wins' if minimax_value == -1 else 'Draw'}")
    
    # Print complete console tree for Minimax
    print_complete_console_tree(G_minimax, initial_state, "minimax")
    
    # Print optimal path for Minimax
    minimax_path = print_optimal_path(G_minimax, initial_state, "minimax")
    
    # Print Minimax statistics
    print_algorithm_statistics(G_minimax, set(), "minimax")
    
    # ALPHA-BETA ANALYSIS
    print("\n" + "="*80)
    print("ALPHA-BETA PRUNING ANALYSIS")
    print("="*80)
    
    G_alphabeta = build_complete_game_tree(initial_state)
    pruned_nodes = set()
    alphabeta_memo = {}
    alphabeta_value = alphabeta_with_pruning_tracking(initial_state, True, G_alphabeta, 
                                                    alpha=float('-inf'), beta=float('inf'),
                                                    memo=alphabeta_memo, pruned_nodes=pruned_nodes)
    
    print(f"Root node value: {alphabeta_value}")
    print(f"Optimal outcome: {'Player 1 wins' if alphabeta_value == 1 else 'Player 2 wins' if alphabeta_value == -1 else 'Draw'}")
    
    # Print complete console tree for Alpha-Beta
    print_complete_console_tree(G_alphabeta, initial_state, "alphabeta")
    
    # Print optimal path for Alpha-Beta
    alphabeta_path = print_optimal_path(G_alphabeta, initial_state, "alphabeta")
    
    # Print Alpha-Beta statistics
    print_algorithm_statistics(G_alphabeta, pruned_nodes, "alphabeta")
    
    # COMPARISON
    print("\n" + "="*80)
    print("ALGORITHM COMPARISON")
    print("="*80)
    
    total_nodes = len(G_minimax.nodes())
    minimax_evaluated = total_nodes
    alphabeta_evaluated = len([n for n in G_alphabeta.nodes() if G_alphabeta.nodes[n].get('evaluated', False) or G_alphabeta.nodes[n].get('terminal', False)])
    
    print(f"Minimax nodes evaluated: {minimax_evaluated}")
    print(f"Alpha-Beta nodes evaluated: {alphabeta_evaluated}")
    print(f"Reduction: {minimax_evaluated - alphabeta_evaluated} nodes ({((minimax_evaluated - alphabeta_evaluated) / minimax_evaluated * 100):.1f}%)")
    
    # Verify both algorithms found the same optimal value
    if minimax_value == alphabeta_value:
        print("Both algorithms found the same optimal value")
    else:
        print("Algorithms found different values - this should not happen!")

if __name__ == "__main__":
    main()