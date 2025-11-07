#Hailey Newbold
#CSC 362 D01
#11/06/2025
#HW4: Problem 2

import networkx as nx
import matplotlib.pyplot as plt
import time
from collections import deque
import math

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
                moves.append(tuple(new_state))
        return moves
    
    def is_terminal(self, state):
        """Check if game is over"""
        return all(h == 0 for h in state)
    
    def evaluate_terminal(self, state, is_max):
        """Evaluate terminal state"""
        if all(h == 0 for h in state):
            return -1 if is_max else 1
        return 0

class OptimizedNimSolver:
    def __init__(self, initial_state):
        self.initial_state = initial_state
        self.nim = Nim(initial_state)
        self.transposition_table = {}
        self.nodes_evaluated = 0
    
    def heuristic_evaluation(self, state, is_max):
        """
        Heuristic evaluation function for Nim
        Based on mathematical properties of Nim (XOR strategy)
        """
        if self.nim.is_terminal(state):
            return -1 if is_max else 1
        
        # Calculate nim-sum (XOR of all heap sizes)
        nim_sum = 0
        for heap in state:
            nim_sum ^= heap
        
        # If nim-sum is 0, it's a losing position for current player
        if nim_sum == 0:
            return -0.5 if is_max else 0.5
        
        # Winning position - estimate how good based on number of winning moves
        winning_moves = 0
        moves = self.nim.get_moves(state)
        for move in moves:
            move_nim_sum = 0
            for heap in move:
                move_nim_sum ^= heap
            if move_nim_sum == 0:  # This move creates losing position for opponent
                winning_moves += 1
        
        # Normalize winning moves count to range [-1, 1]
        max_possible_moves = sum(state)
        heuristic_value = winning_moves / max(max_possible_moves, 1)
        
        return heuristic_value if is_max else -heuristic_value
    
    def standard_minimax(self, state, is_max, depth, max_depth):
        """Standard minimax without optimizations"""
        self.nodes_evaluated += 1
        
        if self.nim.is_terminal(state):
            return self.nim.evaluate_terminal(state, is_max)
        
        if depth >= max_depth:
            return self.heuristic_evaluation(state, is_max)
        
        moves = self.nim.get_moves(state)
        if is_max:
            best_value = float('-inf')
            for move in moves:
                value = self.standard_minimax(move, False, depth + 1, max_depth)
                best_value = max(best_value, value)
            return best_value
        else:
            best_value = float('inf')
            for move in moves:
                value = self.standard_minimax(move, True, depth + 1, max_depth)
                best_value = min(best_value, value)
            return best_value
    
    def minimax_with_transposition(self, state, is_max, depth, max_depth):
        """Minimax with transposition table"""
        state_key = (state, is_max, depth)
        
        if state_key in self.transposition_table:
            return self.transposition_table[state_key]
        
        self.nodes_evaluated += 1
        
        if self.nim.is_terminal(state):
            value = self.nim.evaluate_terminal(state, is_max)
            self.transposition_table[state_key] = value
            return value
        
        if depth >= max_depth:
            value = self.heuristic_evaluation(state, is_max)
            self.transposition_table[state_key] = value
            return value
        
        moves = self.nim.get_moves(state)
        if is_max:
            best_value = float('-inf')
            for move in moves:
                value = self.minimax_with_transposition(move, False, depth + 1, max_depth)
                best_value = max(best_value, value)
            self.transposition_table[state_key] = best_value
            return best_value
        else:
            best_value = float('inf')
            for move in moves:
                value = self.minimax_with_transposition(move, True, depth + 1, max_depth)
                best_value = min(best_value, value)
            self.transposition_table[state_key] = best_value
            return best_value
    
    def iterative_deepening_minimax(self, max_depth, use_transposition=True):
        """
        Iterative deepening minimax search
        Returns value and search statistics
        """
        print(f"\nPerforming iterative deepening up to depth {max_depth}...")
        
        results = {}
        total_nodes = 0
        
        for depth in range(1, max_depth + 1):
            self.nodes_evaluated = 0
            self.transposition_table.clear()
            
            start_time = time.time()
            
            if use_transposition:
                value = self.minimax_with_transposition(self.initial_state, True, 0, depth)
            else:
                value = self.standard_minimax(self.initial_state, True, 0, depth)
            
            end_time = time.time()
            
            results[depth] = {
                'value': value,
                'nodes_evaluated': self.nodes_evaluated,
                'time_taken': end_time - start_time,
                'transposition_size': len(self.transposition_table)
            }
            
            total_nodes += self.nodes_evaluated
            
            print(f"Depth {depth}: value={value:.3f}, nodes={self.nodes_evaluated}, "
                  f"time={end_time - start_time:.4f}s, transposition_entries={len(self.transposition_table)}")
            
            # Early termination if we found a proven win/loss
            if abs(value) == 1:
                print(f"Terminal state found at depth {depth}, stopping early.")
                break
        
        return results
    
    def heuristic_minimax(self, state, is_max, depth, max_depth, use_heuristic=True):
        """Minimax using heuristic evaluation at cutoff"""
        self.nodes_evaluated += 1
        
        if self.nim.is_terminal(state):
            return self.nim.evaluate_terminal(state, is_max)
        
        if depth >= max_depth:
            if use_heuristic:
                return self.heuristic_evaluation(state, is_max)
            else:
                # Use simple evaluation if no heuristic
                return 0  # Assume neutral position
        
        moves = self.nim.get_moves(state)
        if is_max:
            best_value = float('-inf')
            for move in moves:
                value = self.heuristic_minimax(move, False, depth + 1, max_depth, use_heuristic)
                best_value = max(best_value, value)
            return best_value
        else:
            best_value = float('inf')
            for move in moves:
                value = self.heuristic_minimax(move, True, depth + 1, max_depth, use_heuristic)
                best_value = min(best_value, value)
            return best_value

def analyze_2_ply_game(initial_state):
    """Detailed analysis of 2-ply game for explanation"""
    print("=" * 70)
    print("2-PLY GAME ANALYSIS")
    print("=" * 70)
    
    solver = OptimizedNimSolver(initial_state)
    
    # Analyze initial state
    print(f"\nInitial State: {initial_state}")
    heuristic_value = solver.heuristic_evaluation(initial_state, True)
    print(f"Heuristic evaluation: {heuristic_value:.3f}")
    
    # Get all moves from initial state (1-ply)
    moves_1_ply = solver.nim.get_moves(initial_state)
    print(f"\n1-PLY MOVES (from initial state):")
    
    for i, move in enumerate(moves_1_ply):
        h_val = solver.heuristic_evaluation(move, False)
        print(f"  Move {i+1}: {initial_state} -> {move}, heuristic = {h_val:.3f}")
    
    # Analyze 2-ply moves
    print(f"\n2-PLY ANALYSIS:")
    for i, move_1 in enumerate(moves_1_ply[:3]):  # Show first 3 for brevity
        print(f"\nAfter move {i+1}: {initial_state} -> {move_1}")
        moves_2_ply = solver.nim.get_moves(move_1)
        
        for j, move_2 in enumerate(moves_2_ply[:2]):  # Show first 2 responses
            h_val = solver.heuristic_evaluation(move_2, True)
            terminal = " [TERMINAL]" if solver.nim.is_terminal(move_2) else ""
            print(f"  Response {j+1}: {move_1} -> {move_2}, heuristic = {h_val:.3f}{terminal}")

def compare_optimization_techniques(initial_state):
    """Compare different optimization techniques"""
    print("=" * 70)
    print("OPTIMIZATION TECHNIQUES COMPARISON")
    print("=" * 70)
    
    solver = OptimizedNimSolver(initial_state)
    
    # Test different approaches
    techniques = [
        ("Standard Minimax", lambda: solver.standard_minimax(initial_state, True, 0, 3)),
        ("Minimax + Heuristic", lambda: solver.heuristic_minimax(initial_state, True, 0, 3, True)),
        ("Minimax + Transposition", lambda: solver.minimax_with_transposition(initial_state, True, 0, 3)),
    ]
    
    results = []
    for name, func in techniques:
        solver.nodes_evaluated = 0
        solver.transposition_table.clear()
        
        start_time = time.time()
        value = func()
        end_time = time.time()
        
        results.append({
            'technique': name,
            'value': value,
            'nodes_evaluated': solver.nodes_evaluated,
            'time_taken': end_time - start_time
        })
        
        print(f"{name:25} | Value: {value:6.3f} | Nodes: {solver.nodes_evaluated:6d} | "
              f"Time: {end_time - start_time:.6f}s")
    
    return results

def visualize_optimization_comparison(results, initial_state):
    """
    Create visualization comparing optimization techniques
    CORRECTION: Titles adjusted to prevent overlap.
    """
    techniques = [r['technique'] for r in results]
    nodes = [r['nodes_evaluated'] for r in results]
    times = [r['time_taken'] * 1000 for r in results]  # Convert to milliseconds
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Main title is now set on the first subplot, with improved placement
    ax1.set_title(f'Nim Optimization Techniques Comparison (State: {initial_state}) - Nodes', 
                  fontsize=12, fontweight='bold', pad=20) 
    
    # Nodes evaluated comparison
    bars1 = ax1.bar(techniques, nodes, color=['lightcoral', 'lightblue', 'lightgreen'])
    ax1.set_ylabel('Number of Nodes')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, node_count in zip(bars1, nodes):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(nodes)*0.01,
                 f'{node_count}', ha='center', va='bottom')
    
    # Time taken comparison
    ax2.set_title(f'Time Comparison (State: {initial_state})', fontsize=12, pad=20)
    bars2 = ax2.bar(techniques, times, color=['lightcoral', 'lightblue', 'lightgreen'])
    ax2.set_ylabel('Time (milliseconds)')
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, time_taken in zip(bars2, times):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(times)*0.01,
                 f'{time_taken:.2f}ms', ha='center', va='bottom')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust layout to ensure space for titles
    plt.show()

def demonstrate_iterative_deepening(initial_state):
    """
    Demonstrate iterative deepening with analysis
    CORRECTION: Titles adjusted to prevent overlap.
    """
    print("=" * 70)
    print("ITERATIVE DEEPENING DEMONSTRATION")
    print("=" * 70)
    
    solver = OptimizedNimSolver(initial_state)
    
    print(f"Initial state: {initial_state}")
    print("\nPerforming iterative deepening search...")
    
    # Run iterative deepening
    results = solver.iterative_deepening_minimax(max_depth=4)
    
    # Analyze results
    print(f"\nITERATIVE DEEPENING ANALYSIS:")
    print(f"Total nodes evaluated across all depths: {sum(r['nodes_evaluated'] for r in results.values())}")
    
    # Show progression
    depths = list(results.keys())
    nodes_per_depth = [results[d]['nodes_evaluated'] for d in depths]
    times_per_depth = [results[d]['time_taken'] for d in depths]
    
    print(f"\nDepth progression:")
    for depth in depths:
        r = results[depth]
        print(f"  Depth {depth}: {r['nodes_evaluated']} nodes, {r['time_taken']:.4f}s")
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(depths, nodes_per_depth, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Search Depth')
    plt.ylabel('Nodes Evaluated')
    # Main title is now set on the first subplot
    plt.title(f'Iterative Deepening Analysis (State: {initial_state}) - Nodes vs Search Depth', 
              fontsize=12, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(depths, times_per_depth, 'ro-', linewidth=2, markersize=8)
    plt.xlabel('Search Depth')
    plt.ylabel('Time (seconds)')
    plt.title('Time vs Search Depth', pad=20) # Simple title for the second plot
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust layout to ensure space for titles
    plt.show()
    
    return results

def heuristic_effectiveness_analysis(initial_state):
    """Analyze effectiveness of heuristic evaluation"""
    print("=" * 70)
    print("HEURISTIC EFFECTIVENESS ANALYSIS")
    print("=" * 70)
    
    solver = OptimizedNimSolver(initial_state)
    
    # Compare heuristic vs non-heuristic at different depths
    depths = [2, 3, 4]
    
    print(f"Initial state: {initial_state}")
    print("\nComparing heuristic vs non-heuristic minimax:")
    print("Depth | Heuristic Value | Exact Value | Difference | Nodes (H) | Nodes (Exact)")
    print("-" * 80)
    
    for depth in depths:
        # Heuristic search
        solver.nodes_evaluated = 0
        heuristic_value = solver.heuristic_minimax(initial_state, True, 0, depth, True)
        heuristic_nodes = solver.nodes_evaluated
        
        # Exact search (deeper to get true value)
        solver.nodes_evaluated = 0
        exact_value = solver.standard_minimax(initial_state, True, 0, depth + 2)  # Deeper search for "true" value
        exact_nodes = solver.nodes_evaluated
        
        difference = abs(heuristic_value - exact_value)
        
        print(f"{depth:5} | {heuristic_value:15.3f} | {exact_value:11.3f} | {difference:9.3f} | "
              f"{heuristic_nodes:9} | {exact_nodes:12}")

def main():
    # Test with different initial states
    test_states = [(3, 2), (4, 3), (2, 2, 1)]
    
    for initial_state in test_states:
        print("\n" + "=" * 80)
        print(f"ANALYZING NIM STATE: {initial_state}")
        print("=" * 80)
        
        # 2-ply analysis
        analyze_2_ply_game(initial_state)
        
        # Optimization techniques comparison
        optimization_results = compare_optimization_techniques(initial_state)
        visualize_optimization_comparison(optimization_results, initial_state)
        
        # Iterative deepening demonstration
        iterative_results = demonstrate_iterative_deepening(initial_state)
        
        # Heuristic effectiveness analysis
        heuristic_effectiveness_analysis(initial_state)
        
        print("\n" + "=" * 80)
        print(f"ANALYSIS COMPLETE FOR STATE: {initial_state}")
        print("=" * 80)

if __name__ == "__main__":
    main()