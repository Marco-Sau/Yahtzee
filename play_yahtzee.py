############################################################
# YAHTZEE DQN AGENT GAME PLAYER AND ANALYZER
# Main script for loading and running a trained DQN agent,
# playing games, and analyzing/visualizing performance.
# Provides tools for both single-game analysis and
# multi-game performance evaluation.
############################################################

"""
Yahtzee DQN Agent Game Player and Analyzer

This script provides functionality to:
1. Load and run a trained Deep Q-Network (DQN) agent for playing Yahtzee
2. Play individual games with detailed move tracking
3. Run multiple games to analyze agent performance
4. Visualize performance metrics through various plots

Key Components:
- Model Loading: Loads trained DQN model with appropriate device handling
- Gameplay: Manages game state, action selection, and history tracking
- Analysis: Tracks scores and generates performance visualizations
- Visualization: Real-time and final plotting of performance metrics

Dependencies:
- torch: Neural network and model loading
- numpy: Numerical operations and statistics
- matplotlib: Visualization and plotting
- yahtzee_api: Game logic implementation
- yahtzee_dqn: DQN agent implementation
"""

import torch
import numpy as np
from yahtzee_api import YahtzeeGame
from yahtzee_dqn import YahtzeeAgent
from train_yahtzee import get_state_representation
import matplotlib.pyplot as plt
from collections import deque

############################################################
# MODEL LOADING AND INITIALIZATION
# Handles loading of trained model weights and initialization
# of the DQN agent with appropriate architecture and device
# configuration.
############################################################

def load_model(model_path):
    """Load the trained model from a checkpoint file."""
    # Device configuration - Uses MPS (Metal Performance Shaders) for Mac M1
    device = torch.device("mps")
    print(f"Using device: {device}")
    
    # Initialize game and agent with same architecture as training
    # - state_size=32: represents dice values and game state
    # - action_size=45: 32 possible reroll combinations + 13 scoring categories
    # - hidden_size=256: size of neural network layers
    game = YahtzeeGame()
    state_size = 32
    action_size = 45
    agent = YahtzeeAgent(
        state_size=state_size,
        action_size=action_size,
        game=game,
        hidden_size=256
    )
    
    # Load saved model weights and move to appropriate device
    checkpoint = torch.load(model_path, map_location=device)
    agent.qnetwork_local.load_state_dict(checkpoint['model_state_dict'])
    agent.qnetwork_local = agent.qnetwork_local.to(device)
    agent.qnetwork_target = agent.qnetwork_target.to(device)
    agent.qnetwork_local.eval()  # Disable dropout and other training-specific features
    
    return agent

############################################################
# GAMEPLAY AND HISTORY TRACKING
# Manages the execution of individual games, including action
# selection, state tracking, and detailed history recording
# for analysis.
############################################################

def play_game(agent, verbose=True):
    """Play a complete game of Yahtzee using the trained agent."""
    game = agent.game
    game.newGame()
    game_history = []  # Track all actions and their outcomes
    
    # Print agent configuration if verbose mode is enabled
    if verbose:
        print("\nStarting new game!")
        print("\nAgent Settings:")
        print(f"Hidden Layer Size: {agent.qnetwork_local.feature_layer[0].out_features}")
        print(f"Learning Rate: {agent.optimizer.param_groups[0]['lr']}")
        print(f"Gamma (Discount Factor): {agent.gamma}")
        print(f"Memory Size: {agent.memory.memory.maxlen}")
        print(f"Batch Size: {agent.batch_size}")
        print(f"Update Frequency: Every {agent.update_every} steps")
        print("\nGame Progress:")
    
    turn = 1
    while not game.hasFinished():
        # Get current state and choose best action (no exploration)
        state = get_state_representation(game)
        action = agent.act(state, eps=0.0)  # eps=0.0 means purely exploiting learned policy
        
        # Record detailed information about the current turn
        turn_info = {
            'turn': turn,
            'dice': game.getDiceValues().copy(),
            'rolls_left': game.rollLeft(),
            'action': action,
            'action_type': 'reroll' if action < 32 else 'score'
        }
        
        # Execute the chosen action and record outcomes
        try:
            game.chooseAction(action)
            
            # Handle reroll actions (actions 0-31)
            if action < 32:
                reroll_flags = [(action >> i) & 1 for i in range(5)]
                turn_info.update({'reroll_pattern': reroll_flags, 'new_dice': game.getDiceValues().copy()})
            # Handle scoring actions (actions 32-44)
            else:
                category = list(game.scorecard.keys())[action - 32]
                turn_info.update({'category': category, 'score': game.getLastReward()})
            
            game_history.append(turn_info)
            
        except Exception as e:
            print(f"Error: {e}")
            break
        
        # Increment turn counter when a new turn starts
        if game.rollLeft() == 2:
            turn += 1
    
    return game.getTotalReward(), game_history

############################################################
# VISUALIZATION AND PERFORMANCE TRACKING
# Real-time plotting and analysis of agent performance,
# including moving averages and score distributions.
############################################################

def update_plot(scores, fig, ax1, window_size=10):
    """Update the live performance plot with new scores.
    - Calculates and displays moving average of scores
    - Updates plot in real-time as games are played"""
    ax1.clear()
    moving_avg = [np.mean(scores[max(0, i-window_size):i+1]) 
                 for i in range(len(scores))]
    
    x = range(1, len(scores) + 1)
    ax1.plot(x, moving_avg, 'b-', label=f'Moving Average (window={window_size})')
    
    # Set labels and title
    ax1.set_xlabel('Games')
    ax1.set_ylabel('Average Score')
    plt.title('Yahtzee Performance Metrics')
    
    # Add legend
    ax1.legend(loc='upper left')
    
    plt.tight_layout()
    plt.pause(0.1)

############################################################
# MAIN EXECUTION
# Orchestrates the overall workflow: model loading, gameplay,
# analysis, and visualization generation. Runs both single
# and multiple game scenarios.
############################################################
n_games = 100
def main(n_games=n_games):
    """Main execution function that:
    1. Loads the trained model
    2. Plays one detailed game with verbose output
    3. Runs multiple games while tracking performance
    4. Generates visualization plots:
       - Score distribution histogram
       - Performance metrics over time
       - Live updating plot during gameplay"""
    # Load the trained model
    model_path = "models/yahtzee_model_final_00.pth"  # Update this path to your saved model
    try:
        agent = load_model(model_path)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Play a single game with verbose output
    print("\nPlaying a sample game with detailed output:")
    score, game_history = play_game(agent)
    
    # Setup for live plotting
    plt.ion()  # Turn on interactive mode
    fig = plt.figure(figsize=(10, 6))
    ax1 = plt.gca()
    
    # Play multiple games with live plotting
    # n_games = 100
    scores = []
    print(f"\nPlaying {n_games} games with live performance plotting...")
    
    all_game_histories = []
    for i in range(n_games):
        score, history = play_game(agent, verbose=(i == 0))  # Only show verbose output for first game
        scores.append(score)
        all_game_histories.append(history)
        
        # Update plot every game
        update_plot(scores, fig, ax1)
        
        # Print current game stats
        print(f"Game {i+1}/{n_games} - Score: {score}")
    
    print(f"\nFinal Results:")
    print(f"Average score: {np.mean(scores):.2f}")
    print(f"Best score: {max(scores)}")
    
    # Plot 1: Score Distribution Histogram (New Window)
    plt.figure(figsize=(8, 6))
    plt.hist(scores, bins=20, edgecolor='black')
    plt.title('Score Distribution')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.savefig('score_distribution.png')
    
    # Plot 2: Performance Metrics (New Window)
    plt.figure(figsize=(8, 6))
    window_size = 10 # represents the number of games to average over
    moving_avg = [np.mean(scores[max(0, i-window_size):i+1]) 
                 for i in range(len(scores))]
    
    x = range(1, len(scores) + 1)
    plt.plot(x, moving_avg, 'b-', label=f'Moving Average)')
    plt.xlabel('Games')
    plt.ylabel('Score')
    plt.title('Performance Metrics')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('performance_metrics.png')
    
    # Plot 3: Live Plot (Already in its own window)
    plt.show()

if __name__ == "__main__":
    main() 