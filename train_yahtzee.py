############################################################
# SETUP AND IMPORTS
# Import all required libraries for:
# - Deep learning (PyTorch)
# - Data manipulation (NumPy)
# - Visualization (Matplotlib)
# - Progress tracking and file management
############################################################
import numpy as np
import torch
from collections import deque
import matplotlib.pyplot as plt
import os
from datetime import datetime
from tqdm import tqdm
from yahtzee_api import YahtzeeGame
from yahtzee_dqn import YahtzeeAgent
import random

############################################################
# STATE REPRESENTATION
# Converts the Yahtzee game state into a format suitable for 
# the neural network. Creates a 32-dimensional vector that
# captures all relevant game information.
############################################################
def get_state_representation(game):
    """Convert game state to neural network input format
    Returns a 32-dimensional state vector containing:
    - 5 dice values
    - 1 rolls remaining value
    - 13 binary flags for completed categories
    - 13 score values for each category
    """
    # Get current dice values and remaining rolls
    dice = game.getDiceValues()
    rolls_left = game.rollLeft()
    
    # Create one-hot encoding for completed rows (13 values)
    # 1 indicates category is used, 0 indicates available
    completed_rows = [0] * 13
    for row in game.getCompletedRows():
        idx = list(game.scorecard.keys()).index(row)
        completed_rows[idx] = 1
    
    # Get scores for each category, using 0 for unfilled categories
    scores = [game.scorecard[category] if game.scorecard[category] is not None else 0 
             for category in game.scorecard.keys()]
    
    # Combine all state components into a single vector
    state = np.concatenate([
        dice,                    # 5 values: current dice faces
        [rolls_left],           # 1 value: remaining rolls (0-3)
        completed_rows,         # 13 values: category completion status
        scores                  # 13 values: current category scores
    ])
    
    return state

############################################################
# MAIN TRAINING ALGORITHM
# Implements the Deep Q-Learning training loop:
# 1. Environment setup
# 2. Agent initialization
# 3. Episode-based training
# 4. Experience collection and learning
# 5. Performance monitoring and model saving
############################################################
def train_yahtzee(
    n_episodes=100000,  # Number of episodes to train the agent
    max_t=1000,         # Maximum number of steps per episode
    eps_start=1.0,      # Initial exploration rate
    eps_end=0.01,       # Minimum exploration rate
    eps_decay=0.5       # Rate of exploration decay
):
    ############################
    # INITIALIZATION PHASE
    # Setup the environment, agent, and tracking metrics
    ############################
    # Initialize game and agent
    game = YahtzeeGame()
    state_size = 32
    action_size = 45
    agent = YahtzeeAgent(
        state_size=state_size,
        action_size=action_size,
        game=game,
        hidden_size=256,        # Neural network hidden layer size
        batch_size=256,         # Number of experiences to learn from at once
        learning_rate=1e-3,     # How quickly the network adapts
        gamma=0.1,              # Discount factor for future rewards
        tau=1e-2,               # Soft update parameter for target network
        memory_size=200000,     # Size of replay buffer
        update_every=10         # How often to update the network
    )
    
    # Setup tracking and visualization
    save_dir = "models"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Initialize metrics tracking
    scores = []                           # Tracks rewards (scores) per episode
    scores_window = deque(maxlen=100)     # Rolling window to compute average score
    eps = eps_start                       # Initialize epsilon for exploration
    avg_scores = []                       # Tracks moving average of scores
    max_scores = []                       # Stores highest score per episode
    best_avg_score = -np.inf              # Tracks the best average score achieved
        
    # Setup visualization
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    progress_bar = tqdm(range(1, n_episodes+1), desc='Training Progress')
    
    ############################
    # TRAINING PHASE
    # Main loop where the agent learns to play Yahtzee
    ############################
    for i_episode in progress_bar:
        # Initialize episode
        game.newGame()
        state = get_state_representation(game)
        score = 0
        episode_actions = []
        episode_rewards = []
        
        ############################
        # EPISODE EXECUTION
        # Agent interacts with environment and collects experiences
        ############################
        while not game.hasFinished():
            # Action selection and execution
            action = agent.act(state, eps)
            
            # Handle invalid actions
            try:
                game.chooseAction(action)
            except Exception as e:
                if action < 32 and game.rollLeft() == 0:
                    available_categories = [i for i, completed in 
                                         enumerate(game.completed_rows) if not completed]
                    if available_categories:
                        action = random.choice(available_categories) + 32
                        game.chooseAction(action)
            
            # Experience collection
            next_state = get_state_representation(game)
            reward = game.getLastReward()
            done = game.hasFinished()
            
            # Learning step
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score = game.getTotalReward()
            
            # Record episode history
            episode_actions.append(action)
            episode_rewards.append(reward)
        
        ############################
        # METRICS UPDATE PHASE
        # Update tracking metrics and adjust exploration
        ############################
        scores_window.append(score)
        scores.append(score)
        eps = max(eps_end, eps_decay * eps)
        
        ############################
        # VISUALIZATION AND LOGGING PHASE
        # Every 1000 episodes, update plots and save progress
        ############################
        if i_episode % 1000 == 0:
            # Calculate current metrics
            current_avg = np.mean(scores[-1000:] if len(scores) >= 1000 else scores)
            current_max = np.max(scores[-1000:] if len(scores) >= 1000 else scores)
            avg_scores.append(current_avg)
            max_scores.append(current_max)
            
            # Update visualization plots
            ax1.clear()
            ax2.clear()
            x_values = range(1000, i_episode + 1, 1000)
            
            # Plot average scores
            ax1.plot(x_values, avg_scores, color='blue')
            ax1.set_ylabel('Average Score')
            ax1.set_xlabel('Episode')
            ax1.set_title('Average Score over Training')
            ax1.grid(True)
            
            # Plot max scores
            ax2.plot(x_values, max_scores, color='red')
            ax2.set_ylabel('Max Score')
            ax2.set_xlabel('Episode')
            ax2.set_title('Max Score over Training')
            ax2.grid(True)
            
            plt.tight_layout()
            plt.pause(0.1)
            
            # Update progress bar and print statistics
            progress_bar.set_description(
                f'Avg Score: {current_avg:.2f} | Eps: {eps:.2f}'
            )
            print(f'\nEpisode {i_episode}')
            print(f'Average Score: {current_avg:.2f}')
            print(f'Max Score: {current_max:.2f}')
        
        ############################
        # MODEL CHECKPOINTING PHASE
        # Save model if it achieves best performance
        ############################
        if np.mean(scores_window) > best_avg_score:
            best_avg_score = np.mean(scores_window)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = os.path.join(save_dir, f'yahtzee_model_best_{timestamp}.pth')
            torch.save({
                'episode': i_episode,
                'model_state_dict': agent.qnetwork_local.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'score': best_avg_score,
                'device': str(agent.device)
            }, model_path)
            progress_bar.write(f'\nNew best model saved with average score: {best_avg_score:.2f}')
    
    ############################
    # CLEANUP AND FINAL SAVE PHASE
    # Save final model and return training metrics
    ############################
    progress_bar.close()
    plt.ioff()
    final_path = os.path.join(save_dir, 'yahtzee_model_final.pth')
    torch.save({
        'episode': n_episodes,
        'model_state_dict': agent.qnetwork_local.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'score': np.mean(scores_window),
        'device': str(agent.device)
    }, final_path)
    
    return scores, avg_scores, max_scores

############################################################
# VISUALIZATION
# Creates and saves final training metrics plots
############################################################
def plot_final_metrics(scores, avg_scores, max_scores):
    """
    Create and save final visualization of training metrics
    """
    plt.figure(figsize=(12, 8))
    
    # Plot average scores
    plt.subplot(211)
    x_values = range(1000, len(scores) + 1, 1000)
    plt.plot(x_values, avg_scores, color='blue')
    plt.title('Average Score over Training')
    plt.xlabel('Episode')
    plt.ylabel('Average Score')
    plt.grid(True)
    
    # Plot max scores
    plt.subplot(212)
    plt.plot(x_values, max_scores, color='red')
    plt.title('Max Score over Training')
    plt.xlabel('Episode')
    plt.ylabel('Max Score')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.show()

############################################################
# MAIN EXECUTION
# Entry point of the script
############################################################
if __name__ == "__main__":
    scores, avg_scores, max_scores = train_yahtzee()
    plot_final_metrics(scores, avg_scores, max_scores)