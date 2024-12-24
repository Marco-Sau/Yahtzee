🎲 Yahtzee DQN Agent - AI Game Player and Analyzer

This repository contains a Deep Q-Network (DQN)-based AI agent designed to play and analyze the game of Yahtzee. The project leverages reinforcement learning techniques to train an agent capable of making optimal decisions in the game environment.

📂 Project Overview

The project consists of the following key components:

- Agent Training: Train a DQN agent to play Yahtzee.
- Gameplay and Analysis: Play games using the trained agent and visualize performance metrics.
- Visualization: Track agent performance over multiple games and plot results to evaluate learning progress.

🚀 Key Features

- Dueling DQN Architecture: Separates state value and action advantage to enhance learning stability.
- Performance Tracking: Real-time and post-game plotting of performance metrics.
- Model Persistence: Trained models can be saved and reloaded to continue gameplay or retraining.
- Error Handling: Graceful exception handling during gameplay to prevent crashes.

📋 Requirements

Before running the project, ensure you have the following dependencies installed:

pip install torch numpy matplotlib  

📁 Project Structure

```
project-directory
├── play_yahtzee.py        # Main script to load and run trained agent
├── train_yahtzee.py       # Script to train the DQN agent
├── yahtzee_api.py         # Yahtzee game logic and environment API
├── yahtzee_dqn.py         # DQN architecture and agent implementation
└── models/                # Directory to save trained model checkpoints
```

📜 Script Descriptions

1. play_yahtzee.py

Purpose:
- This script loads a trained DQN agent and uses it to play Yahtzee games. It can run single or multiple games, track performance, and visualize results.

Main Components:

- Model Loading: Loads a saved model checkpoint to resume gameplay.
- Gameplay Execution: Simulates game rounds and records performance.
- Visualization: Generates live plots and performance distribution histograms.

Usage:

python play_yahtzee.py  

Key Functions:

- load_model(model_path): Loads trained model weights and initializes the DQN agent.
- play_game(agent, verbose=True): Plays a full game using the agent. Tracks game state, actions, and scores.
- update_plot(scores, fig, ax1, window_size=10): Updates a live plot showing moving average scores.
- main(): Orchestrates model loading, gameplay, and visualization.

1. train_yahtzee.py

Purpose:

- This script trains a DQN agent by interacting with the Yahtzee environment. It saves the model checkpoint after training.

Main Components:

- Replay Memory: Stores past experiences to enable experience replay during training.
- Target Network: Stabilizes learning by periodically updating a target network.
- Exploration-Exploitation: Uses an ε-greedy strategy for balancing exploration and exploitation.

Key Functions:

- train_dqn(agent, episodes): Trains the DQN agent over a specified number of episodes.
- optimize_model(): Samples experiences from memory and performs optimization.
- save_model(path): Saves model weights to disk after training.

3. yahtzee_api.py

Purpose:

- Implements the core game logic and Yahtzee environment. Provides the interface for dice rolls, rerolls, and scoring.

Key Classes and Methods:

- YahtzeeGame(): Class that simulates a full Yahtzee game, managing dice, scorecards, and game rules.
- getDiceValues(): Returns the current values of the dice.
- chooseAction(action): Executes an action (reroll or score) based on the agent's decision.

4. yahtzee_dqn.py

Purpose:

- Defines the DQN architecture and the dueling network used to approximate Q-values for actions.

Key Components:

- YahtzeeDQN (nn.Module): Implements the dueling DQN architecture.
- YahtzeeAgent: Wraps the DQN network to manage action selection, replay memory, and learning.

Architecture:

- Feature Layer: Shared feature extraction layer.
- Value Stream: Estimates the overall value of the current state.
- Advantage Stream: Estimates the relative advantage of each action.

🎮 How to Run

Training the Agent

python train_yahtzee.py  

Playing the Game with a Trained Agent

python play_yahtzee.py  

📊 Visualization

- Score Distribution Histogram: Shows the distribution of scores across multiple games.
- Performance Over Time: Tracks the moving average of scores to show learning progress.

Example Visualization:

plt.hist(scores, bins=20)  
plt.title('Yahtzee Agent Performance')  
plt.show()  

🛠️ Model Saving and Loading

- Trained models are saved in the /models directory.
- To load a model, specify the path in play_yahtzee.py:

model_path = "models/yahtzee_model_final.pth"  

📈 Performance Metrics

- Moving Average: Tracks the rolling average of scores over the last 10 games.
- Best Score: Highlights the highest score achieved during testing.

🔧 Troubleshooting

- CUDA Errors: If CUDA is unavailable, ensure the script uses MPS for Apple Silicon or switch to CPU.
- Model Load Issues: Ensure paths to model files are correct. If errors persist, retrain the model.

📖 Acknowledgments

- PyTorch Documentation
- Reinforcement Learning Literature
- OpenAI DQN References
