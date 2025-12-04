"""
FrozenLake Q-Learning Agent

This script trains an agent to solve the FrozenLake environment using Q-learning.
"""

import gymnasium as gym
import numpy as np


class QLearningAgent:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.99,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        """
        Initialize Q-Learning agent.

        Args:
            env: Gymnasium environment
            learning_rate: Learning rate (alpha)
            discount_factor: Discount factor (gamma)
            epsilon: Initial exploration rate
            epsilon_decay: Rate at which epsilon decays
            epsilon_min: Minimum epsilon value
        """
        self.env = env
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Initialize Q-table with zeros
        self.q_table = np.zeros([env.observation_space.n, env.action_space.n])

    def choose_action(self, state):
        """Choose action using epsilon-greedy policy."""
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()  # Explore
        else:
            return np.argmax(self.q_table[state])  # Exploit

    def update_q_table(self, state, action, reward, next_state, done):
        """Update Q-table using Q-learning formula."""
        current_q = self.q_table[state, action]

        if done:
            target_q = reward
        else:
            target_q = reward + self.gamma * np.max(self.q_table[next_state])

        # Q-learning update
        self.q_table[state, action] = current_q + self.lr * (target_q - current_q)

    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def train(self, episodes=10000, print_every=1000):
        """Train the agent."""
        rewards_history = []

        for episode in range(episodes):
            state, info = self.env.reset()
            total_reward = 0
            done = False

            while not done:
                action = self.choose_action(state)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                self.update_q_table(state, action, reward, next_state, done)

                state = next_state
                total_reward += reward

            rewards_history.append(total_reward)
            self.decay_epsilon()

            # Print progress
            if (episode + 1) % print_every == 0:
                avg_reward = np.mean(rewards_history[-print_every:])
                print(f"Episode {episode + 1}/{episodes} | "
                      f"Avg Reward: {avg_reward:.3f} | "
                      f"Epsilon: {self.epsilon:.3f}")

        return rewards_history

    def evaluate(self, episodes=100, render=False):
        """Evaluate the trained agent."""
        if render:
            eval_env = gym.make("FrozenLake-v1", render_mode="human")
        else:
            eval_env = self.env

        total_rewards = 0
        wins = 0

        for episode in range(episodes):
            state, info = eval_env.reset()
            done = False
            episode_reward = 0

            while not done:
                # Use greedy policy (no exploration)
                action = np.argmax(self.q_table[state])
                next_state, reward, terminated, truncated, info = eval_env.step(action)
                done = terminated or truncated

                state = next_state
                episode_reward += reward

            total_rewards += episode_reward
            if episode_reward > 0:
                wins += 1

        if render:
            eval_env.close()

        avg_reward = total_rewards / episodes
        win_rate = (wins / episodes) * 100

        print(f"\n{'='*50}")
        print(f"Evaluation over {episodes} episodes:")
        print(f"Average Reward: {avg_reward:.3f}")
        print(f"Win Rate: {win_rate:.1f}%")
        print(f"{'='*50}\n")

        return avg_reward, win_rate


def main():
    # Create FrozenLake environment
    env = gym.make("FrozenLake-v1", is_slippery=True)

    print("FrozenLake Q-Learning Training")
    print("="*50)
    print(f"Observation Space: {env.observation_space}")
    print(f"Action Space: {env.action_space}")
    print("="*50 + "\n")

    # Initialize agent
    agent = QLearningAgent(
        env=env,
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01
    )

    # Train agent
    print("Training agent...")
    rewards_history = agent.train(episodes=10000, print_every=1000)

    # Evaluate agent
    print("\nEvaluating trained agent...")
    agent.evaluate(episodes=100, render=False)

    # Close environment
    env.close()

    print("Training complete! You can now run a visual demo.")
    print("To see the agent play, run: python frozenlake_demo.py")


if __name__ == "__main__":
    main()
