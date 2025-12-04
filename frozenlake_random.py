"""
FrozenLake Random Agent

A simple example that demonstrates the FrozenLake environment with a random agent.
"""

import gymnasium as gym
import numpy as np


def run_random_agent(episodes=100, render=False):
    """
    Run a random agent in the FrozenLake environment.

    Args:
        episodes: Number of episodes to run
        render: Whether to render the environment
    """
    if render:
        env = gym.make("FrozenLake-v1", is_slippery=True, render_mode="human")
    else:
        env = gym.make("FrozenLake-v1", is_slippery=True)

    print("FrozenLake Random Agent")
    print("="*50)
    print(f"Observation Space: {env.observation_space}")
    print(f"Action Space: {env.action_space}")
    print("Actions: 0=Left, 1=Down, 2=Right, 3=Up")
    print("="*50 + "\n")

    total_rewards = 0
    wins = 0
    total_steps = 0

    for episode in range(episodes):
        state, info = env.reset()
        done = False
        episode_reward = 0
        steps = 0

        while not done:
            # Take a random action
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            steps += 1
            state = next_state

        total_rewards += episode_reward
        total_steps += steps

        if episode_reward > 0:
            wins += 1
            if episodes <= 10:  # Print details for small number of episodes
                print(f"Episode {episode + 1}: SUCCESS in {steps} steps!")
        elif episodes <= 10:
            print(f"Episode {episode + 1}: Failed after {steps} steps")

    env.close()

    # Print statistics
    avg_reward = total_rewards / episodes
    win_rate = (wins / episodes) * 100
    avg_steps = total_steps / episodes

    print(f"\n{'='*50}")
    print(f"Results after {episodes} episodes:")
    print(f"Win Rate: {win_rate:.1f}%")
    print(f"Average Reward: {avg_reward:.3f}")
    print(f"Average Steps: {avg_steps:.1f}")
    print(f"{'='*50}\n")


def main():
    print("Running random agent for 1000 episodes...\n")
    run_random_agent(episodes=1000, render=False)

    print("\nThis demonstrates the baseline performance.")
    print("The Q-learning agent should perform much better!")


if __name__ == "__main__":
    main()
