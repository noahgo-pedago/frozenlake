"""
FrozenLake Visual Demo

This script provides a visual demonstration of both trained and untrained agents
playing FrozenLake with graphical rendering.
"""

import gymnasium as gym
import numpy as np
import time


class QLearningAgent:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.99,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        """Initialize Q-Learning agent."""
        self.env = env
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = np.zeros([env.observation_space.n, env.action_space.n])

    def choose_action(self, state, greedy=False):
        """Choose action using epsilon-greedy policy."""
        if greedy or np.random.random() >= self.epsilon:
            return np.argmax(self.q_table[state])
        else:
            return self.env.action_space.sample()

    def update_q_table(self, state, action, reward, next_state, done):
        """Update Q-table using Q-learning formula."""
        current_q = self.q_table[state, action]
        if done:
            target_q = reward
        else:
            target_q = reward + self.gamma * np.max(self.q_table[next_state])
        self.q_table[state, action] = current_q + self.lr * (target_q - current_q)

    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def train(self, episodes=10000, verbose=True):
        """Train the agent."""
        for episode in range(episodes):
            state, info = self.env.reset()
            done = False

            while not done:
                action = self.choose_action(state)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                self.update_q_table(state, action, reward, next_state, done)
                state = next_state

            self.decay_epsilon()

            if verbose and (episode + 1) % 1000 == 0:
                print(f"Training progress: {episode + 1}/{episodes} episodes")


def visual_demo_random(episodes=5, delay=0.5):
    """
    Demonstrate a random agent with visual rendering.

    Args:
        episodes: Number of episodes to demonstrate
        delay: Delay between steps in seconds
    """
    print("\n" + "="*60)
    print("RANDOM AGENT DEMO")
    print("="*60)
    print("Watch the agent move randomly (no learning)...")
    print("Press Ctrl+C to stop early\n")

    env = gym.make("FrozenLake-v1", is_slippery=True, render_mode="human")

    action_names = {0: "LEFT", 1: "DOWN", 2: "RIGHT", 3: "UP"}

    try:
        for episode in range(episodes):
            state, info = env.reset()
            print(f"\n--- Episode {episode + 1}/{episodes} ---")
            done = False
            step = 0

            while not done:
                time.sleep(delay)
                action = env.action_space.sample()
                print(f"Step {step + 1}: Action = {action_names[action]}")

                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                if done:
                    if reward > 0:
                        print(f"SUCCESS! Reached the goal in {step + 1} steps!")
                    else:
                        print(f"Failed! Fell in a hole or timed out after {step + 1} steps.")

                state = next_state
                step += 1

    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")
    finally:
        env.close()


def visual_demo_trained(episodes=5, delay=0.5, training_episodes=10000):
    """
    Train an agent and demonstrate it with visual rendering.

    Args:
        episodes: Number of episodes to demonstrate
        delay: Delay between steps in seconds
        training_episodes: Number of training episodes
    """
    print("\n" + "="*60)
    print("TRAINED AGENT DEMO")
    print("="*60)

    # Train agent without rendering
    print(f"Training agent for {training_episodes} episodes...")
    train_env = gym.make("FrozenLake-v1", is_slippery=True)
    agent = QLearningAgent(train_env)
    agent.train(episodes=training_episodes, verbose=True)
    train_env.close()

    print("\nTraining complete! Now demonstrating the trained agent...")
    print("Watch how the agent learned to navigate to the goal!\n")

    # Demonstrate with rendering
    demo_env = gym.make("FrozenLake-v1", is_slippery=True, render_mode="human")
    action_names = {0: "LEFT", 1: "DOWN", 2: "RIGHT", 3: "UP"}

    successes = 0

    try:
        for episode in range(episodes):
            state, info = demo_env.reset()
            print(f"\n--- Episode {episode + 1}/{episodes} ---")
            done = False
            step = 0

            while not done:
                time.sleep(delay)
                # Use greedy policy (no exploration) for demonstration
                action = np.argmax(agent.q_table[state])
                print(f"Step {step + 1}: Action = {action_names[action]}")

                next_state, reward, terminated, truncated, info = demo_env.step(action)
                done = terminated or truncated

                if done:
                    if reward > 0:
                        print(f"SUCCESS! Reached the goal in {step + 1} steps!")
                        successes += 1
                    else:
                        print(f"Failed! Fell in a hole or timed out after {step + 1} steps.")

                state = next_state
                step += 1

        print(f"\n{'='*60}")
        print(f"Demo Results: {successes}/{episodes} successes ({successes/episodes*100:.1f}%)")
        print(f"{'='*60}")

    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")
    finally:
        demo_env.close()


def compare_agents(episodes_each=3, delay=0.3):
    """
    Compare random and trained agents side by side.

    Args:
        episodes_each: Number of episodes for each agent
        delay: Delay between steps in seconds
    """
    print("\n" + "="*60)
    print("AGENT COMPARISON")
    print("="*60)
    print("First, training an agent...\n")

    # Train agent
    train_env = gym.make("FrozenLake-v1", is_slippery=True)
    agent = QLearningAgent(train_env)
    agent.train(episodes=10000, verbose=True)
    train_env.close()

    print("\n" + "="*60)
    print("Now watch: RANDOM vs TRAINED")
    print("="*60)

    # Demo random agent
    print("\n[1/2] RANDOM AGENT")
    visual_demo_random(episodes=episodes_each, delay=delay)

    time.sleep(2)  # Pause between demos

    # Demo trained agent
    print("\n[2/2] TRAINED AGENT (using same Q-table)")
    demo_env = gym.make("FrozenLake-v1", is_slippery=True, render_mode="human")
    action_names = {0: "LEFT", 1: "DOWN", 2: "RIGHT", 3: "UP"}

    successes = 0

    try:
        for episode in range(episodes_each):
            state, info = demo_env.reset()
            print(f"\n--- Episode {episode + 1}/{episodes_each} ---")
            done = False
            step = 0

            while not done:
                time.sleep(delay)
                action = np.argmax(agent.q_table[state])
                print(f"Step {step + 1}: Action = {action_names[action]}")

                next_state, reward, terminated, truncated, info = demo_env.step(action)
                done = terminated or truncated

                if done:
                    if reward > 0:
                        print(f"SUCCESS! Reached the goal in {step + 1} steps!")
                        successes += 1
                    else:
                        print(f"Failed! Fell in a hole or timed out after {step + 1} steps.")

                state = next_state
                step += 1

        print(f"\nTrained Agent: {successes}/{episodes_each} successes")

    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")
    finally:
        demo_env.close()


def main():
    """Main menu for visual demonstrations."""
    print("\n" + "="*60)
    print("FROZENLAKE VISUAL DEMO")
    print("="*60)
    print("\nChoose a demo mode:")
    print("1. Random Agent (no learning)")
    print("2. Trained Agent (Q-learning)")
    print("3. Compare Both Agents")
    print("4. Quick Demo (3 episodes, trained agent)")
    print("\nPress Ctrl+C at any time to exit")
    print("="*60)

    try:
        choice = input("\nEnter your choice (1-4): ").strip()

        if choice == "1":
            episodes = int(input("How many episodes? (default 5): ") or "5")
            delay = float(input("Delay between steps in seconds? (default 0.5): ") or "0.5")
            visual_demo_random(episodes=episodes, delay=delay)

        elif choice == "2":
            episodes = int(input("How many demo episodes? (default 5): ") or "5")
            delay = float(input("Delay between steps in seconds? (default 0.5): ") or "0.5")
            training = int(input("Training episodes? (default 10000): ") or "10000")
            visual_demo_trained(episodes=episodes, delay=delay, training_episodes=training)

        elif choice == "3":
            episodes = int(input("Episodes per agent? (default 3): ") or "3")
            delay = float(input("Delay between steps in seconds? (default 0.3): ") or "0.3")
            compare_agents(episodes_each=episodes, delay=delay)

        elif choice == "4":
            print("\nRunning quick demo with trained agent...")
            visual_demo_trained(episodes=3, delay=0.3, training_episodes=10000)

        else:
            print("Invalid choice. Exiting.")

    except KeyboardInterrupt:
        print("\n\nExiting...")
    except Exception as e:
        print(f"\nError: {e}")


if __name__ == "__main__":
    main()
