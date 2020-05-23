import gym
from gym.envs.toy_text.frozen_lake import generate_random_map
import numpy as np

LEARNING_RATE = 0.1

DISCOUNT = 0.95
EPISODES = 50000
SHOW_EVERY = 5000

# Exploration settings
INITIAL_EPSILON = 1
INITIAL_LEARNING_RATE = 0.8
MIN_EPSILON = 0
MIN_LEARNING_RATE = 0.1
DECAY_RATE = 0.99


class FrozenLakeAgent(object):
    def __init__(self):
        random_map = generate_random_map(size=5, p=0.8)
        self.env = gym.make("FrozenLake-v0", is_slippery=False, desc=random_map)
        self.env.reset()

        self.epsilon = INITIAL_EPSILON
        self.learning_rate = INITIAL_LEARNING_RATE

        self.action_space = self.env.action_space.n
        self.state_space = [self.env.observation_space.n]
        self.q_table = np.random.uniform(low=-2, high=0, size=(self.state_space + [self.action_space]))

    def _run_single_episode(self, episode):
        state = self.env.reset()

        done = False
        while not done:
            if np.random.random() > self.epsilon:
                action = np.argmax(self.q_table[state])
            else:
                action = self.env.action_space.sample()

            new_state, reward, done, _ = self.env.step(action)

            # Make the Q-learning computations
            max_future_q = np.max(self.q_table[new_state])
            current_q = self.q_table[state, action]
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            self.q_table[state, action] = new_q

            state = new_state

        if reward > 0:
            print(f"{episode} - Won!")
        elif episode % SHOW_EVERY == 0:
            print(episode)
            self.env.render()
            print()
            self.epsilon = INITIAL_EPSILON
            self.learning_rate = INITIAL_LEARNING_RATE

    def run_episodes(self, num_episodes):
        for episode in range(num_episodes):
            self._run_single_episode(episode)

            # Decay epsilon and learning rate
            self.epsilon = max(MIN_EPSILON, self.epsilon * DECAY_RATE)
            self.learning_rate = max(MIN_LEARNING_RATE, self.learning_rate * DECAY_RATE)


if __name__ == "__main__":
    agent = FrozenLakeAgent()
    agent.run_episodes(EPISODES)
