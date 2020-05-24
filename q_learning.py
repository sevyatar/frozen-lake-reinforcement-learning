import gym
from gym.envs.toy_text.frozen_lake import generate_random_map
import numpy as np

DISCOUNT = 0.99
EPISODES = 100000
SHOW_EVERY = 5000
RESTART_DECAY_EVERY = 10000

# Exploration settings
INITIAL_EPSILON = 1
INITIAL_LEARNING_RATE = 0.8
MIN_EPSILON = 0.05
MIN_LEARNING_RATE = 0.1

# Exploration decay
EPISODE_TO_END_DECAY = (EPISODES * 3) // 4
EPSILON_DECAY_FACTOR = (MIN_EPSILON / INITIAL_EPSILON) ** (1 / EPISODE_TO_END_DECAY)
LEARNING_DECAY_FACTOR = (MIN_LEARNING_RATE / INITIAL_LEARNING_RATE) ** (1 / EPISODE_TO_END_DECAY)


class FrozenLakeAgent(object):
    def __init__(self):
        random_map = generate_random_map(size=8, p=0.8)
        self.env = gym.make("FrozenLake-v0", is_slippery=True, desc=random_map)
        self.env.reset()

        self.epsilon = INITIAL_EPSILON
        self.learning_rate = INITIAL_LEARNING_RATE

        self.action_space = self.env.action_space.n
        self.state_space = [self.env.observation_space.n]
        self.q_table = np.random.uniform(low=-2, high=0, size=(self.state_space + [self.action_space]))

    def _run_single_episode(self, episode, only_follow_best_route=False):
        state = self.env.reset()

        done = False
        while not done:
            if only_follow_best_route:
                action = np.argmax(self.q_table[state])
                self.env.render()
            else:
                if np.random.random() <= self.epsilon:
                    action = self.env.action_space.sample()
                else:
                    action = np.argmax(self.q_table[state])

            new_state, reward, done, _ = self.env.step(action)

            # Make the Q-learning computations
            max_future_q = np.max(self.q_table[new_state])
            current_q = self.q_table[state, action]
            new_q = (1 - self.learning_rate) * current_q + self.learning_rate * (reward + DISCOUNT * max_future_q)
            self.q_table[state, action] = new_q

            state = new_state

        if reward > 0:
            print(f"{episode} - Won!")

        if only_follow_best_route:
            self.env.render()
        else:
            if episode % SHOW_EVERY == 0:
                print(episode)
                self.env.render()
                print()

            if episode % RESTART_DECAY_EVERY == 0:
                self.epsilon = INITIAL_EPSILON
                self.learning_rate = INITIAL_LEARNING_RATE

    def run_episodes(self, num_episodes):
        for episode in range(num_episodes):
            self._run_single_episode(episode)

            # Decay epsilon and learning rate
            self.epsilon = max(MIN_EPSILON, self.epsilon * EPSILON_DECAY_FACTOR)
            self.learning_rate = max(MIN_LEARNING_RATE, self.learning_rate * LEARNING_DECAY_FACTOR)

        self._run_single_episode(0, only_follow_best_route=True)


if __name__ == "__main__":
    agent = FrozenLakeAgent()
    agent.run_episodes(EPISODES)
