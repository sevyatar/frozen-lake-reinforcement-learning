import random

import gym
from gym.envs.toy_text.frozen_lake import generate_random_map
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

DISCOUNT = 0.99
EPISODES = 2000
SHOW_EVERY = 100
RESTART_DECAY_EVERY = 1000

REPLAY_MEMORY_SIZE = 50_000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 1_000  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 16  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)

# Exploration settings
INITIAL_EPSILON = 1
MIN_EPSILON = 0.05

# Exploration decay
EPISODE_TO_END_DECAY = EPISODES * 0.8
EPSILON_DECAY_FACTOR = (MIN_EPSILON / INITIAL_EPSILON) ** (1 / EPISODE_TO_END_DECAY)


class FrozenLakeDQNAgent(object):
    def __init__(self):
        random_map = generate_random_map(size=6, p=0.8)
        self.env = gym.make("FrozenLake-v0", is_slippery=True, desc=random_map)
        self.env.reset()

        self.state_space = self.env.observation_space.n
        self.action_space = self.env.action_space.n

        self.epsilon = INITIAL_EPSILON

        # Main model
        self.model = self._create_model()

        # Target network
        self.target_model = self._create_model()
        self.target_model.set_weights(self.model.get_weights())

        # An array with last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0

    def _create_model(self):
        model = Sequential()
        model.add(Dense(self.state_space, input_dim=self.state_space, activation='relu'))
        model.add(Dense(self.state_space, activation='relu'))
        model.add(Dense(self.action_space, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=0.001), metrics=['accuracy'])

        return model

    def _encode_state(self, state):
        # TODO - play around with this function by adding the actual map data (equivalent to "seeing" the map of holes)
        state_m = np.zeros(self.state_space)
        state_m[state] = 1
        return state_m

    def _train(self, final_step):
        if len(self.replay_memory) < MINIBATCH_SIZE:
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, min(MINIBATCH_SIZE, len(self.replay_memory)))

        # Get current/new states from minibatch
        current_states = np.array([transition[0] for transition in minibatch])
        new_current_states = np.array([transition[3] for transition in minibatch])

        # Query NN model for Q values (current + new)
        current_qs_list = self.model.predict(current_states)
        new_qs_list = self.target_model.predict(new_current_states)

        X = []
        y = []

        # Enumerate all transitions in the minibatch
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            max_future_q = 0
            if not done:
                max_future_q = np.max(new_qs_list[index])

            new_q = reward + DISCOUNT * max_future_q

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # And append to our training data
            X.append(current_state)
            y.append(current_qs)

        # Fit on all samples as one batch, log only on terminal state
        self.model.fit(np.array(X), np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False)

        # Update target network counter every episode
        if final_step:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    def _run_single_episode(self, episode, only_follow_best_route=False):
        state = self._encode_state(self.env.reset())

        done = False
        while not done:
            if only_follow_best_route:
                action = np.argmax(self.model.predict(np.array([state])))
                self.env.render()
            else:
                if np.random.random() <= self.epsilon:
                    action = self.env.action_space.sample()
                else:
                    action = np.argmax(self.model.predict(np.array([state])))

            new_state, reward, done, _ = self.env.step(action)
            new_state = self._encode_state(new_state)

            self.replay_memory.append((state, action, reward, new_state, done))
            agent._train(final_step=done)

            state = new_state

        # If needed, print some stuff to the standard output
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

    def run_episodes(self, num_episodes):
        for episode in range(num_episodes):
            self._run_single_episode(episode)

            # Decay epsilon
            self.epsilon = max(MIN_EPSILON, self.epsilon * EPSILON_DECAY_FACTOR)

            # TODO - should I update some version of TensorBoard here?

        self._run_single_episode(0, only_follow_best_route=True)


if __name__ == "__main__":
    agent = FrozenLakeDQNAgent()
    agent.run_episodes(EPISODES)
