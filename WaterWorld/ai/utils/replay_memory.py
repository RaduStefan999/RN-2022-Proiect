import numpy as np
from collections import deque


class ReplayMemory:
    def __init__(self, max_size: int, min_size: int):
        self.min_replay_size = min_size
        self.memory = deque(maxlen=max_size)

    def __len__(self):
        return len(self.memory)

    def add(self, transition):
        self.memory.append(transition)

    def train_agent_batch(self, agent):
        if len(self.memory) > self.min_replay_size:
            states, targets = self._random_batch(agent)  # get a random batch
            return agent.model.train_on_batch(states, targets)  # ERR?
        else:
            return None

    def _random_batch(self, agent):
        inputs = np.zeros(agent.input_shape)
        targets = np.zeros((agent.batch_size, agent.num_actions))

        seen = []
        idx = agent.rng.randint(
            0,
            high=len(
                self.memory) -
            agent.num_frames -
            1)

        for i in range(agent.batch_size):
            while idx in seen:
                idx = agent.rng.randint(0, high=len(
                    self.memory) - agent.num_frames - 1)

            states = np.array([self.memory[idx + j][0]
                               for j in range(agent.num_frames + 1)])
            art = np.array([self.memory[idx + j][1:]
                            for j in range(agent.num_frames)])

            actions = art[:, 0].astype(int)
            rewards = art[:, 1]
            terminals = art[:, 2]

            state = states[:-1]
            state_next = states[1:]

            inputs[i, ...] = state.reshape(agent.state_shape)
            # we could make zeros but pointless.
            targets[i] = agent.predict_single(state)
            Q_prime = np.max(agent.predict_single(state_next))

            targets[i, actions] = rewards + \
                (1 - terminals) * (agent.discount * Q_prime)

            seen.append(idx)

        return inputs, targets
