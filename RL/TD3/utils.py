import numpy as np


class ReplayBuffer:
    def __init__(self, max_size=1000):
        self.buffer = []
        self.max_size = int(max_size)
        self.size = 0

    def add(self, transition):
        self.size += 1
        # transiton is tuple of (state, action, reward, next_state, done)
        self.buffer.append(transition)

    def sample(self, batch_size):
        # delete 1/5th of the buffer when full
        if self.size > self.max_size:
            del self.buffer[0:int(self.size / 5)]
            self.size = len(self.buffer)

        indexes = np.random.randint(0, len(self.buffer), size=batch_size)
        state, action, reward, next_state, done, state_full, next_state_full  = [], [], [], [], [], [], []

        for i in indexes:
            s, a, r, s_, d, s_f, s_full_ = self.buffer[i]
            state.append(np.array(s, copy=False))
            action.append(np.array(a, copy=False))
            reward.append(np.array(r, copy=False))
            next_state.append(np.array(s_, copy=False))
            done.append(np.array(d, copy=False))
            state_full.append(np.array(s_f, copy=False))
            next_state_full.append(np.array(s_full_, copy=False))

        return np.array(state), np.array(action), np.array(reward), np.array(
            next_state), np.array(done), np.array(state_full), np.array(
                next_state_full)
