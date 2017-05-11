import numpy as np
from sklearn.preprocessing import label_binarize

class QAgent:
    def __init__(self, model, memory, **kwargs):
        self.model = model
        self.memory = memory

        self.gamma = kwargs['gamma']

        self.epsilon_decay = kwargs['epsilon_decay']
        self.epsilon_start = kwargs['epsilon_start']
        self.epsilon_end = kwargs['epsilon_end']

        self.batch_size = kwargs['batch_size']

        self.num_observations = 0

    def init(self, env): 
        self.env = env
        n_obs = env.observation_space.shape[0]
        n_act = env.action_space.n
        
        self.model.init(n_obs, n_act)
        self.memory.init(n_obs, n_act)

    def act(self, state):
        epsilon = self.epsilon_start * np.exp(-self.num_observations * self.epsilon_decay)
        epsilon += self.epsilon_end
        if np.random.rand(1) > epsilon:
            action = np.argmax(self.model.predict(state))
            is_random = False
        else:
            action = self.env.action_space.sample()
            is_random = True

        return action, is_random

    def observe(self, state, action, next_state, reward, done):
        self.num_observations += 1

        target_qs = self.model.predict(state).flatten()
        next_qs = self.model.predict(next_state).flatten()
        if done:
            q_update = reward
        else:
            q_update = reward + self.gamma * np.amax(next_qs)
        target_qs[action] = q_update
        
        self.memory.add(state, target_qs)

    def train(self):
        X, y = self.memory.sample(self.batch_size)
        if X is not None:
            self.model.fit(X, y)
        
class ProbAgent:
    def __init__(self, q_model, a_model, memory, **kwargs):
        self.q_model = q_model
        self.a_model = a_model
        self.memory = memory

        self.gamma = kwargs['gamma']
        self.batch_size = kwargs['batch_size']

        self.num_observations = 0

    def init(self, env): 
        self.env = env
        n_obs = env.observation_space.shape[0]
        n_act = env.action_space.n
        
        self.q_model.init(n_obs, n_act)
        self.a_model.init(n_obs, n_act)
        self.memory.init(n_obs, n_act)

    def act(self, state):
        action_probs = self.a_model.predict(state)
        action = np.random.choice(range(self.env.action_space.n), p=action_probs.ravel())

        return action, False

    def observe(self, state, action, next_state, reward, done):
        self.num_observations += 1
        
        target_qs = self.q_model.predict(state).flatten()
        next_qs = self.q_model.predict(next_state).flatten()
        if done:
            q_update = reward
        else:
            q_update = reward + self.gamma * np.amax(next_qs)
        
        target_qs[action] = q_update
        self.memory.add(state, target_qs)

    def train(self):
        X, y = self.memory.sample(self.batch_size)
        if X is not None:
            self.q_model.fit(X, y)
            classes = np.argmax(self.q_model.predict(X), axis=1)
            correct_actions = np.zeros(y.shape)
            for i in range(y.shape[1]):
                correct_actions[classes == i, i] = 1
            self.a_model.fit(X, correct_actions)
        
