from __future__ import division
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
        self.epsilon = self.epsilon_start

        self.batch_size = kwargs['batch_size']
        self.obs_to_start_training = kwargs.get('obs_to_start_training', self.batch_size)

        self.num_observations = 0

    def init(self, env): 
        self.env = env
        n_obs = env.observation_space.shape[0]
        n_act = env.action_space.n
        
        self.model.init(n_obs, n_act)

    def act(self, state):
        if self.num_observations < self.obs_to_start_training:
            self.epsilon = 1.0
        else:
            decay_factor = (self.num_observations - self.obs_to_start_training) / self.epsilon_decay
            decay_amount = (self.epsilon_end - self.epsilon_start) * decay_factor
            self.epsilon = np.maximum(self.epsilon_end, self.epsilon_start + decay_amount)
        
        if np.random.rand(1) > self.epsilon:
            action = np.argmax(self.model.predict(state))
        else:
            action = self.env.action_space.sample()

        return action

    def observe(self, state, action, next_state, reward, done):
        self.num_observations += 1
 
        self.memory.add(state.ravel(), action, next_state.ravel(), reward, done)

    def train(self):
        if self.memory.get_size() >= self.obs_to_start_training:
            states, actions, next_states, rewards, done = self.memory.sample(self.batch_size)
            
            target_qs = self.model.predict(states)
            next_qs = self.model.predict(next_states)
            not_done = ~(done.ravel())
            q_updates = rewards.ravel()
            q_updates[not_done] = rewards[not_done].ravel() + self.gamma * np.max(next_qs[not_done], axis=1)
            
            target_qs[np.arange(len(target_qs)), actions.ravel()] = q_updates
            
            self.model.fit(states, target_qs) 
