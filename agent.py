import random
import numpy as np
from collections import deque
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
import tensorflow as tf
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

class DQNAgent(object):
    def __init__(self,
                 intersection_id,
                 state_size=8,
                 action_size=8,
                 batch_size=32,
                 phase_list=[],
                 timing_list = [],
                 env=None
                 ):
        self.env = env
        self.intersection_id = intersection_id
        self.action_size = action_size
        self.batch_size = batch_size #32
        self.state_size = state_size

        self.memory_size = 2000
        self.memory = deque(maxlen=self.memory_size)

        self.gamma = 0.95  # discount rate
        self.epsilon = 0.1  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001 

        self.model = self._build_model()

        self.step = 0
        self.phase_list = phase_list
        self.timing_list = timing_list

    def _build_model(self, mode='action'):
        '''Initialize a Q network'''
        feat0 = Input(shape=(1,))
        feat1 = Input(shape=(8,))

        all_flatten_feature = tf.concat([feat0, feat1], axis=1, name="all_flatten_feature")
        dense0 = Dense(20, activation="sigmoid")(all_flatten_feature)
        dense1 = Dense(20, activation="sigmoid")(dense0)
        q_values = Dense(self.action_size, activation="linear", name="q_values")(dense1)

        network = Model(inputs=[feat0, feat1],
                        outputs=q_values)
        network.compile(optimizer=RMSprop(lr=self.learning_rate),
                        loss="mean_squared_error")
        network.summary()
        return network

    def _reshape_ob(self, ob):
        return np.reshape(ob, (1, -1))

    def get_action(self, phase, ob):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        ob = self._reshape_ob(ob)
        act_values = self.model.predict([phase, ob])
        return np.argmax(act_values[0])
    
    def sample(self):
        return random.randrange(self.action_size)

    def remember(self, ob, phase, action, reward, next_ob, next_phase):
        action = self.phase_list.index(action)
        self.memory.append((ob, phase, action, reward, next_ob, next_phase))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        self.step += 1
        state_t = self.memory[-1][-1]
        replay_batch = random.sample(self.memory, self.batch_size)
        s_batch = np.reshape(np.array([replay[0] for replay in replay_batch]), [self.batch_size, self.state_size])
        phases = np.reshape(np.array([replay[1] for replay in replay_batch]), [self.batch_size, 1])
        next_s_batch = np.reshape(np.array([replay[4] for replay in replay_batch]), [self.batch_size, self.state_size])
        next_phases = np.reshape(np.array([replay[5] for replay in replay_batch]), [self.batch_size, 1])

        Q = self.model.predict([phases, s_batch])
        Q_next = self.model.predict([next_phases, next_s_batch])

        lr = 1
        for i, replay in enumerate(replay_batch):
            _, _, a, reward, state_n, _ = replay
            if (state_t == state_n).all():
                Q[i][a] = (1 - lr) * Q[i][a] + lr * reward
            else:
                Q[i][a] = (1 - lr) * Q[i][a] + lr * (reward + self.gamma * np.amax(Q_next[i]))
        self.model.fit([phases, s_batch], Q, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load_model(self, dir="model/dqn", e = 0):
        name = "dqn_agent_{}_{}.h5".format(self.intersection_id, e)
        model_name = os.path.join(dir, name)
        self.model.load_weights(model_name)

    def save_model(self, dir="model/dqn", e = 0):
        name = "dqn_agent_{}_{}.h5".format(self.intersection_id, e)
        model_name = os.path.join(dir, name)
        self.model.save_weights(model_name)
