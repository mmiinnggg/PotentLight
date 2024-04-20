import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam, RMSprop, SGD
from keras.layers import Input, Dense, Conv2D, Flatten
from keras.models import Model
from keras.layers.merge import concatenate
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

class DQNAgent(object):
    def __init__(self,
                 intersection_id,
                 state_size=9,
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
        self.state_size = 8

        self.memory_size = 2000
        self.memory = deque(maxlen=self.memory_size)

        self.learning_start = 2000
        self.update_model_freq = 1
        self.update_target_model_freq = 20

        self.gamma = 0.95  # discount rate
        self.epsilon = 0.1  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001 #0.005
        self.d_dense = 20
        self.n_layer = 2

        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_network()

        self.step = 0
        self.phase_list = phase_list
        self.timing_list = timing_list

    def _build_model(self):
        '''Initialize a Q network'''

        dic_input_node = {"feat1": Input(shape=(1,), name="input_cur_phase"),
                          "feat2": Input(shape=(8,), name="input_feat2")}

        list_all_flatten_feature = []
        for feature_name in dic_input_node:
            list_all_flatten_feature.append(dic_input_node[feature_name])
        all_flatten_feature = concatenate(list_all_flatten_feature, axis=1, name="all_flatten_feature")

        locals()["dense_0"] = Dense(self.d_dense, activation="relu", name="dense_0")(all_flatten_feature)
        for i in range(1, self.n_layer):
            locals()["dense_%d" % i] = Dense(self.d_dense, activation="relu", name="dense_%d" % i)(
                locals()["dense_%d" % (i - 1)])
        q_values = Dense(self.action_size, activation="linear", name="q_values")(
            locals()["dense_%d" % (self.n_layer - 1)])
        network = Model(inputs=[dic_input_node[feature_name] for feature_name in ["feat1", "feat2"]],
                        outputs=q_values)
        network.compile(optimizer=RMSprop(lr=self.learning_rate),
                        loss="mean_squared_error")
        # network.summary()
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

    def update_target_network(self):
        weights = self.model.get_weights()
        self.target_model.set_weights(weights)

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
