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
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size #32

        self.memory_size = 2000
        self.memory = deque(maxlen=self.memory_size)

        self.learning_start = 2000
        self.update_model_freq = 1
        self.update_target_model_freq = 20

        self.gamma = 0.95  # discount rate
        self.epsilon = 0.1  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.005 #0.001

        self.dic_agent_conf = {
            "LEARNING_RATE": 0.001,
            "SAMPLE_SIZE": 1000,
            "BATCH_SIZE": 20,
            "EPOCHS": 100,
            "UPDATE_Q_BAR_FREQ": 5,
            "UPDATE_Q_BAR_EVERY_C_ROUND": False,
            "GAMMA": 0.8,
            "MAX_MEMORY_LEN": 10000,
            "PATIENCE": 10,
            "D_DENSE": 20,
            "N_LAYER": 2,
            "EPSILON": 0.8,
            "EPSILON_DECAY": 0.95,
            "MIN_EPSILON": 0.2,
            "LOSS_FUNCTION": "mean_squared_error",
            "SEPARATE_MEMORY": False,
            "NORMAL_FACTOR": 20,
            "TRAFFIC_FILE": "cross.2phases_rou01_equal_450.xml",
        }
        self.dic_traffic_env_conf = {
            "ACTION_PATTERN": "set",
            "NUM_INTERSECTIONS": 1,
            "MIN_ACTION_TIME": 10,
            "YELLOW_TIME": 5,
            "ALL_RED_TIME": 0,
            "NUM_PHASES": 2,
            "NUM_LANES": 1,
            "ACTION_DIM": 2,
            "MEASURE_TIME": 10,
            "IF_GUI": True,
            "DEBUG": False,

            "INTERVAL": 1,
            "THREADNUM": 8,
            "SAVEREPLAY": True,
            "RLTRAFFICLIGHT": True,

            "DIC_FEATURE_DIM": dict(
                D_LANE_QUEUE_LENGTH=(4,),
                D_LANE_NUM_VEHICLE=(4,),

                D_COMING_VEHICLE=(4,),
                D_LEAVING_VEHICLE=(4,),

                D_LANE_NUM_VEHICLE_BEEN_STOPPED_THRES1=(4,),
                D_CUR_PHASE=(1,),
                D_NEXT_PHASE=(1,),
                D_TIME_THIS_PHASE=(1,),
                D_TERMINAL=(1,),
                D_LANE_SUM_WAITING_TIME=(4,),
                D_VEHICLE_POSITION_IMG=(4, 60,),
                D_VEHICLE_SPEED_IMG=(4, 60,),
                D_VEHICLE_WAITING_TIME_IMG=(4, 60,),

                D_PRESSURE=(1,),

                D_ADJACENCY_MATRIX=(2,)
            ),

            "LIST_STATE_FEATURE": [
                "cur_phase",
                # "time_this_phase",
                # "vehicle_position_img",
                # "vehicle_speed_img",
                # "vehicle_acceleration_img",
                # "vehicle_waiting_time_img",
                "lane_num_vehicle",
                # "lane_num_vehicle_been_stopped_thres01",
                # "lane_num_vehicle_been_stopped_thres1",
                # "lane_queue_length",
                # "lane_num_vehicle_left",
                # "lane_sum_duration_vehicle_left",
                # "lane_sum_waiting_time",
                # "terminal",

                # "coming_vehicle",
                # "leaving_vehicle",
                # "pressure",

                # "adjacency_matrix"

            ],

            "DIC_REWARD_INFO": {
                "flickering": 0,
                "sum_lane_queue_length": 0,
                "sum_lane_wait_time": 0,
                "sum_lane_num_vehicle_left": 0,
                "sum_duration_vehicle_left": 0,
                "sum_num_vehicle_been_stopped_thres01": 0,
                "sum_num_vehicle_been_stopped_thres1": -0.25,
                "pressure": 0,
            },

            "LANE_NUM": {
                "LEFT": 1,
                "RIGHT": 1,
                "STRAIGHT": 1
            },

            "PHASE": [
                'WSES',
                'NSSS',
                'WLEL',
                'NLSL',
                # 'WSWL',
                # 'ESEL',
                # 'NSNL',
                # 'SSSL',
            ],

        }

        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_network()

        self.step = 0
        self.phase_list = phase_list
        self.timing_list = timing_list

    def _build_model_(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(100, input_dim=self.state_size, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def _build_model(self):
        '''Initialize a Q network'''

        # initialize feature node
        dic_input_node = {}
        for feature_name in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]:
            if "phase" in feature_name or "adjacency" in feature_name:
                _shape = self.dic_traffic_env_conf["DIC_FEATURE_DIM"]["D_" + feature_name.upper()]
            else:
                # _shape = (self.ob_length,)
                _shape = (self.state_size,)
                # _shape = (self.dic_traffic_env_conf["DIC_FEATURE_DIM"]["D_"+feature_name.upper()])
            dic_input_node[feature_name] = Input(shape=_shape,
                                                 name="input_" + feature_name)

        # add cnn to image features
        dic_flatten_node = {}
        for feature_name in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]:
            if len(self.dic_traffic_env_conf["DIC_FEATURE_DIM"]["D_" + feature_name.upper()]) > 1:
                dic_flatten_node[feature_name] = Flatten()(dic_input_node[feature_name])
            else:
                dic_flatten_node[feature_name] = dic_input_node[feature_name]

        # concatenate features
        list_all_flatten_feature = []
        for feature_name in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]:
            list_all_flatten_feature.append(dic_flatten_node[feature_name])
        all_flatten_feature = concatenate(list_all_flatten_feature, axis=1, name="all_flatten_feature")

        # shared dense layer, N_LAYER
        locals()["dense_0"] = Dense(self.dic_agent_conf["D_DENSE"], activation="relu", name="dense_0")(
            all_flatten_feature)
        for i in range(1, self.dic_agent_conf["N_LAYER"]):
            locals()["dense_%d" % i] = Dense(self.dic_agent_conf["D_DENSE"], activation="relu", name="dense_%d" % i)(
                locals()["dense_%d" % (i - 1)])
        # dense1 = Dense(self.dic_agent_conf["D_DENSE"], activation="relu", name="dense_1")(all_flatten_feature)
        # dense2 = Dense(self.dic_agent_conf["D_DENSE"], activation="relu", name="dense_2")(dense1)
        q_values = Dense(self.action_size, activation="linear", name="q_values")(
            locals()["dense_%d" % (self.dic_agent_conf["N_LAYER"] - 1)])
        network = Model(inputs=[dic_input_node[feature_name]
                                for feature_name in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]],
                        outputs=q_values)
        network.compile(optimizer=RMSprop(lr=self.dic_agent_conf["LEARNING_RATE"]),
                        loss=self.dic_agent_conf["LOSS_FUNCTION"])
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

    def choose_action(self, state):
        if np.random.uniform() < self.epsilon:
        # if np.random.uniform() < self.epsilon - self.step * 0.0002:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def sample(self):
        return random.randrange(self.action_size)

    def update_target_network(self):
        weights = self.model.get_weights()
        self.target_model.set_weights(weights)

    '''
    def remember_timing(self, state, timing, reward, next_state):
        timing = self.timing_list.index(timing)  # index
        self.memory.append((state, timing, reward, next_state))
    '''

    def remember_(self, state, action, reward, next_state):
        action = self.phase_list.index(action)  # index
        self.memory.append((state, action, reward, next_state))

    def remember(self, ob, phase, action, reward, next_ob, next_phase):
        action = self.phase_list.index(action)
        self.memory.append((ob, phase, action, reward, next_ob, next_phase))

    def replay__(self): # Timing if flag for agent_timing
        if self.batch_size > len(self.memory):
            minibatch = self.memory
        else:
            minibatch = random.sample(self.memory, self.batch_size)
        obs, phases, actions, rewards, next_obs, next_phases = [np.stack(x) for x in np.array(minibatch).T]
        target = rewards + self.gamma * np.amax(self.target_model.predict([next_phases, next_obs]), axis=1)
        target_f = self.model.predict([phases, obs])
        for i, action in enumerate(actions):
            target_f[i][action] = target[i]
        history = self.model.fit([phases, obs], target_f, epochs=1, verbose=0)
        # print(history.history['loss'])
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def replay(self): # Timing if flag for agent_timing
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
            # ob, phase, action, reward, next_ob, next_phase
            # state, action, reward, next_state
            _, _, a, reward, state_n, _ = replay
            if (state_t == state_n).all():
                Q[i][a] = (1 - lr) * Q[i][a] + lr * reward
            else:
                Q[i][a] = (1 - lr) * Q[i][a] + lr * (reward + self.gamma * np.amax(Q_next[i]))
        # 传入网络训练
        # print("s_batch:\n", s_batch, "Q:\n", Q)
        self.model.fit([phases, s_batch], Q, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def replay_(self): # Timing if flag for agent_timing
        if len(self.memory) < self.batch_size:
            return
        self.step += 1
        state_t = self.memory[-1][-1]
        replay_batch = random.sample(self.memory, self.batch_size)
        s_batch = np.reshape(np.array([replay[0] for replay in replay_batch]), [self.batch_size, self.state_size])
        next_s_batch = np.reshape(np.array([replay[3] for replay in replay_batch]),
                                      [self.batch_size, self.state_size])

        Q = self.model.predict(s_batch)
        Q_next = self.model.predict(next_s_batch)

        lr = 1
        for i, replay in enumerate(replay_batch):
            _, a, reward, state_n = replay
            if (state_t == state_n).all():
                Q[i][a] = (1 - lr) * Q[i][a] + lr * reward
            else:
                Q[i][a] = (1 - lr) * Q[i][a] + lr * (reward + self.gamma * np.amax(Q_next[i]))

        # 传入网络训练
        # print("s_batch:\n", s_batch, "Q:\n", Q)
        self.model.fit(s_batch, Q, verbose=0)

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
