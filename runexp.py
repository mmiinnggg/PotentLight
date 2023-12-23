from tqdm import tqdm
from datetime import datetime
from time import *
import csv

from cityflow_env import CityFlowEnvM
from agent import DQNAgent
from metric.travel_time import TravelTimeMetric
from metric.throughput import ThroughputMetric
from utility import *

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description='Run Example')
parser.add_argument('--thread', type=int, default=8, help='number of threads')
parser.add_argument('--num_step', type=int, default=3600, help='number of steps')
parser.add_argument('--time_interval', type=int, default=5, help='how often agent make decisions')
parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
###
parser.add_argument('--epoch', type=int, default=20, help='training episodes')
parser.add_argument("--save_rate", type=int, default=20, help="save model once every time this many episodes are completed")
parser.add_argument('--eta', type=int, default=0.1, help='')
parser.add_argument('--dataset', type=str, default='hangzhou', help='dataset name')
parser.add_argument('--identifier', type=str, default='config4', help='identifier')
###
parser.add_argument('--config_file', type=str, default='./data/config4.json', help='path of config file')
parser.add_argument('--save_dir', type=str, default="model/config4", help='directory in which model should be saved')
parser.add_argument('--log_dir', type=str, default="log/config4", help='directory in which logs should be saved')
args = parser.parse_args()

if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)
crt_time = datetime.now().strftime('%Y%m%d-%H%M%S')
log_name = args.log_dir + '/'+ args.identifier + '_potent_' + str(args.eta) + '_' + str(args.epoch)+  '_' +  crt_time + '.csv'
CsvFile = open(log_name, 'w')
CsvWriter = csv.writer(CsvFile)
CsvWriter.writerow(
    ["Mode", "episode", "step", "travel_time", "throughput"])
CsvFile.close()

def build(path=args.config_file):
    with open(path) as f:
        cityflow_config = json.load(f)

    config = {}
    roadnetFile = cityflow_config['dir'] + cityflow_config['roadnetFile']
    config["lane_phase_info"] = parse_roadnet(roadnetFile)

    intersection_id = list(config['lane_phase_info'].keys())
    config["intersection_id"] = intersection_id
    phase_list = {id_: config["lane_phase_info"][id_]["phase"] for id_ in intersection_id}
    config["phase_list"] = phase_list

    world = CityFlowEnvM(config["lane_phase_info"],
                       intersection_id,
                       eta = args.eta,
                       num_step=args.num_step,
                       thread_num=args.thread,
                       cityflow_config_file=args.config_file,
                       dataset=args.dataset
                       )
    print("\n world built.")

    config["state_size"] = 8
    agents = {}
    for id_ in intersection_id:
        agent = DQNAgent(intersection_id = id_,
                      state_size = config["state_size"],
                      action_size = len(phase_list[id_]),
                      batch_size = args.batch_size,
                      phase_list = phase_list[id_],
                      env = world)
        agents[id_] = agent
    print("agents built.")

    metrics = [TravelTimeMetric(world), ThroughputMetric(world)]

    return config, world, agents, metrics

def train(path = args.config_file):
    config, world, agents, metrics = build(path)
    print("training processing...")

    episode_travel_time = []
    total_step = 0
    total_decision_num = {id_: 0 for id_ in config["intersection_id"]}

    print("EPISODES:{} ".format(args.epoch))
    print("num_step:{} ".format(args.num_step))
    with tqdm(total=args.epoch * args.num_step / args.time_interval) as pbar:
        for e in range(args.epoch):
            action = {}
            action_phase = {}
            timing_phase = {}
            reward = {id_: 0 for id_ in config["intersection_id"]}
            rest_timing = {id_: 0 for id_ in config["intersection_id"]}

            episodes_decision_num = {id_: 0 for id_ in config["intersection_id"]}
            episodes_rewards = {id_: 0 for id_ in config["intersection_id"]}

            world.reset()
            for metric in metrics:
                metric.reset()
            state = {}
            last_phase = {}
            for id_ in config["intersection_id"]:
                state[id_] = world.get_state_(id_)

            i = 0
            while i < args.num_step:
                if i % args.time_interval == 0:
                    for id_ in config["intersection_id"]:
                        if rest_timing[id_] == 0:
                            for id_ in config["intersection_id"]:
                                last_phase[id_] = world.current_phase[id_]
                                action[id_] = agents[id_].get_action([world.current_phase[id_]], state[id_])
                                action_phase[id_] = config["phase_list"][id_][action[id_]]

                                p, timing_phase[id_] = world.get_timing_(id_, action_phase[id_])
                                rest_timing[id_] = timing_phase[id_]

                    for _ in range(args.time_interval):
                        next_state, reward_, t1 = world.step(action_phase, i)
                        i += 1
                    for id_ in rest_timing:
                        rest_timing[id_] -= args.time_interval
                    for metric in metrics:
                        metric.update()

                    for id_ in config["intersection_id"]:
                        agents[id_].remember(state[id_], last_phase[id_], action_phase[id_], reward_[id_], next_state[id_], world.current_phase[id_])
                        total_decision_num[id_] += 1
                        episodes_decision_num[id_] += 1
                        episodes_rewards[id_] += reward[id_]
                        state[id_] = next_state[id_]

                for id_ in config["intersection_id"]:
                    agents[id_].replay()

                total_step += 1
                pbar.update(1)
                pbar.set_description(
                    "t_st:{}, epi:{}, st:{}".format(total_step, e+1, i))

            if e % args.save_rate == args.save_rate - 1:
                if not os.path.exists(args.save_dir):
                    os.makedirs(args.save_dir)
                for id_ in config["intersection_id"]:
                    agents[id_].save_model(args.save_dir, e)

            episode_travel_time.append(world.eng.get_average_travel_time())
            print('\n Epoch {} travel time:'.format(e+1), world.eng.get_average_travel_time())
            for metric in metrics:
                print(f"\t{metric.name}: {metric.eval()}")

            mean_reward = {id_: [] for id_ in config["intersection_id"]}
            for id_ in config["intersection_id"]:
                mean_reward[id_] = episodes_rewards[id_] / episodes_decision_num[id_]

            CsvFile = open(log_name, 'a+')
            CsvWriter = csv.writer(CsvFile)
            CsvWriter.writerow(
                ["-", e+1, i, metrics[0].eval(), metrics[1].eval()])
            CsvFile.close()

    plot_data_lists([episode_travel_time], ['travel time'], figure_name=args.log_dir + '/'+ args.identifier + '_potent_'  + str(args.eta) + '_' + str(args.epoch)+  '_' +  crt_time  + '_travel time.pdf')

def test(path = args.config_file):
    config, world, agents, metrics = build(path)
    print("testing processing...")

    total_step = 0
    with tqdm(total= args.num_step / args.time_interval) as pbar:
        action = {}
        action_phase = {}
        timing_phase = {}
        reward = {id_: 0 for id_ in config["intersection_id"]}
        rest_timing = {id_: 0 for id_ in config["intersection_id"]}
        pressure = {id_: [] for id_ in config["intersection_id"]}

        episodes_decision_num = {id_: 0 for id_ in config["intersection_id"]}
        episodes_rewards = {id_: 0 for id_ in config["intersection_id"]}

        world.reset()
        for metric in metrics:
            metric.reset()
        state = {}
        last_phase = {}
        for id_ in config["intersection_id"]:
            state[id_] = world.get_state_(id_)
            agents[id_].load_model(args.save_dir, args.epoch - 1)
        print("agents loaded...")

        i = 0
        while i < args.num_step:
            if i % args.time_interval == 0:
                for id_ in config["intersection_id"]:
                    if rest_timing[id_] == 0:
                        for id_ in config["intersection_id"]:
                            last_phase[id_] = world.current_phase[id_]
                            action[id_] = agents[id_].get_action([world.current_phase[id_]], state[id_])
                            action_phase[id_] = config["phase_list"][id_][action[id_]]

                            p, timing_phase[id_] = world.get_timing_(id_, action_phase[id_])
                            rest_timing[id_] = timing_phase[id_]

                if i % 20 == 0:
                    for id_ in config["intersection_id"]:
                        p = world.get_pressure_(id_)
                        pressure[id_].append(p)

                for _ in range(args.time_interval):
                    next_state, reward_, t1 = world.step(action_phase, i)
                    i += 1
                for id_ in rest_timing:
                    rest_timing[id_] -= args.time_interval
                for metric in metrics:
                    metric.update()

                for id_ in config["intersection_id"]:
                    agents[id_].remember(state[id_], last_phase[id_], action_phase[id_], reward_[id_], next_state[id_], world.current_phase[id_])
                    episodes_decision_num[id_] += 1
                    episodes_rewards[id_] += reward[id_]
                    state[id_] = next_state[id_]

            total_step += 1
            pbar.update(1)
            pbar.set_description(
                "t_st:{}, epi:{}, st:{} ".format(total_step, 0, i+1))

        print('\n Test Epoch {} travel time:'.format(0), world.eng.get_average_travel_time())
        for metric in metrics:
            print(f"\t{metric.name}: {metric.eval()}")

        mean_reward = {id_: [] for id_ in config["intersection_id"]}
        for id_ in config["intersection_id"]:
            mean_reward[id_] = episodes_rewards[id_] / episodes_decision_num[id_]

        '''
        for id_ in config["intersection_id"]:
            with open(args.log_dir + '/'+ args.identifier + '_potent_pressure_' + id_  + '_' + crt_time + '.txt', 'a+') as ttf:
                ttf.write(str(pressure[id_]) + "\n\n")
            ttf.close()
        '''

        CsvFile = open(log_name, 'a+')
        CsvWriter = csv.writer(CsvFile)
        CsvWriter.writerow(
            ["test", "-", i, metrics[0].eval(), metrics[1].eval()])
        CsvFile.close()

    return world.eng.get_average_travel_time()

if __name__ == '__main__':
    start_time = time()

    train()
    test()

    end_time = time()
    run_time = end_time - start_time
    print('Run time:', run_time)

