# Introduction
PotentLight, an innovative RL-based solution for adaptive traffic signal control, rooted in the theoretical framework of MP control

# Environment & Dependency:

|Type|Name| Version |
|---|---|---------|
|language|python| 3.7     |
|frame|Keras| v2.3.1   |
|backend|tensorflow-gpu| 1.14.0  |
|simulation platform|CityFlow| 1.1.0   |
|other | collections | 3.0.0  |
|other | tqdm | 4.65.0 |
|other | numpy | 1.21.5  |
|other | math | --  |
|other | matplotlib |  3.5.3  |
|other | argparse | --  |
|other | datetime | --   |
|other | time | --  |
|other | csv | --  |
|other | json | --  |


- CityFlow Installation Guide: https://cityflow.readthedocs.io/en/latest/install.html

# Files
* ``runexp.py``
  The main file of experiments where the args can be changed.

* ``agent.py``
  Implement RL agent for proposed method.

* ``cityflow_env.py``
  Define a simulator environment to interact with the simulator and obtain needed data.

* ``utility.py``
  Other functions for experiments.

* ``metric/travel_time.py`` & ``metric/throughput.py``
  Two representative measures as evaluation criteria to digitally assess the performance of the method.


# Datasets

This repo containes four real-world datasets, which are stored in the `./data`

| Configurations |City| Traffic flow |
|----------------|--|---------|
| Config 1       |Jinan | anon_3_4_jinan_real     |
| Config 2       |Jinan | anon_3_4_jinan_real_2000   |
| Config 3       |Jinan | anon_3_4_jinan_real_2500  |
| Config 4       |Hangzhou | anon_4_4_hangzhou_real   |


# How to run
In terminal:
```shell
cd project_dir
```
and then:
```shell
python3 runexp.py
```
