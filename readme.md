# Introduction
PotentLight, an innovative RL-based solution for adaptive traffic signal control, rooted in the theoretical framework of MP control

# Environment & Dependency:
- This code has been tested on Python 3.7, and compatibility with other versions is not guaranteed. It is recommended to use Python versions 3.5 and above.
- For installing CityFlow, it is recommended to follow the instructions provided at https://cityflow.readthedocs.io/en/latest/install.html.
  
|Name| Version |
|---|---------|
|Keras| v2.3.1   |
|tensorflow-gpu| 1.14.0  |
|CityFlow| 1.1.0   |
| tqdm | 4.65.0 |
| numpy | 1.21.5  |
| matplotlib |  3.5.3  |



# Files
* ``runexp.py``
  The main file of experiments where the args can be changed.
  
 > The **arg '--dataset'** requires special attention, and it should be consistent with the dataset being used. For example, config1, config2, and config3 correspond to '--dataset==jinan', while config4 corresponds to '--dataset==hangzhou'.
 
* ``agent.py``
  Implement RL agent for proposed method.

* ``cityflow_env.py``
  Define a simulator environment to interact with the simulator and obtain needed data.

* ``utility.py``
  Other functions for experiments.

* ``metric/travel_time.py`` & ``metric/throughput.py``
  Two representative measures as evaluation criteria to digitally assess the performance of the method.


# Datasets

This repo repository includes four real-world datasets. When extracting the 'data.zip' file, the resulting files will be stored in the 'project dir/data' directory.
 > The **storage path -- "dir"** to each dataset, as written in its corresponding JSON file, should be accurately specified based on your local machine's configuration.
  
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
