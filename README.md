[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/43851646-d899bf20-9b00-11e8-858c-29b5c2c94ccc.png "Crawler"

# AI Agent 🤖 to Control a Mechanical Arm 🦾

## Introduction

This project aims to explore the power of teaching an agent 
through Reinforced Learning (RL) to learn how to control an Arm.  

For this we are using a Deep Deterministic Policy Gradients (DDPG) algorithm 
to learn how to control efficiently the arm.

We are working with the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment from OpenAI.

## Project Details

### Environment

![Trained Agent][image1]

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

### Space State
The observation space consists of 33 variables corresponding to position, rotation, velocity, 
and angular velocities of the arm.

You can run the environment with One or Multiple Agents

So for example, in the case of Multi Agent version (2º version):

```
There are 20 agents. Each observes a state with length: 33
The state for the first agent looks like: [ 0.00000000e+00 -4.00000000e+00  0.00000000e+00  1.00000000e+00
 -0.00000000e+00 -0.00000000e+00 -4.37113883e-08  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00 -1.00000000e+01  0.00000000e+00
  1.00000000e+00 -0.00000000e+00 -0.00000000e+00 -4.37113883e-08
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00  5.75471878e+00 -1.00000000e+00
  5.55726624e+00  0.00000000e+00  1.00000000e+00  0.00000000e+00
 -1.68164849e-01]
```

### Conditions to consider solved

For submission, one didn't need to solve with One and Multiple agents, but I tried to 
show that it was easy to do so with few adaptions to the code.
And wanted to compare the performance of the algorithm with the different environments as indicated at 
the [Report.md](https://github.com/joao-d-oliveira/RL-SmartAgent-BananaGame/blob/main/Report.md). 

#### Option 1: Solve the First Version

The task is episodic, and in order to solve the environment,  your agent must get an average score of +30 over 100 consecutive episodes.

#### Option 2: Solve the Second Version

The barrier for solving the second version of the environment is slightly different, to take into account the presence of many agents.  In particular, your agents must get an average score of +30 (over 100 consecutive episodes, and over all agents).  Specifically,
- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent.  This yields 20 (potentially different) scores.  We then take the average of these 20 scores. 
- This yields an **average score** for each episode (where the average is over all 20 agents).

The environment is considered solved, when the average (over 100 episodes) of those average scores is at least +30. 

## Distributed Training

For this project, we can either use:
- The first version of the environment with a single agent.
- The second version contains 20 identical agents, each with its own copy of the environment.  

For this we used the DDPG algorithm for both the 1st and 2nd version, so with 1 or 20 agents, and noticed that with the Multi version the results are much faster
(see [Report.md](https://github.com/joao-d-oliveira/RL-SmartAgent-BananaGame/blob/main/Report.md) for further info)

With the optional challenge [Crawler](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#crawler), 
we see that DDPG is not enough (or too slow) to converge, so for this I tried to search for another algorithm such 
as [PPO](https://arxiv.org/pdf/1707.06347.pdf).

## Instructions

### Getting Started

Before starting any of the environments you need to install 
`unityagents` package from Unity (version 0.4.0).
As this is an old version, the best is to download the
[python folder](https://github.com/joao-d-oliveira/RL-RobotArm/blob/main/pythonx) and run 
`cd python; pip install  .` to install it manually.

#### For Reacher Environment 

1. You need to have installed the requirements (specially mlagents==0.4.0).
   Due to deprecated libraries, I've included a [python folder](https://github.com/joao-d-oliveira/RL-SmartAgent-BananaGame/tree/main/python) which will help
   with installation of the system.
      - Clone the repository: `git clone https://github.com/joao-d-oliveira/RL-RobotArm.git`
      - Go to python folder: `cd RL-RobotArm/python`
      - Compile and install needed libraries `pip install .`
2. Download the environment from one of the links below
   Download only the environment that matches your operating system:
    - **_Version 1: One (1) Agent_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

    - **_Version 2: Twenty (20) Agents_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux_NoVis.zip) (version 1) or [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip) (version 2) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

3. Place the downloaded file for your environment in the DRLND GitHub repository, in the ``RL-RobotArm`` folder, and unzip (or decompress) the file.

#### For Crawler Environment

![Crawler][image2]

In this continuous control environment, the goal is to teach a creature with four legs to walk forward without falling.  

You can read more about this environment in the ML-Agents GitHub [here](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#crawler).  To solve this harder task, you'll need to download a new Unity environment.  (**Note**: Udacity students should not submit a project with this new environment.)

You need only select the environment that matches your operating system:
- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Crawler/Crawler_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Crawler/Crawler.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Crawler/Crawler_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Crawler/Crawler_Windows_x86_64.zip)

Then, place the file in the `RL-RobotArm` folder in the DRLND GitHub repository, and unzip (or decompress) the file.  Next, open `Crawler.ipynb` and follow the instructions to learn how to use the Python API to control the agent.

(_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Crawler/Crawler_Linux_NoVis.zip) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)


### Action Space
Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

### Files

#### Code
1. [utils/agent_reacher_ddpg.py](https://github.com/joao-d-oliveira/RL-RobotArm/blob/main/utils/agent_reacher_ddpg.py) - Agent class containing methods to help the agent learn and acquire knowledge using DDPG algorithm
1. [utils/model_reacher.py](https://github.com/joao-d-oliveira/RL-RobotArm/blob/main/utils/model_reacher.py) - DDQG model of Actor and Critic class setup 
1. [Continuous_Control.ipynb](https://github.com/joao-d-oliveira/RL-RobotArm/blob/main/Continuous_Control.ipynb) - Jupyter Notebook for running experiment, for Reacher Control
---
1. [utils/agent_crawler_ppo.py](https://github.com/joao-d-oliveira/RL-RobotArm/blob/main/utils/agent_crawler_ppo.py) - Agent class containing methods to help the agent learn and acquire knowledge using PPO algorithm for Crawler environment
1. [utils/model_crawler.py](https://github.com/joao-d-oliveira/RL-RobotArm/blob/main/utils/model_crawler.py) - PPO model of Actor and Critic class setup 
1. [Crawler.ipynb](https://github.com/joao-d-oliveira/RL-RobotArm/blob/main/Crawler.ipynb) - Jupyter Notebook for running experiment, for Crawler Control

#### Documentation
1. [README.md](https://github.com/joao-d-oliveira/RL-RobotArm/blob/main/README.md) - This file
1. [Report.md](https://github.com/joao-d-oliveira/RL-RobotArm/blob/main/Report.md) - Detailed Report on the project

#### Models
All models are saved on the subfolder ([saved_models](https://github.com/joao-d-oliveira/RL-RobotArm/blob/main/saved_models)).
For example, [checkpoint_Mult_actor.pth](https://github.com/joao-d-oliveira/RL-RobotArm/blob/main/saved_models/checkpoint_Mult_actor.pth) 
and [checkpoint_Mult_critic.pth](https://github.com/joao-d-oliveira/RL-RobotArm/blob/main/saved_models/checkpoint_Mult_critic.pth) are 
files which has been saved upon success of achieving the goal, and
[finished_Reacher_Mult_actor.pth](https://github.com/joao-d-oliveira/RL-RobotArm/blob/main/saved_models/finished_Reacher_Mult_actor.pth) 
and [finished_Reacher_Mult_critic.pth](https://github.com/joao-d-oliveira/RL-RobotArm/blob/main/saved_models/finished_Reacher_Mult_critic.pth) 
are the end model after runing all episodes.

### Running Reacher training

#### Structure of Notebook

The structure of the notebook follows the following:
> 1. Initial Setup: _(setup for parameters of experience, check [report](https://github.com/joao-d-oliveira/RL-RobotArm/blob/main/Report.md) for more details)_ <br>
> 2. Continuous Control <br>
> 2.1 Start the Environment: _(load environment for the game)_<br>
> 2.2 Helper Functions: _(functions to help the experience, such as Optuna, DDPG, ...)_<br>
> 2.3 Baseline: _(section to train an agent with the standard parameters, without searching for hyper-parameters)_<br>
> 2.4 HyperParameters: _(section to train an agent with hyperparameters)_<br>
> 3.0 Plot all results: _(section where all the results from above sections are plotted to compare performance)_

At Initial Setup, you can define whether you want 
One or Multiple agents by turning at `SETUP` dictionary:
- `'MULTI_ONE': 'One',  # 'Mult' or 'One'` 

#### Running

After fulling the requirements on section [Getting Started](https://github.com/joao-d-oliveira/RL-RobotArm#getting-started) and at 
[requirements.txt](https://github.com/joao-d-oliveira/RL-RobotArm/blob/main/requirements.txt) 
0. Load Jupyter notebook [Navigation.ipynb](https://github.com/joao-d-oliveira/RL-SmartAgent-BananaGame/blob/main/Navigation.ipynb)
1. Adapt dictionary `SETUP = {` with the desired paramenters
2. Load the environment. Running sections: 
   > 1 Initial Setup <br>
   > 2.1 Start the Environment <br>
   > 2.2. Helper Functions

### Running Crawler Environment

After fulling the requirements on section [Getting Started](https://github.com/joao-d-oliveira/RL-RobotArm#getting-started) and at 
[requirements.txt](https://github.com/joao-d-oliveira/RL-RobotArm/blob/main/requirements.txt) 
0. Load Jupyter notebook [Crawler.ipynb](https://github.com/joao-d-oliveira/RL-SmartAgent-BananaGame/blob/main/Crawler.ipynb)
1. Run all cells, no special options needed

