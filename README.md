[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/43851646-d899bf20-9b00-11e8-858c-29b5c2c94ccc.png "Crawler"

# AI Agent 🤖 to Control a Mechanical Arm 🦾

## Introduction

This project aims to explore the power of teaching an agent 
through Reinforced Learning (RL) to learn how to control an Arm.  

For this we are using a Deep Deterministic Policy Gradients (DDPG) algorithm 
to learn how to control efficiently the arm.

For this project, we are working with the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment.

### Environment

![Trained Agent][image1]

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

## Getting Started

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

## Distributed Training

For this project, we will provide you with two separate versions of the Unity environment:
- The first version contains a single agent.
- The second version contains 20 identical agents, each with its own copy of the environment.  

The second version is useful for algorithms like [PPO](https://arxiv.org/pdf/1707.06347.pdf), [A3C](https://arxiv.org/pdf/1602.01783.pdf), and [D4PG](https://openreview.net/pdf?id=SyZipzbCb) that use multiple (non-interacting, parallel) copies of the same agent to distribute the task of gathering experience.  

### Solving the Environment

Note that your project submission need only solve one of the two versions of the environment. 

#### Option 1: Solve the First Version

The task is episodic, and in order to solve the environment,  your agent must get an average score of +30 over 100 consecutive episodes.

#### Option 2: Solve the Second Version

The barrier for solving the second version of the environment is slightly different, to take into account the presence of many agents.  In particular, your agents must get an average score of +30 (over 100 consecutive episodes, and over all agents).  Specifically,
- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent.  This yields 20 (potentially different) scores.  We then take the average of these 20 scores. 
- This yields an **average score** for each episode (where the average is over all 20 agents).

The environment is considered solved, when the average (over 100 episodes) of those average scores is at least +30. 

## Instructions

### Files

#### Code
1. [agent.py](https://github.com/joao-d-oliveira/RL-SmartAgent-BananaGame/blob/main/agent.py) - Agent class containing Q-Learning algorithm and all supoprt for `Vanilla DQN`, `Double DQN`, `Dueling DQN` and `Priorized Replay Experience DQN`.
1. [model.py](https://github.com/joao-d-oliveira/RL-SmartAgent-BananaGame/blob/main/model.py) - DQN model class setup (containing configuration for `Dueling DQN`) 
1. [Navigation.ipynb](https://github.com/joao-d-oliveira/RL-SmartAgent-BananaGame/blob/main/Navigation.ipynb) - Jupyter Notebook for running experiment, with simple navigation (getting state space through vector)
---
1. [agent_vision.py](https://github.com/joao-d-oliveira/RL-SmartAgent-BananaGame/blob/main/agent_vision.py) - Agent class containing Q-Learning algorithm Visual environment
1. [model_vision.py](https://github.com/joao-d-oliveira/RL-SmartAgent-BananaGame/blob/main/model_vision.py) - DQN model class setup for Visual environment 
1. [Navigation_Pixels.ipynb](https://github.com/joao-d-oliveira/RL-SmartAgent-BananaGame/blob/main/Navigation_Pixels.ipynb) - Jupyter Notebook for running experiment, with pixel navigation (getting state space through pixeis)

#### Documentation
1. [README.md](https://github.com/joao-d-oliveira/RL-SmartAgent-BananaGame/blob/main/README.md) - This file
1. [Report.md](https://github.com/joao-d-oliveira/RL-SmartAgent-BananaGame/blob/main/Report.md) - Detailed Report on the project

#### Models
All models are saved on the subfolder (models).
For example, [checkpoint.pt](https://github.com/joao-d-oliveira/RL-SmartAgent-BananaGame/blob/main/models/checkpoint.pt) is 
a file which has been saved upon success of achieving the goal, and [model.pt](https://github.com/joao-d-oliveira/RL-SmartAgent-BananaGame/blob/main/models/model.pt) is the end model after runing all episodes.

### Running Normal navigation with state space of `37` dimensions

#### Structure of Notebook

The structure of the notebook follows the following:
> 1. Initial Setup: _(setup for parameters of experience, check [report](https://github.com/joao-d-oliveira/RL-SmartAgent-BananaGame/blob/main/Report.md) for more details)_ <br>
> 2. Navigation <br>
> 2.1 Start the Environment: _(load environment for the game)_<br>
> 2.2 HelperFunctions: _(functions to help the experience, such as Optuna, DQNsearch, ...)_<br>
> 2.3 Baseline DQN: _(section to train an agent with the standard parameters, without searching for hyper-parameters)_<br>
> 2.4 Vanilla DQN: _(section to train an agent with a Vanilla DQN)_<br>
> 2.5 Double DQN: _(section to train an agent with a Double DQN)_<br>
> 2.6 Dueling DQN: _(section to train an agent with a Dueling DQN)_<br>
> 2.7 Prioritized Experience Replay (PER) DQN: _(section to train an agent with a PER DQN)_<br>
> 2.8 Double DQN with PER: _(section to train an agent with a PER and Double DQN at same time)_<br>
> 2.9 Double with Dueling and PER DQN: _(section to train an agent with a PER and Double and dueling DQN)_<br>
> 3.0 Plot all results: _(section where all the results from above sections are plotted to compare performance)_

Each of the sections: [`2.3 Baseline DQN`, `2.4 Vanilla DQN`, `2.5 Double DQN`, `2.6 Dueling DQN`, `2.7 Prioritized Replay DQN`, `2.8 Double DQN with PER`, `2.9 Double with Dueling and PER DQN`]

Have subsessions:
> 2.x.1 Find HyperParameters (Optuna) <br>
> 2.x.1.1 Ploting Optuna Results <br>
> 2.x.2 Run (network) DQN <br>
> 2.x.3 Plot Scores <br>

Each section relevant to the respective DQN. <br>
You can choose whether to use the regular parameters, or try to find them through Optuna

#### Running

After fulling the requirements on section [Getting Started](https://github.com/joao-d-oliveira/RL-SmartAgent-BananaGame#getting-started) and at 
[requirements.txt](https://github.com/joao-d-oliveira/RL-SmartAgent-BananaGame/blob/main/requirements.txt) 
0. Load Jupyter notebook [Navigation.ipynb](https://github.com/joao-d-oliveira/RL-SmartAgent-BananaGame/blob/main/Navigation.ipynb)
1. Adapt dictionary `SETUP = {` with the desired paramenters
2. Load the environment. Running sections: 
   > 1 Initial Setup <br>
   > 2.1 Start the Environment <br>
   > 2.2. Helper Functions
3. Then go the section of the Network you want to run [`2.3 Baseline DQN`, `2.4 Vanilla DQN`, `2.5 Double DQN`, `2.6 Dueling DQN`, `2.7 Prioritized Replay DQN`, `2.8 Double DQN with PER`, `2.9 Double with Dueling and PER DQN`]
   There you will be able to either run Optuna to find the theoretically best parameters, or run the model with the base paramenters.

### (Optional) Challenge: Crawler Environment

After you have successfully completed the project, you might like to solve the more difficult **Crawler** environment.

![Crawler][image2]

In this continuous control environment, the goal is to teach a creature with four legs to walk forward without falling.  

You can read more about this environment in the ML-Agents GitHub [here](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#crawler).  To solve this harder task, you'll need to download a new Unity environment.  (**Note**: Udacity students should not submit a project with this new environment.)

You need only select the environment that matches your operating system:
- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Crawler/Crawler_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Crawler/Crawler.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Crawler/Crawler_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Crawler/Crawler_Windows_x86_64.zip)

Then, place the file in the `p2_continuous-control/` folder in the DRLND GitHub repository, and unzip (or decompress) the file.  Next, open `Crawler.ipynb` and follow the instructions to learn how to use the Python API to control the agent.

(_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Crawler/Crawler_Linux_NoVis.zip) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)




## Project Details
### Rules of The Game

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

### State Space
The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.
Given this information, the agent has to learn how to best select actions.

The state space of the **"Visual"** environment is composed by the snapshot of the video of the game, meaning that is 
an array composed by (84, 84, 3) which means, 84 of width and height and 3 channels (R.G.B.).

### Action Space
Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

### Conditions to consider solved

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

