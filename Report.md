[image-reacher1]: https://github.com/joao-d-oliveira/RL-RobotArm/blob/master/Images/generated_run_reacher.gif?raw=true "Trained Agent"
[image-crawler1]: https://github.com/joao-d-oliveira/RL-RobotArm/blob/master/Images/generated_run_crawler.gif?raw=true "Trained Agent"
[optuna1]: https://github.com/joao-d-oliveira/RL-RobotArm/blob/master/Images/optuna_run_history.png?raw=true "Optuna Detail Run"
[optuna2]: https://github.com/joao-d-oliveira/RL-RobotArm/blob/master/Images/optuna_parallel_coordinate.png?raw=true "Optuna Parameters Coordinates"
[optuna3]: https://github.com/joao-d-oliveira/RL-RobotArm/blob/master/Images/optuna_parameters_importance.png?raw=true "Optuna Parameters Importance"
[rolling]: https://github.com/joao-d-oliveira/RL-RobotArm/blob/master/Images/run_allrolling.png?raw=true "All Rolling Averages"

# AI Agent 🤖 to Control a Mechanical Arm 🦾 [**Report**]

------

![Trained Agent with DDPG for Reacher][image-reacher1]<br>
Image of player playing with DDPG

![Trained Agent with PPO for crawler][image-crawler1]<br>
Image of player playing with PPO


## Introduction
This project aimed to get a sense and discover the potential of 
algorithms such as Deep Deterministic Policy Gradients (DDPG) with Single or 
Multiple Agents in the field of Reinforced Learning (RL).

Initially this project aimed to use a basic DDPG 
(with pytorch) to solve the problem of an AI agent trying to control a mechanical arm.

In case sucessfully implementation, there was an optional challenge of 
trying to solve the crawler game, in which we try to teach a machine 
to walk properly. For this the DDPG won't be sufficient, therefore we needed 
to use something else like a PPO algorithm to enhance the learning fase with multiple agents.

## Methodology

I started by taking the notes from the [Udacity class - Reinforced Learning](https://classroom.udacity.com/nanodegrees/nd893/dashboard/overview)
and implementing a simple DDPG for the Reacher problem.
It worked very good, for both the Single and Multiple agent.

After successfully running and passing the criteria, I moved into introducing [Optuna](https://optuna.org/) to try to aid in finding the best parameteres.
I did notice that the HyperParameters make an improvement, but not so significant 
for the case of multiple agents.

Having successfully passed these, I decided to try to take a chance at solving the Crawler challenge.
There I found that the DDPG isn't enough, or at least too slow for solving the Crawler challenge.

Therefore I tought on exploring the power of [PPO](https://arxiv.org/pdf/1707.06347.pdf).
Found online several implementations of the algorithm, some a bit more complex than others like 
[this example from Towards Data Science](https://towardsdatascience.com/a-graphic-guide-to-implementing-ppo-for-atari-games-5740ccbe3fbc?gi=1e7667c5ac9d) 
or other [github example from kotsonis](https://github.com/kotsonis/ddpg-crawler) or [from ShangtongZhang](https://github.com/ShangtongZhang/DeepRL). <br>
Finaly I decided to readapt the most [simple version from JacobXPX](https://github.com/JacobXPX/Crawler_using_PPO)


## Learning Algorithm - Reacher

The learning is done by the [Agent class](https://github.com/joao-d-oliveira/RL-RobotArm/blob/master/utils/agent_reacher_ddpg.py),
together with the [Model class](https://github.com/joao-d-oliveira/RL-RobotArm/blob/master/utils/model_reacher.py) which represents the local and target 
Network. In order to ease the training, the agent uses a 
[ReplayBuffer class](https://github.com/joao-d-oliveira/RL-RobotArm/blob/master/utils/agent_reacher_ddpg.py#L169)
as well as a [Ornstein-Uhlenbeck process class](https://github.com/joao-d-oliveira/RL-RobotArm/blob/master/utils/agent_reacher_ddpg.py#L207).

### Agent Details

The agent actions and learning are defined at [utils/agent_reacher_ddpg.py](https://github.com/joao-d-oliveira/RL-RobotArm/blob/master/utils/agent_reacher_ddpg.py).

The learning takes bellow initial parameters that guide the training:
```
Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
            num_agents (int): number of how many agents are training
            buffer_size (int): memory buffer size for ReplayBuffer
            batch_size (int): batch size for sampling to learn
            gamma (float): discount factor
            tau (float): interpolation parameter
            lr_actor (float): learning rate for Actor NN
            lr_critic (float): learning rate for Critic NN
            fc1_units (int): first hidden layer size
            fc2_units (int): second hidden layer size
            weight_decay (float): decay for Layer 2
            noise_mu (float): average factor for Ornstein-Uhlenbeck noise
            noise_theta (float): theta factor calculating Ornstein-Uhlenbeck noise
            noise_sigma (float): initial sigma factor to calculate Ornstein-Uhlenbeck noise
            noise_sigma_min (float): minimum sigma factor to calculate Ornstein-Uhlenbeck noise
            noise_sigma_decay (float): % to update sigma to calculate Ornstein-Uhlenbeck noise
        """
```
#### Agent functions:


**Step Function** to save each action and respecitve experience (rewards, ...) and learn from that step: <br>
`def step(self, state, action, reward, next_state, done):`

**Act Function** which takes a state and returns accordingly the action as per current policy  <br>
`def act(self, state, add_noise=True)`

**Learn Function** which updates accordingly the networks (Actor and Critic) <br>
`def learn(self, experiences)`

**Soft Update Function** performs a soft update to the model parameters <br>
`def soft_update(self, local_model, target_model, tau):`

**Reset Function** which resets the noise class
`def reset(self):`

#### Agent Auxiliary variables

Besides the functions described above, the Agent also uses a set of Variables/Objects to help its functioning.<br>
Out of which, important to mention **memory** which is an object of [ReplayBuffer](https://github.com/joao-d-oliveira/RL-RobotArm/blob/master/utils/agent_reacher_ddpg.py#L169) class
as well as **self.actor_local** and **self.critic_local** which is an object of [Actor and Critic](https://github.com/joao-d-oliveira/RL-RobotArm/blob/master/utils/model_reacher.py).
Besides that, at each step there's also a [Noise element](https://github.com/joao-d-oliveira/RL-RobotArm/blob/master/utils/agent_reacher_ddpg.py#L207) added to help the agent.

### Model NeuralNetwork

The NeuralNetwork for both Actor and Critic (defined at [model_reacher.py](https://github.com/joao-d-oliveira/RL-RobotArm/blob/master/utils/model_reacher.py)) 
are composed by 3 initial Linear Layers. (made them flexbile to receive the hidden sizes via parameter) <br>
The 1st and last Layer are the same for both Actor and Critic:
```
self.fc1 = nn.Linear(state_size, fc1_units)
(...)
self.fc3 = nn.Linear(fc2_units, action_size)
```

But the 2nd Linear Layer differs size, given that the Critic needs to **"critize"**
the choices made by the Actor. For that, the critic has embeded the actions in its hidden unit.

```
Actor: self.fc2 = nn.Linear(fc1_units, fc2_units)
Critic: self.fc2 = nn.Linear(fc1_units+action_size, fc2_units)
```
This will obviously mean that the forward method is also different:

```
"""Build a critic (value) network that maps (state, action) pairs -> Q-values."""
xs = F.relu(self.fcs1(state))
x = torch.cat((xs, action), dim=1)
x = F.relu(self.fc2(x))
```

### ReplayBuffer 

Buffer used to store experiences which the Agent can learn from.<br>
The Buffer is initialized with the option to have **Prioritizd Experience Replay** of not and adjusted the methods accordingly.

#### ReplayBuffer methods

**Add Function** to add an experience to the buffer <br>
`def add(self, state, action, reward, next_state, done)`

**Sample Function** that takes out a sample of `batch_size` from the buffer.
Depending on whether PER is on or not, it checks the priorities and weights accordingly before sampling.

`def sample(self, beta=0.03)`

**Update Priorities Function** given a set of indexes from the buffer, updates accordingly with the new priorities.
This function is only called in the case of PER (Prioritized Experience Replay) which updates the priorities / weights of the experiences accordingly after going through an experience.

`def update_priorities(self, indx, new_prio)`

### Hyper-Parameters
The project has some parameters which we can tweak to improve the performance. <br>
They are the ones also used with Optuna to try to achieve a best result.

You can find them below, followed by their _search-space_ used by Optuna to find the optimal value.
    
* `'GAMMA': [0.85, 1],` # discount value used to discount each experiences, thought that going below 0.85 would turn the agent too "stubborn" to learn so made that limit
* `'TAU': [1, 6],` # value for interpolation: **attention** this value is them modified to 1e-VALUE, so the real range is between 1e-6, 1e-1
* `'LR_ACTOR': [3, 6],` # learning rate used for optimizer of gradient-descend: **attention** this value is them modified to 1e-VALUE, so the real range is between 1e-6, 1e-3
* `'LR_CRITIC': [3, 6],` # learning rate used for optimizer of gradient-descend: **attention** this value is them modified to 1e-VALUE, so the real range is between 1e-6, 1e-3
* `'FC1_UNITS': [32,64,128,256],` # Values for 1st Hidden Linear Layer   
* `'FC2_UNITS': [32,64,128,256],`# Values for 1st Hidden Linear Layer
* `'NOISE_SIGMA_MIN': [1, 6],` # lower limit of EPS (used for greedy approach): **attention** this value is them modified to 1e-VALUE, so the real range is between 1e-6, 1e-3
* `'NOISE_SIGMA_DECAY': [0.85, 1],` # value for which EPS (used for greedy approach) is multiplied accordingly to decrease until reaching the lower limit

#### Images from HyperParameters
Below are an example of the Hyper Parameters search from Optuna:
![Optuna Detail Runs][optuna1]
![Optuna Parameters Coordinates][optuna2]
![Optuna Parameters Importance][optuna3]

Below you can find the different results run by the agent.
As well the different results of rolling window, first time to hit the target, ending average, ...

### Individual Results

<table align="center">
        <tr><td>
<img src="https://github.com/joao-d-oliveira/RL-RobotArm/blob/master/Images/run_ddpg_baseline_mult.png?raw=true" alt="Scores plot - Baseline Multiple Agents Run" align="center" height="300" width="350" >
<img src="https://github.com/joao-d-oliveira/RL-RobotArm/blob/master/Images/run_ddpg_baseline_one.png?raw=true" alt="Scores plot - Baseline Single Agent Run" align="center" height="300" width="350" >
<img src="https://github.com/joao-d-oliveira/RL-RobotArm/blob/master/Images/run_ddpg_hyperparameters_mult.png?raw=true" alt="Scores plot - HyperParamenters Run" align="center" height="300" width="350" >

</table>
        

### Plotted rolling windows

![Rolling Averages][Rolling]<br>


## Conclusion

As you can see from the 
[charts](https://github.com/joao-d-oliveira/RL-RobotArm/blob/master/Report.md#plot-of-rewards) 
The HyperParameters out-perform the Baseline overall, 
and one of the things that it is clear is that the algorithm converges much faster with multiple agents.

## Ideas for the Future

There is always room for improvement. <br>
From the vast number of ideas, or options that one can do to improve the performance of the agent, 
will try to least some of them to give some food-for-tought.

Some ideas for the future:
— After solving the [Crawler challenge](https://github.com/joao-d-oliveira/RL-RobotArm#for-crawler-environment),
it was evident that the [PPO algorithm](https://arxiv.org/pdf/1707.06347.pdf) outperforms DDPG for 
multi-agents. So the [PPO algorithm](https://arxiv.org/pdf/1707.06347.pdf) could be used as well for the [Reacher Environment](https://github.com/joao-d-oliveira/RL-RobotArm#environment).
— Other algorithms could be tested as well, such as [A3C](https://arxiv.org/pdf/1602.01783.pdf), and [D4PG](https://openreview.net/pdf?id=SyZipzbCb)
— Another possibility would be to spend more time finding the right HyperParameters
— Could test further parameters for Noise methods; 

------

## Rubric / Guiding Evaluation
[Original Rubric](https://review.udacity.com/#!/rubrics/1890/view)

#### Training Code

| Criteria       		|     Meets Specifications	        			            | 
|:---------------------|:---------------------------------------------------------| 
| ✅ Training Code  | The repository includes functional, well-documented, and organized code for training the agent. |
| ✅ Framework  | The code is written in PyTorch and Python 3. |
| ✅ Saved Model Weights | The submission includes the saved model weights of the successful agent. |

#### README

| Criteria       		|     Meets Specifications	        			            | 
|:---------------------|:---------------------------------------------------------| 
| ✅ `README.md`  | The GitHub (or zip file) submission includes a `README.md` file in the root of the repository. |
| ✅ Project Details  | The README describes the the project environment details (i.e., the state and action spaces, and when the environment is considered solved). |
| ✅ Getting Started | The README has instructions for installing dependencies or downloading needed files. |
| ✅ Instructions | The README describes how to run the code in the repository, to train the agent. For additional resources on creating READMEs or using Markdown, see here and here. |


#### Report

| Criteria       		|     Meets Specifications	        			            | 
|:---------------------|:---------------------------------------------------------| 
| ✅ Report | The submission includes a file in the root of the GitHub repository or zip file 
(one of `Report.md`, `Report.ipynb`, or `Report.pdf`) that provides a description of the implementation. |
| ✅ Learning Algorithm | The report clearly describes the learning algorithm, along with the chosen hyperparameters. It also describes the model architectures for any neural networks. |
| ✅ Plot of Rewards | A plot of rewards per episode is included to illustrate that either: <br> * [version 1] the agent receives an average reward (over 100 episodes) of at least +30, or <br>* [version 2] the agent is able to receive an average reward (over 100 episodes, and over all 20 agents) of at least +30.<br><br>The submission reports the number of episodes needed to solve the environment. |
| ✅ Ideas for Future Work | The submission has concrete future ideas for improving the agent's performance. |

#### Bonus :boom:
* ✅ Include a GIF and/or link to a YouTube video of your trained agent!
* ✅ Solve Crawler problem 

