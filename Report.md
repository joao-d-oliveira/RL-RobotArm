[image1]: https://github.com/joao-d-oliveira/RL-RobotArm/blob/main/images/generated_run.gif?raw=true "Trained Agent"
[optuna1]: https://github.com/joao-d-oliveira/RL-RobotArm/blob/main/images/optuna_detail_runs.png?raw=true "Optuna Detail Run"
[optuna2]: https://github.com/joao-d-oliveira/RL-RobotArm/blob/main/images/optuna_parameters_coordinates.png?raw=true "Optuna Parameters Coordinates"
[optuna3]: https://github.com/joao-d-oliveira/RL-RobotArm/blob/main/images/optuna_parameters_importance.png?raw=true "Optuna Parameters Importance"
[rolling]: https://github.com/joao-d-oliveira/RL-RobotArm/blob/main/images/run_allrolling.png?raw=true "All Rolling Averages"

# AI Agent 🤖 to Control a Mechanical Arm 🦾 [**Report**]

------

![Trained Agent with DQN][image1]<br>
Image of player being trained on Vanilla DQN
## Introduction

## Methodology


## Learning Algorithm

### Agent Details

#### Agent functions:

#### Agent Auxiliary variables

### Model NeuralNetwork

### ReplayBuffer 

#### ReplayBuffer methods

### Hyper-Parameters


#### Images from HyperParameters
Below are an example of the Hyper Parameters search from Optuna:
![Optuna Detail Runs][optuna1]
![Optuna Parameters Coordinates][optuna2]
![Optuna Parameters Importance][optuna3]

## Plot of Rewards


### Individual Results
 

### Plotted rolling windows

![Rolling Averages][Rolling]<br>

## Conclusion

## Ideas for the Future

------

## Rubric / Guiding Evaluation
[Original Rubric](https://review.udacity.com/#!/rubrics/1890/view)

#### Training Code

| Criteria       		|     Meets Specifications	        			            | 
|:---------------------|:---------------------------------------------------------| 
| ❗ Training Code  | The repository includes functional, well-documented, and organized code for training the agent. |
| ❗ Framework  | The code is written in PyTorch and Python 3. |
| ❗ Saved Model Weights | The submission includes the saved model weights of the successful agent. |

#### README

| Criteria       		|     Meets Specifications	        			            | 
|:---------------------|:---------------------------------------------------------| 
| ✅ `README.md`  | The GitHub (or zip file) submission includes a `README.md` file in the root of the repository. |
| ❗ Project Details  | The README describes the the project environment details (i.e., the state and action spaces, and when the environment is considered solved). |
| ❗ Getting Started | The README has instructions for installing dependencies or downloading needed files. |
| ❗ Instructions | The README describes how to run the code in the repository, to train the agent. For additional resources on creating READMEs or using Markdown, see here and here. |


#### Report

| Criteria       		|     Meets Specifications	        			            | 
|:---------------------|:---------------------------------------------------------| 
| ✅ Report | The submission includes a file in the root of the GitHub repository or zip file 
(one of `Report.md`, `Report.ipynb`, or `Report.pdf`) that provides a description of the implementation. |
| ❗ Learning Algorithm | The report clearly describes the learning algorithm, along with the chosen hyperparameters. It also describes the model architectures for any neural networks. |
| ❗ Plot of Rewards |A plot of rewards per episode is included to illustrate that either:

* [version 1] the agent receives an average reward (over 100 episodes) of at least +30, or
* [version 2] the agent is able to receive an average reward (over 100 episodes, and over all 20 agents) of at least +30.

The submission reports the number of episodes needed to solve the environment. |
| ❗ Ideas for Future Work | The submission has concrete future ideas for improving the agent's performance. |

#### Bonus :boom:
* ✅ Include a GIF and/or link to a YouTube video of your trained agent!
* ❗ Write a blog post explaining the project and your implementation!

