# Unity Banana Navigation

Goal of this project is to train agent to navigate and collect yellow bananas, in square world.

When doing this it can get reward of **+1** when yellow banana is collected and reward **-1** in case of blue one, so we can see that agent should avoid blue one while collecting yellow one.

State space has 37 dimensions and contains the agent's velocity plus ray-based perception of objects around agent's forward direction. Given this information, the agent has to learn how to best select actions. There are following four actions available:

**0** – move forward
**1** – move backward
**2** – turn left
**3** – turn right

**Details of given environment can be found below:**

```        Number of Visual Observations (per agent): 0```
```        Vector Observation space type: continuous```
```        Vector Observation space size (per agent): 37```
```        Number of stacked Vector Observation: 1```
```        Vector Action space type: discrete```
```        Vector Action space size (per agent): 4```

**Goal:**  average score of **+13** in **100** following episodes.

# Installation details:

- You have to have python 3.6 installed 
- You have to have environment of your choose activated
- You have to have Unity installed
- Copy repository to your computer and enter it

**Then run:**
```
pip install -r packages_needed.txt
pip -q install ./python
```
After this all thats left is open Jupyter Notebook, open **NavigationSolution.ipynb** and run it.

# File list:

**agent.py-** file contains agent that is used in this task.
**network_model.py** – file contains neural network that agent is using to solve environment
**priority_replay.py** – file contains implementation of sumTree plus priority experience replay
**NavigationSolution.ipynb** – file contains enviroment that runs unity/agent
**model_weights.pth** - pretrained model weights

