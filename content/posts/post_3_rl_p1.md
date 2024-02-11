---
title: "The K-Arm Bandit Problem - Part One"
date: 2024-02-11
draft: false
math: true
ShowToc: true
---
# Introduction

In this blog we are going to explore and solve a basic problem in the reinforcement learning paradigm; the k-arm bandit problem. It states that, given k-arms or levers you can pull anyone at any timestep $t$ and reap a reward $r_k$ corresponding to the lever. Our goal is to maximize the cumulative reward $R$ over a sequence of timesteps $T$.  

## Prerequisites 
In the section above, the problem has been defined on an abstract level, so let's delve deeper and know more about the problem and some basics of reinforcement learning.

**Agent**: An agent is a virtual entity that interacts with environment to achieve our goals.
**Action**:  An action is a decision taken by an agent based on the environment conditions at any time $t$.
**Reward**: The reward is a numeric value you earned by the agent when it takes an action $a$.
**Policy**: The policy $\pi$ is the directive according to which the agent takes action in an environment. 
**Expected Return**: It is the mean reward gained when we take the action $a$ at time $t$.  It is also said to be the value of that action  and often denoted as $Q(a)$. 

$Q(a) =E[R_t | A_t = a]$,  its just the expected value of Reward $R_t$ which we get after taking action $a$. Based on the value of any action we choose the best possible action. The easiest way is to take the action with the highest value. 

Now that we know the basics about reinforcement learning we will move ahead and generalize it to our problem.
-   **Agents and Environment:**  We have an agent (the decision-maker) interacting with an environment consisting of k "arms" (think of these as slot machines).
-   **Actions:**  At each time step (t), the agent chooses one arm to pull.
-   **Rewards:**  When an arm is pulled, the environment generates a reward, typically drawn from an unknown probability distribution specific to that arm.
-   **Goal:**  The agent's goal is to maximize the cumulative reward received over a series of pulls.

We are well covered on all the basics, however this is a very abstract and minimal explanation to get started. 

## Action-Values
 The value of any action as defined earlier is the expected  reward $R_t$ which we get after taking action $a$. As we do not have prior information on the true action-value function $Q(a)$ we estimate it. The simplest way is to estimate it with the method known as ***sample-averages***.  The sample average is just the running mean of the rewards reaped by taking action a before the timestep $t$. It is given by,
 
 $Q_t(a)=\frac{\sum{r_t | A_t = a}}{n_a}$

Here,  $Q_t(a)$ is the estimate of the true action-value function  $Q(a)$ and we sum over all the rewards we accumulate before time $t$ for action $a$ divided by $n_a$; the number of times the action $a$ is taken before the time $t$. This estimation holds true by the law of large number that states that,

> The average of the results obtained from a large number of independent and identical random samples converges to the true value.

We will use this to our benefit to estimate the action-value function and use it in the approaches to solve the bandit problem. 

## Approach 1: The Greedy Policy  
In the greedy method is the most simplistic approach to solving the bandit problem. As the name suggests the agent will act according to a greedy policy and will only choose the action that has the highest action-value at the timestep $t$.  The simplistic approach align perfectly with our aim to achieve the maximum expected return over some series of timesteps $T$ by always choosing the **seemingly** optimal action. 

#### Algorithm 
The algorithm below describes how we are going to implement the k-arm bandit problem. This brief algorithm can be used with any policy that we want to use. In this first case we will use the greedy policy.
```
1.  Initialization:
    -   Set up parameters such as the number of bandit arms (k), steps per trial, and the number of trials.
    -   Initialize lists to store the true values of actions (q_a_list) and the rewards obtained (rewards_list).
    
2.  Calculate Value Function:
    -   Define a method to calculate the value of an action using the sample-average method, considering the sum of rewards and the number of times the action has been taken.
    
3.  Trial Execution:
    -   For each trial:
        -   Initialize dictionaries to track the sum of rewards for each action and the number of times each action is taken.
        -   Generate random values for the true values of actions and corresponding rewards.
        -   Record the true values of actions and rewards.
        -   For each step within the trial:
            -   Calculate the estimated values of actions (Q(a)_t) using the sample-average method.
            -   Choose an action based on a given policy (provided as an argument).
            -   Record whether the chosen action is optimal.
            -   Update the sum of rewards and the number of times the chosen action has been taken.
            
4.  Simulation Execution:
    -   Print initial conditions.
    -   Execute multiple trials, each comprising multiple steps:
        -   Record whether each step's chosen action is optimal.
        -   Record the actions taken and the estimated values of actions.
      
5.  Return Results:
    -   Return arrays containing information about the optimality of actions, the actions taken, and the estimated values of actions throughout the simulation.
```

**Step 1: Initialization** 
We define a class Bandit where we initialize the environment; take the information about the number of bandit arms (k), steps per trial, and the number of trials. - Initialize lists to store the true values of actions (q_a_list) and the rewards obtained (rewards_list).
```
import  numpy  as  np
import  tqdm
np.random.seed(101)

class  Bandit:
	def  __init__(self, k:int, steps:int, trials:int):
		#number of arms
		self.k  =  k 
		##timesteps per trial
		self.steps  =  steps 
		##number of trials
		self.n_trials  =  trials 
		#### Optional ####
		#Store originial q_a at for every trial
		self.q_a_list  = [] 
		#Store original rewards in every trial
		self.rewards_list  = [] 
```

Remember to always set a random seed while sampling from numpy as it will help in reproducing the results. 

**Step 2: Calculate Value Function**
We are going to use the sample average method we defined early. Our implementation would assume that the user has already stored the sum of the rewards collected when action $a$ is being taken prior to the timestep $t$.
```
def  calcaulate_value(self, sum, n):
	# handling the case where actions haven't been taken before; n=0
	if  n==0:
		return  0
	return  sum/n
``` 
Remember that $n$ is the number of times the action $a$ is being taken prior to the timestep $t$.

**Step 3: Trial Execution**
The trial execution is the heart of the program, it is where our agent interacts with the environment and learns from the experience it gains. There can be $N_{trials}$ and in each trial there will be a series of $T$ timesteps($t_1, t_2, ... ,t_T$).
```
def  trial(self, policy):
	## Initiate dictionaries to calculate sum of rewards for each action/bandit
	sum_action_reward  = {k:v  for  k,v  in  zip(range(self.k),[0]*self.k)}
	## Initiate dictionary for keeping record of number of times an action is taken
	action_taken  = {k:v  for  k,v  in  zip(range(self.k),[0]*self.k)}
	## Initialize the actual values for actions
	q_a  =  np.array([np.random.standard_normal(size=self.k)]).flatten()
	optimal_action  =  np.argmax(q_a)
	## Initialize rewards based on the actual values of action
	rewards  =  np.array([np.random.normal(loc=mean) for  mean  in  q_a]).flatten()
	## For record keeping
	self.q_a_list.append(q_a)
	self.rewards_list.append(rewards)
	is_optimal  = []
	q_t_list  = []
	for  _  in  range(self.steps):
		## We will use the sample-average method to calculate Q(a)_t
		#Calculate the q-value for all actions prior step t
		Q_a__t  = [self.calcaulate_value(sum=item[0],n=item[1]) for  item  in
					zip(sum_action_reward.values(),action_taken.values())]
		#select an action according to the policy
		action, q_t  =  policy(q_vals=Q_a__t)
		q_t_list.append(q_t)
		if  action  ==  optimal_action:
			is_optimal.append(1)
		else:
			is_optimal.append(0)
		#Add to the sum of the rewards for that action
		#and the number of times the action is taken
		sum_action_reward[action] +=  rewards[action]
		action_taken[action]+=1
	return  is_optimal, action_taken, np.array(q_t_list)
```

**Step 4: Simulation Execution**
Here, we bring the first three steps together and implement the logic to run all $N_{trials}$ and store their results. This is going to be the function the user will interact with. 

```
def  simulate(self, policy):
	print("""
	Initial conditions:
	Number of Trials = {trials}
	Number of Steps per Trial = {steps}
	NUmber of Bandit Arms = {k}
	""".format(trials=self.n_trials, steps=self.steps, k=self.k))

	is_optimals  =  np.zeros(shape=(self.n_trials,self.steps))
	actions  = {}
	q_matrix  =  np.zeros(shape=(self.n_trials,self.steps))
	
	for  t  in  tqdm.tqdm(range(self.n_trials),colour="blue"):
		s, a, q_t  =  self.trial(policy=policy)
		is_optimals[t, :] =  s
		actions[t] =  a
		q_matrix[t, :] =  q_t
	
	return  is_optimals, actions, q_matrix
```
Awesome! Till here we have completed to create the environment and how it will work. However, we still have one thing missing; the brain of the agent, the policy! As we are going to use a greedy policy here, it is straightforward to implement it. The logic is to look over the action-values of all the actions at time $t$ and choose the one with the highest action-value.

```
def  greedy_policy(q_vals:list):
	"""
	The function chooses the action with the highest value.
	Params
	------
	q_vals: The list of value of each action, the index corresponds to the arm
	Returns
	-------
	action, value
	"""
	index  =  np.argmax(q_vals)
	return  index, q_vals[index]
```
Now we are all set to perform our experiments,
```
#Using the Bandit class defined in environment.py to make our env
#with the following specs,
#1. Trials, is the number of experiments or episodes we want to conduct
#2. Steps are the number of timesteps taken in each episode
#3. k, is the number of arms of the Bandit

trials  =  2000
steps  =  1000
k = 10
env  =  Bandit(trials=trials,
			   steps=steps,
			   k=k)

#Get the reward matrix for each trial, the row represent a unique trial and
#the cols represent expected return at each timestep
#This is one is with using greedy policy

is_optimal,_, rewards  =  env.simulate(policy=greedy_policy)
```
**Output:**
```
Initial conditions: Number of Trials = 2000 
Number of Steps per Trial = 1000 
NUmber of Bandit Arms = 10
100%|██████████| 2000/2000 [01:11<00:00, 27.79it/s]
```
### Results
The agent learnt navigate through our environment and achieve its goals; to maximize the cumulative reward received over a series of pulls. Let us have a look over the results and how well did our agent performs. The first result would be the average reward it receives over each timestep($t_1, t_2, ... ,t_T$) averaged over $N_{trials}$.

![Mean Reward Per Step](https://i.postimg.cc/8zV614CW/image.png)


The above graph shows that at each step over $N_{trials}$  our agent reaped the same reward. However this contradicts from the learning perspective doesn't it? If the agent was supposed to learn from the experience then why does it always get the same reward at every timestep. Didn't it learned anyway to pull the right levers to get bet better step by step ? What do you think is wrong here ? 

![Optimal Actions Taken](https://i.postimg.cc/rz6Kw1SK/image.png)

Also, looking at this plot which conveys how many times out the $N_{trial}$ times does our agent took the right/optimal action. The optimal action is the action which have the highest true action-value. The true action-values for each trial is defined in the **Step 2** above. As one might notice, only 20% of the times our agent takes the actual optimal action which also conveys that it ***doesn't learn and it performs poorly***. 

**Why?**
There is a simple explanation to all the questions you have, in any trial the initial state of any estimate of action value is 0; $Q_t(a)=0$. According to our policy, we choose the action with highest action-value. Here, at time $t_1$ we have estimated action-values as a list of zeros; $Q_1=[0, 0, 0, ... , 0]$. Hence, our policy can choose any action initially and according to numpy's argmax implementation, it would be the first action. Suppose we take the first action at $t_1$ then we update $n_a[1]+=1$ and sum of the rewards for action $a$ as $R_a[1]+=r_a$.  Furthermore, at time $t_2$, action-values $Q_2$ will be $[r_{a},0, 0, ..., 0]$, hence the first action will be taken again and this will go on until the timestep $T$. This situation in reinforcement is known as **exploitation** ; the agent takes advantage of the knowledge(the greedy action) and it doesn't explore other actions available. We will discuss exploitation and exploration more in the next blog and see solutions that employ exploration.
