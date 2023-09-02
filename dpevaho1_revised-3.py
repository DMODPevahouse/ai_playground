#!/usr/bin/env python
# coding: utf-8

# # Module 11 - Programming Assignment
# 
# ## Directions
# 
# 1. Change the name of this file to be your JHED id as in `jsmith299.ipynb`. Because sure you use your JHED ID (it's made out of your name and not your student id which is just letters and numbers).
# 2. Make sure the notebook you submit is cleanly and fully executed. I do not grade unexecuted notebooks.
# 3. Submit your notebook back in Blackboard where you downloaded this file.
# 
# *Provide the output **exactly** as requested*

# ## Reinforcement Learning with Value Iteration
# 
# These are the same maps from Module 1 but the "physics" of the world have changed. In Module 1, the world was deterministic. When the agent moved "south", it went "south". When it moved "east", it went "east". Now, the agent only succeeds in going where it wants to go *sometimes*. There is a probability distribution over the possible states so that when the agent moves "south", there is a small probability that it will go "east", "north", or "west" instead and have to move from there.
# 
# There are a variety of ways to handle this problem. For example, if using A\* search, if the agent finds itself off the solution, you can simply calculate a new solution from where the agent ended up. Although this sounds like a really bad idea, it has actually been shown to work really well in video games that use formal planning algorithms (which we will cover later). When these algorithms were first designed, this was unthinkable. Thank you, Moore's Law!
# 
# Another approach is to use Reinforcement Learning which covers problems where there is some kind of general uncertainty in the actions. We're going to model that uncertainty a bit unrealistically here but it'll show you how the algorithm works.
# 
# As far as RL is concerned, there are a variety of options there: model-based and model-free, Value Iteration, Q-Learning and SARSA. You are going to use Value Iteration.

# ## The World Representation
# 
# As before, we're going to simplify the problem by working in a grid world. The symbols that form the grid have a special meaning as they specify the type of the terrain and the cost to enter a grid cell with that type of terrain:
# 
# ```
# token   terrain    cost 
# .       plains     1
# *       forest     3
# ^       hills      5
# ~       swamp      7
# x       mountains  impassible
# ```
# 
# When you go from a plains node to a forest node it costs 3. When you go from a forest node to a plains node, it costs 1. You can think of the grid as a big graph. Each grid cell (terrain symbol) is a node and there are edges to the north, south, east and west (except at the edges).
# 
# There are quite a few differences between A\* Search and Reinforcement Learning but one of the most salient is that A\* Search returns a plan of N steps that gets us from A to Z, for example, A->C->E->G.... Reinforcement Learning, on the other hand, returns  a *policy* that tells us the best thing to do in **every state.**
# 
# For example, the policy might say that the best thing to do in A is go to C. However, we might find ourselves in D instead. But the policy covers this possibility, it might say, D->E. Trying this action might land us in C and the policy will say, C->E, etc. At least with offline learning, everything will be learned in advance (in online learning, you can only learn by doing and so you may act according to a known but suboptimal policy).
# 
# Nevertheless, if you were asked for a "best case" plan from (0, 0) to (n-1, n-1), you could (and will) be able to read it off the policy because there is a best action for every state. You will be asked to provide this in your assignment.

# We have the same costs as before. Note that we've negated them this time because RL requires negative costs and positive rewards:

# In[1]:


costs = { '.': -1, '*': -3, '^': -5, '~': -7}
costs


# and a list of offsets for `cardinal_moves`. You'll need to work this into your **actions**, A, parameter.

# In[2]:


cardinal_moves = [(0,-1), (1,0), (0,1), (-1,0)]


# In[3]:


reward = 10


# For Value Iteration, we require knowledge of the *transition* function, as a probability distribution.
# 
# The transition function, T, for this problem is 0.70 for the desired direction, and 0.10 each for the other possible directions. That is, if the agent selects "north" then 70% of the time, it will go "north" but 10% of the time it will go "east", 10% of the time it will go "west", and 10% of the time it will go "south". If agent is at the edge of the map, it simply bounces back to the current state.
# 
# You need to implement `value_iteration()` with the following parameters:
# 
# + world: a `List` of `List`s of terrain (this is S from S, A, T, gamma, R)
# + costs: a `Dict` of costs by terrain (this is part of R)
# + goal: A `Tuple` of (x, y) stating the goal state.
# + reward: The reward for achieving the goal state.
# + actions: a `List` of possible actions, A, as offsets.
# + gamma: the discount rate
# 
# you will return a policy: 
# 
# `{(x1, y1): action1, (x2, y2): action2, ...}`
# 
# Remember...a policy is what to do in any state for all the states. Notice how this is different than A\* search which only returns actions to take from the start to the goal. This also explains why reinforcement learning doesn't take a `start` state.
# 
# You should also define a function `pretty_print_policy( cols, rows, policy)` that takes a policy and prints it out as a grid using "^" for up, "<" for left, "v" for down and ">" for right. Use "x" for any mountain or other impassable square. Note that it doesn't need the `world` because the policy has a move for every state. However, you do need to know how big the grid is so you can pull the values out of the `Dict` that is returned.
# 
# ```
# vvvvvvv
# vvvvvvv
# vvvvvvv
# >>>>>>v
# ^^^>>>v
# ^^^>>>v
# ^^^>>>G
# ```
# 
# (Note that that policy is completely made up and only illustrative of the desired output). Please print it out exactly as requested: **NO EXTRA SPACES OR LINES**.
# 
# * If everything is otherwise the same, do you think that the path from (0,0) to the goal would be the same for both A\* Search and Q-Learning?
# * What do you think if you have a map that looks like:
# 
# ```
# ><>>^
# >>>>v
# >>>>v
# >>>>v
# >>>>G
# ```
# 
# has this converged? Is this a "correct" policy? What are the problems with this policy as it is?
# 

# In[4]:


def read_world(filename):
    result = []
    with open(filename) as f:
        for line in f.readlines():
            if len(line) > 0:
                result.append(list(line.strip()))
    return result


# ---

# In[5]:


from operator import itemgetter
from copy import deepcopy
import random
import pprint
max_iterations = 10000
epsilon = .1


# <a id="create_value_world"></a>
# ## create_value_world
# 
# The purpose of this function is to create an array of 0s that is the same size and same length as the world being used that will be adjusted to hold the state values **Used by**: [value_iteration](#value_iteration), [determinstic_value_iteration](#deterministic_value_iteration)
# 
# * **world**: list[list]: the world that is the basis for the creation of the array.
# * **goal**: tuple: the location of the goal
# * **reward**: int: the reward of the goal
# 
# 
# **returns**: list[list]: this is the array that is desired, which is what is returned

# In[6]:


def create_value_world(world: list[list], goal: tuple, reward: int):
    result = []
    for y in range(len(world)):
        result.append([])
        for x in range(len(world[y])):
            result[y].append(0)
    x, y = goal
    result[y][x] = reward
    return result


# In[7]:


unit_world = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
unit_actions = [1, 1, 1]
unit_test = create_value_world(unit_world, (2, 2), 5)
assert unit_test == [[0, 0, 0], [0, 0, 0], [0, 0, 5]]
unit_test = create_value_world(unit_world, (1, 1), 10)
assert unit_test == [[0, 0, 0], [0, 10, 0], [0, 0, 0]]
unit_test = create_value_world(unit_world, (2, 0), 100)
assert unit_test == [[0, 0, 100], [0, 0, 0], [0, 0, 0]]


# <a id="deterministic_look_ahead"></a>
# ## deterministic_look_ahead
# 
# This function was the original test function for looking ahead on what a specific action will result in a determinitistic value_iteration implementation. It took in the current state and location of the world to compare what the value would be a of a passed action **Used by**: [deterministic_value_iteration](#deterministic_value_iteration)
# 
# * **action**: list[tuple]: The action being investigated
# * **costs**: dict: the costs of the terrain
# * **world**: list[list]: the world that is the basis for the creation of the array.
# * **goal**: tuple: the location of the goal
# * **reward**: int: the reward of the goal
# * **x**: int: x value of the world location 
# * **y**: int: y value of the world location
# * **gamma**: float: the value being multiplied to determine each reinforcement learning value.
# * **value_world**: the world that holds the current values for each location of the world
# 
# **returns**: float: returns the total cost of the equation that will be used to figure out the next steps

# In[8]:


def deterministic_look_ahead(action: list[tuple], costs: dict, world: list[list], goal: tuple, reward: int, x: int, y: int, gamma: float, value_world: list[list]):
    goal_x, goal_y = goal
    if (y, x) == (goal_x, goal_y):
        return reward
    elif y+action[1] > (len(world[0]) - 1) or  y+action[1] < 0 or x+action[0] > (len(world) - 1) or  x+action[0] < 0:
        return -1000000
    elif world[x+action[0]][y+action[1]] == 'x' or world[x][y] == 'x':
        return -1000000
    else:
        current_cost = costs[world[x][y]]
        total = current_cost + (gamma * value_world[x+action[0]][y+action[1]])
        return total


# In[9]:


unit_world = read_world( "small.txt")
unit_value_world = create_value_world(unit_world, (len(unit_world[0]) - 1, len(unit_world) - 1), 5)
unit_test = deterministic_look_ahead((0, 1), costs, unit_world, (len(unit_world[0]) - 1, len(unit_world) - 1), 5, 0, 0, .9, unit_value_world)
assert unit_test == -1
unit_test = deterministic_look_ahead((0, 1), costs, unit_world, (len(unit_world[0]) - 1, len(unit_world) - 1), 5, len(unit_world) - 1,len(unit_world[0]) - 1, .9, unit_value_world)
assert unit_test == 5
unit_test = deterministic_look_ahead((0, 1), costs, unit_world, (len(unit_world[0]) - 1, len(unit_world) - 1), 5, 1, 1, .9, unit_value_world)
assert unit_test == -3


# <a id="look_ahead"></a>
# ## look_ahead
# 
# This function is to take the action that will be looked into to figure out which action would be the next best step to take for the algorithm to learn how to traverse the terrain **Used by**: [value_iteration](#value_iteration).
# 
# * **action**: list[tuple]: The action being investigated
# * **costs**: dict: the costs of the terrain
# * **world**: list[list]: the world that is the basis for the creation of the array.
# * **goal**: tuple: the location of the goal
# * **reward**: int: the reward of the goal
# * **x**: int: x value of the world location 
# * **y**: int: y value of the world location
# * **gamma**: float: the value being multiplied to determine each reinforcement learning value.
# * **value_world**: the world that holds the current values for each location of the world
# 
# **returns**: documentation of the returned value and type.

# In[10]:


def look_ahead(action: list[tuple], costs: dict, world: list[list], goal: tuple, reward: int, x: int, y: int, gamma: float, value_world: list[list]):
    goal_x, goal_y = goal
    if (y, x) == (goal_x, goal_y):
        return reward
    elif y+action[1] > (len(world[0]) - 1) or  y+action[1] < 0 or x+action[0] > (len(world) - 1) or  x+action[0] < 0:
        return -10000
    elif world[x+action[0]][y+action[1]] == 'x' or world[x][y] == 'x':
        return -10000
    else:
        total = (gamma * value_world[x+action[0]][y+action[1]])
        return total


# In[11]:


unit_world = read_world( "small.txt")
unit_value_world = create_value_world(unit_world, (len(unit_world[0]) - 1, len(unit_world) - 1), 5)
unit_test = look_ahead((0, 1), costs, unit_world, (len(unit_world[0]) - 1, len(unit_world) - 1), 5, 0, 0, .9, unit_value_world)
assert unit_test == 0.0
unit_test = look_ahead((0, 1), costs, unit_world, (len(unit_world[0]) - 1, len(unit_world) - 1), 5,  len(unit_world) - 1, len(unit_world[0]) - 1, .9, unit_value_world)
assert unit_test == 5
unit_test = look_ahead((0, -1), costs, unit_world, (len(unit_world[0]) - 1, len(unit_world) - 1), 5, 0, 0, .9, unit_value_world)
assert unit_test == -10000


# <a id="decide_action"></a>
# ## decide_action
# 
# Here is where the actions that have been looked into will be decided, with safe guards to preventing the algorithm from going out of the terrain and using the random variable to decide which to select **Used by**: [select_action](#select_action)
# 
# * **world**: list[list]: the world that is the basis for the creation of the array.
# * **action_list**: list: this contains, in order, a list of actions that have been investigated to see which is best
# * **x**: int: x value of the world location 
# * **y**: int: y value of the world location
# * **costs**: dict: the costs of the terrain
# * **goal**: tuple: the location of the goal
# * **reward**: int: the reward of the goal
# 
# 
# **returns**: tuple, float: returns a valid action and float from previous equations based off of state

# In[12]:


def decide_action(world: list[list], action_list: list, x: int, y: int, costs: dict, goal: tuple, reward: int):
    if (y, x) == goal:
        return (0, 0), reward
    if y+action_list[0][0][1] <= (len(world[0]) - 1) and y+action_list[0][0][1] >= 0 and x+action_list[0][0][0] <= (len(world) - 1) and x+action_list[0][0][0] >= 0 and world[x+action_list[0][0][0]][y+action_list[0][0][1]] != 'x': 
        return action_list[0]
    elif y+action_list[1][0][1] <= (len(world[0]) - 1) and y+action_list[1][0][1] >= 0 and x+action_list[1][0][0] <= (len(world) - 1) and x+action_list[1][0][0] >= 0 and world[x+action_list[1][0][0]][y+action_list[1][0][1]] != 'x': 
        return action_list[1]
    elif y+action_list[2][0][1] <= (len(world[0]) - 1) and y+action_list[2][0][1] >= 0 and x+action_list[2][0][0] <= (len(world) - 1) and x+action_list[2][0][0] >= 0 and world[x+action_list[2][0][0]][y+action_list[2][0][1]] != 'x': 
        return action_list[2]
    else: 
        return action_list[3]


# In[13]:


unit_world = read_world( "small.txt")
unit_action_list = [[(0 ,1), 1], [(1, 0), 2], [(-1, 0), 3], [(0, -1), 4]]
unit_test = decide_action(unit_world, unit_action_list, 0, 0, costs, (9, 9), 10)
assert unit_test == [(0, 1), 1]
unit_test = decide_action(unit_world, unit_action_list, 0, 0, costs, (9, 9), 10)
assert unit_test == [(0, 1), 1]
unit_test = decide_action(unit_world, unit_action_list, 1, 1, costs, (9, 9), 10)
assert unit_test == [(0, 1), 1]


# <a id="select_action"></a>
# ## select_action
# 
# This is the function that will be called to gather all the information needed to take the actions that will be investigated, look into each specific action, then return the best valid action to the value iteration function **Uses**: [look_ahead](#look_ahead), [decide_action](#decide_action) **Used by**: [value_iteration](#value_iteration)
# 
# 
# * **world**: list[list]: the world that is the basis for the creation of the array.
# * **costs**: dict: the costs of the terrain
# * **actions**: list[tuple]: The actions being investigated
# * **goal**: tuple: the location of the goal
# * **reward**: int: the reward of the goal
# * **x**: int: x value of the world location 
# * **y**: int: y value of the world location
# * **gamma**: float: the value being multiplied to determine each reinforcement learning value.
# * **value_world**: the world that holds the current values for each location of the world
# 
# 
# **returns**: tuple, float: this returns the action that has been decided and the amount that was left with what was decided 

# In[14]:


def select_action(world: list[list], costs: dict, actions: list[tuple], goal: tuple, reward: int, gamma: float, x: int, y: int, value_world: list[list]):
    action_list = []
    for main_action in actions:
        sum = 0
        if world[x][y] == 'x': sum = -10000
        else: sum = costs[world[x][y]] + look_ahead(main_action, costs, world, goal, reward, x, y, gamma, value_world)
        for action in actions:
            if action == main_action: continue
            else:
                q = look_ahead(action, costs, world, goal, reward, x, y, 1-gamma, value_world)
                sum = sum + q
                value_world[x][y] = sum
        action_list.append([main_action, sum])
    action_list = sorted(action_list, key=itemgetter(1), reverse=True)
    decided_action = decide_action(world, action_list, x, y, costs, goal, reward)
    return decided_action


# In[15]:


unit_world = read_world( "small.txt")
unit_value_world = create_value_world(unit_world, (len(unit_world[0]) - 1, len(unit_world) - 1), 5)
unit_test = select_action(unit_world, costs, cardinal_moves, (len(unit_world[0]) - 1, len(unit_world) - 1), 5, .9, 0, 0, unit_value_world)
assert unit_test == [(1, 0), -20001.0]
unit_test = select_action(unit_world, costs, cardinal_moves, (len(unit_world[0]) - 1, len(unit_world) - 1), 5, .9, 5, 5, unit_value_world)
assert unit_test == [(1, 0), -9996.5]
unit_test = select_action(unit_world, costs, cardinal_moves, (len(unit_world[0]) - 1, len(unit_world) - 1), 5, .9, 2, 2, unit_value_world)
assert unit_test == [(0, -1), -3.0] 


# <a id="determinisitic_value_iteration"></a>
# ## determinisitic_value_iteration
# 
# This was a test function to get a simplified version of deterministic value iteration working with low values to get an understanding of how value iteration works **Uses**: [deterministic_look_ahead](#deterministic_look_ahead), [create_value_world](#create_value_world) 
# 
# * **world**: list[list]: the world that is the basis for the creation of the array.
# * **costs**: dict: the costs of the terrain
# * **goal**: tuple: the location of the goal
# * **reward**: int: the reward of the goal
# * **actions**: list[tuple]: The actions being investigated
# * **gamma**: float: the value being multiplied to determine each reinforcement learning value.
# 
# **returns**: dict: the polict of every movement in the world

# In[16]:


def deterministic_value_iteration(world, costs, goal, rewards, actions, gamma):
    value_world, policy, delta = create_value_world(deepcopy(world), goal, rewards), {}, 0 
    while True:
        delta += 1
        for y in range(len(world[0])):
            for x in range(len(world)):
                temp_action_costs = []
                for action in actions:
                    temp_action_costs.append([action, deterministic_look_ahead(action, costs, world, goal, rewards, x, y, gamma, value_world)])
                temp_action_costs = sorted(temp_action_costs, key=itemgetter(1), reverse=True)
                value_world[x][y] = temp_action_costs[0][1]
                policy[(x, y)] = temp_action_costs[0][0]
                if world[x][y] == 'x':
                    policy[(x, y)] = 'x'
        if delta > max_iterations: break
    return policy


# no unit test, as it is an alternative solution to the problem

# <a id="value_iteration"></a>
# ## value_iteration
# 
# This is the main algorithm for value iteration, implemented in a schotastic fashion where it will take in the world state, costs, goal, reward, actions, and gamma to incorporate them into the desired mathmatical function with a small amount of randomness implemented in the equation to figure out which function will actually be called **Uses**: [select_action](#select_action), [create_value_world](#create_value_world) 
# 
# * **world**: list[list]: the world that is the basis for the creation of the array.
# * **costs**: dict: the costs of the terrain
# * **goal**: tuple: the location of the goal
# * **reward**: int: the reward of the goal
# * **actions**: list[tuple]: The actions being investigated
# * **gamma**: float: the value being multiplied to determine each reinforcement learning value.
# 
# **returns**: dict: the polict of every movement in the world

# In[17]:


def value_iteration(world, costs, goal, rewards, actions, gamma):
    value_world, policy, delta = create_value_world(deepcopy(world), goal, rewards), {}, 0 
    while True:
        delta += 1
        for y in range(len(world[0])):
            for x in range(len(world)):
                v_pre = value_world[x][y]
                next_action = select_action(world, costs, actions, goal, reward, gamma, x, y, value_world)
                value_world[x][y] = next_action[1]
                policy[(x, y)] = next_action[0]
                if world[x][y] == 'x':
                    policy[(x, y)] = 'x'
                v_after = value_world[x][y]
                if (y, x) == goal:
                    v_after = 10000
                    value_world[x][y] = rewards
        if delta > max_iterations: break
    return policy


# <a id="pretty_print_policy"></a>
# ## pretty_print_policy
# 
# The purpose of this function is to take the policy of the value_iteration function and print it in an easy to read fashion
# 
# * **cols**: list: amount of cols
# * **rows**: list: amount of rows
# * **policy**: dict: the values for the world for how to determine direction 
# * **goal**: tuple: the location of the goal
# 
# **returns**: documentation of the returned value and type.

# In[18]:


def pretty_print_policy( cols: list, rows: list, policy: dict, goal: tuple):
    directions = {(1, 0) : 'v', (0, 1) : '>', (-1, 0) : '^', (0, -1) : '<', 'x' : 'x', (0, 0) : 'H'}
    policy_printer = []
    for x in range(rows):
        row_printer = []
        for y in range(cols):
            if (y, x) == goal:
                 row_printer.append('H')
            else:
                row_printer.append(directions[policy[(x, y)]])
        policy_printer.append(row_printer)
    for item in policy_printer:
        print(*item, sep='')        


# ## Value Iteration
# 
# ### Small World

# In[19]:


small_world = read_world( "small.txt")


# In[20]:


goal = (len(small_world[0])-1, len(small_world)-1)
gamma = 0.9

small_policy = value_iteration(small_world, costs, goal, reward, cardinal_moves, gamma)


# In[21]:


cols = len(small_world[0])
rows = len(small_world)

#pretty_print_policy(cols, rows, test_policy, goal) I changed this to small_policy as that was what was ran top, hope that is ok
pretty_print_policy(cols, rows, small_policy, goal)


# ### Large World

# In[22]:


large_world = read_world( "large.txt")


# In[23]:


goal = (len(large_world[0])-1, len(large_world)-1) # Lower Right Corner FILL ME IN
gamma = 0.9

large_policy = value_iteration(large_world, costs, goal, reward, cardinal_moves, gamma)


# In[24]:


cols = len(large_world[0])
rows = len(large_world)

pretty_print_policy( cols, rows, large_policy, goal)


# ## Before You Submit...
# 
# 1. Did you provide output exactly as requested?
# 2. Did you re-execute the entire notebook? ("Restart Kernel and Rull All Cells...")
# 3. If you did not complete the assignment or had difficulty please explain what gave you the most difficulty in the Markdown cell below.
# 4. Did you change the name of the file to `jhed_id.ipynb`?
# 
# Do not submit any other files.

# My process of figuring out this problem was first going through how to do it deterministically, and my solution felt super simple, to the poiint that I felt like I did it wrong but the policy it created looked fairly close to what an actual solution would be, with a few exceptions that I could not figure out what it was doing wrong
# 
# I did run into issues with doing the schostastic version of it however. It seemed like no mater what I did the reward was not being selected. Was it how I implemented it or how I did the random portion? I have a feeling that the second part was it where I did not fully understand how to incorporate the random functionality into the selection. I know random is not the fully right answer but that it does prevent a deterministic outlook on the policy
# 
# I did end up keeping both variations of my value iteration algorithm, I did not touch the answer key up top (except for an incorrect variable which I hope is correct) which is the schotastic iteration like the instructions specified to implement with the randomness. Below this section is the determinstic implementation which I feel like I got mostly if not fulyl correct based off of what I was seeing in how it stayed away from the edges and the mountain terrain.
# 
# Thank you for your help and have a great day!

# EDITED AND REVISED
# 
# 
# 
# 
# 
# 
# 
# After implementing your recommendations I realized that schotastic was not talking about actually implementing where the algorithm would randomly select a direction, but just using calculations to. So after I fixed that, I believe my schotastic implementation is now correct and looks vastly superior to what the deterministic looks like below. Thank you for the advise and the opportunity to revise!

# In[25]:


goal = (len(small_world[0])-1, len(small_world)-1)
gamma = 0.9

small_policy = deterministic_value_iteration(small_world, costs, goal, reward, cardinal_moves, gamma)


# In[26]:


cols = len(small_world[0])
rows = len(small_world)

#pretty_print_policy(cols, rows, test_policy, goal) I changed this to small_policy as that was what was ran top, hope that is ok
pretty_print_policy(cols, rows, small_policy, goal)


# In[27]:


goal = (len(large_world[0])-1, len(large_world)-1) # Lower Right Corner FILL ME IN
gamma = 0.9

large_policy = deterministic_value_iteration(large_world, costs, goal, reward, cardinal_moves, gamma)


# In[28]:


cols = len(large_world[0])
rows = len(large_world)

pretty_print_policy( cols, rows, large_policy, goal)


# In[ ]:




